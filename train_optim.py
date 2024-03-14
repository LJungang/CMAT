import torch
import os
import time
import numpy as np
from tqdm import tqdm

from tools import save_tensor
from tools.transformer import mixup_transform
from tools.plot import VisualBoard
from tools.loader import dataLoader
from tools.parser import logger
from tools.solver import Cosine_lr_scheduler, Plateau_lr_scheduler, ALRS

scheduler_factor = {
    'plateau': Plateau_lr_scheduler,
    'cosine': Cosine_lr_scheduler,
    'ALRS': ALRS,
}


def attack(cfg, data_root, detector_attacker, save_name, args=None):
    def get_iter():
        return (epoch - 1) * len(data_loader) + index

    logger(cfg, args)
    data_sampler = None
    detector_attacker.init_universal_patch(args.patch)
    data_loader = dataLoader(data_root,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment='1' in cfg.DATA.AUGMENT,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler, shuffle=True)

    detector_attacker.gates = ['jitter', 'median_pool', 'rotate', 'p9_scale']
    if args.random_erase: detector_attacker.gates.append('rerase')

    p_obj = detector_attacker.patch_obj.patch
    optimizer = torch.optim.Adam([p_obj], lr=cfg.ATTACKER.START_LEARNING_RATE, amsgrad=True)
    scheduler = scheduler_factor[cfg.ATTACKER.scheduler](optimizer)
    detector_attacker.attacker.set_optimizer(optimizer)
    loss_array = []
    save_tensor(detector_attacker.universal_patch, f'{save_name}' + '.png', args.save_path)
    vlogger = None
    if not args.debugging:
        vlogger = VisualBoard(optimizer, name=args.board_name, new_process=args.new_process)
        detector_attacker.vlogger = vlogger
    ten_epoch_loss = 0
    for epoch in range(1, cfg.ATTACKER.MAX_EPOCH + 1):
        et0 = time.time()
        ep_loss = 0
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            # for index, (img_tensor_batch, img_tensor_batch2) in enumerate(tqdm(zip(data_loader, data_loader2), desc=f'Epoch {epoch}')):
            if vlogger: vlogger(epoch, get_iter())
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)
            if args.mixup:
                img_tensor_batch = mixup_transform(x1=img_tensor_batch)

            all_preds = detector_attacker.detect_bbox(img_tensor_batch)
            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0: continue

            loss = detector_attacker.attack(img_cv2, mode='optim')
            ep_loss += loss

        if epoch % 10 == 0:
            # patch_name = f'{epoch}_{save_name}'
            patch_name = f'{save_name}' + '.png'
            save_tensor(detector_attacker.universal_patch, patch_name, args.save_path)
            print('Saving patch to ', os.path.join(args.save_path, patch_name))

            if cfg.ATTACKER.scheduler == 'ALRS':
                ten_epoch_loss /= 10
                scheduler.step(ten_epoch_loss)
                ten_epoch_loss = 0

        et1 = time.time()
        ep_loss /= len(data_loader)
        ten_epoch_loss += ep_loss
        if cfg.ATTACKER.scheduler == 'plateau':
            scheduler.step(ep_loss)
        elif cfg.ATTACKER.scheduler != 'ALRS':
            scheduler.step()
        if vlogger:
            vlogger.write_ep_loss(ep_loss)
            vlogger.write_scalar(et1 - et0, 'misc/ep time')
        # print('           ep loss : ', ep_loss)
        loss_array.append(float(ep_loss))
    np.save(os.path.join(args.save_path, save_name + '-loss.npy'), loss_array)


if __name__ == '__main__':
    from tools.parser import ConfigParser
    from attack.attacker import UniversalAttacker
    import argparse
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-m', '--attack_method', type=str, default='optim')
    parser.add_argument('-cfg', '--cfg', type=str, default='optim.yaml')
    parser.add_argument('-n', '--board_name', type=str, default=None)
    parser.add_argument('-d', '--debugging', action='store_true')
    parser.add_argument('-s', '--save_path', type=str, default='./results/exp2/optim')
    parser.add_argument('-re', '--random_erase', action='store_true', default=False)
    parser.add_argument('-mu', '--mixup', action='store_true', default=False)
    parser.add_argument('-np', '--new_process', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_patch_name = args.cfg.split('.')[0] if args.board_name is None else args.board_name
    args.cfg = './configs/' + args.cfg

    print('-------------------------Training-------------------------')
    print('                       device : ', device)
    print('                          cfg :', args.cfg)

    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalAttacker(cfg, device)
    cfg.show_class_label(cfg.attack_list)
    data_root = cfg.DATA.TRAIN.IMG_DIR
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]
    attack(cfg, data_root, detector_attacker, save_patch_name, args)
