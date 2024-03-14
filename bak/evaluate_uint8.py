import argparse

import os
import torch.cuda
import yaml

from tools.loader import read_img_np_batch
from tools.utils import obj
from evaluate import UniversalPatchEvaluator


def save_detections(img_tensor, save_name):
    _, img_numpy_int8 = detector.unnormalize(img_tensor)
    evaluator.plot_boxes(img_numpy_int8, preds[0], save_name)
    print(f'save detections to {save_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str)
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-s', '--save', type=str)
    parser.add_argument('-d', '--data_root', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/military_data/JPEGImages')
    args = parser.parse_args()

    cfg = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
    cfg = obj(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = UniversalPatchEvaluator(cfg, args, device)

    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]
    img_numpy_batch = read_img_np_batch([img_names[193]], cfg.DETECTOR.INPUT_SIZE)

    for detector in evaluator.detectors:
        path = os.path.join(args.save, detector.name)
        os.makedirs(path, exist_ok=True)

        # 对batch的通用处理
        all_preds = None
        img_tensor_batch = detector.init_img_batch(img_numpy_batch)

        # 对clean样本进行检测
        preds, _ = detector(img_tensor_batch)
        all_preds = evaluator.merge_batch_pred(all_preds, preds)

        # 保存在clean样本上的检测结果
        save_detections(img_tensor_batch[0], path+'/original.png')

        # 获取所有的检测框位置
        evaluator.get_patch_pos_batch(all_preds)
        # Transform patch
        adv_img_tensor, _ = evaluator.uap_apply(img_numpy_batch)
        # 再次检测
        preds, _ = detector(adv_img_tensor)
        save_detections(adv_img_tensor[0], path+'/attacked.png')
