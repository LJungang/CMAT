import copy
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

def parser_input():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str)
    parser.add_argument('-p', '--patch_dir', type=str)
    parser.add_argument('-r', '--rule', type=str, default=None)
    parser.add_argument('-s', '--save', type=str)
    parser.add_argument('-e', '--eva_class', type=str, default='-1')
    parser.add_argument('-ng', '--gen_no_label', action='store_true')
    args = parser.parse_args()

    args.patch_dir = os.path.join(ROOT, args.patch_dir)
    if args.rule is None:
        args.rule = '.'
    # args.save = os.path.join(args.save, args.cfg)

    args.cfg = './configs/' + args.cfg + '.yaml'
    cfg = ConfigParser(args.cfg)
    args.label_path = os.path.join(ROOT, cfg.DATA.TRAIN.LAB_DIR)
    args.save = os.path.join(ROOT, args.save)
    os.makedirs(args.save, exist_ok=True)
    args.data_root = os.path.join(ROOT, cfg.DATA.TRAIN.IMG_DIR)
    args.test_origin = False
    args.gen_labels = not args.gen_no_label
    args.detectors = None
    args.stimulate_uint8_loss = False
    args.save_imgs = False
    args.quiet = True

    args_train = args

    return args_train, cfg


def batch_mAP(cfg, key_dir):
    y = {}
    for detector_name in cfg.DETECTOR.NAME:
        y[detector_name.lower()] = []
    x = []
    for patch_file in patch_files:
        x.append(int(re.findall(r"\d+", patch_file)[0]))

        args = copy.deepcopy(args_train)
        args.save += key_dir
        args.patch = os.path.join(args.patch_dir, patch_file)
        args_, cfg, _ = init(args, cfg)
        det_mAPs, _, _, _ = eva(args, cfg)

        for k, v in det_mAPs.items():
            y[k].append(float(v))

        args_.save += '/test'
        args_.data_root = os.path.join(ROOT, cfg.DATA.TEST.IMG_DIR)
        args_.label_path = os.path.join(ROOT, cfg.DATA.TEST.LAB_DIR)
        det_mAPs, _, _, _ = eva(args, cfg)
        for k, v in det_mAPs.items():
            y_test[k].append(float(v))


def readAP(p):
    with open(p, 'r') as f:
        mAP = f.readlines()[1].split('%')[0]
        # print(mAP)
    return float(mAP)


if __name__ == '__main__':
    from tools.parser import ConfigParser
    import os
    import numpy as np
    import re

    import matplotlib.pyplot as plt
    from evaluate import eva
    from evaluate import get_save

    args_train, cfg = parser_input()
    print('save dir: ', args_train.save)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patch_files = os.listdir(args_train.patch_dir)
    patch_files = list(filter(lambda file: args_train.rule in file, patch_files))
    print('patch num: ', len(patch_files))

    y_train = {}
    for detector_name in cfg.DETECTOR.NAME:
        y_train[detector_name.lower()] = []
    y_test = copy.deepcopy(y_train)

    x = []
    try:
        for patch_file in patch_files:
            args = copy.deepcopy(args_train)
            fp = args.save+'all_data/'+patch_file
            if os.path.exists(fp):
                print('exists '+fp)
                try:
                    mAP = readAP(os.path.join(*[fp, 'test/yolov3/det-results/results.txt']))
                    print('mAP read!')
                    y_test['yolov3'].append(float(mAP))
                    continue
                except Exception as e:
                    print(e)

            if '_' not in patch_file:
                continue
            x.append(int(patch_file.split('_')[0]))


            args.save += '/all_data'
            args.patch = os.path.join(args.patch_dir, patch_file)
            args = get_save(args)
            det_mAPs, _, _, _ = eva(args, cfg)

            for k, v in det_mAPs.items():
                y_train[k].append(float(v))

            args.save += '/test'
            args.data_root = os.path.join(ROOT, cfg.DATA.TEST.IMG_DIR)
            args.label_path = os.path.join(ROOT, cfg.DATA.TEST.LAB_DIR)
            det_mAPs, _, _, _ = eva(args, cfg)
            for k, v in det_mAPs.items():
                y_test[k].append(float(v))
            # cur += 1
            # if cur == 3:
            #     break
    except Exception as e:
        print(e)

    np.save(args_train.save+'/x.npy', x)
    np.save(args_train.save+'/y_train.npy', y_train)
    np.save(args_train.save + '/y_test.npy', y_test)

    plt.figure()
    for train_y, test_y in zip(y_train.values(), y_test.values()):
        # print(x, train_y)
        # plt.plot(x, train_y)
        # print(x, train_y, test_y)
        plt.scatter(x, train_y, c='b', label='train')
        plt.scatter(x, test_y, c='r', label='test')
    plt.legend()
    plt.ylabel('mAP(%)')
    plt.xlabel('# iteration')
    plt.savefig(args_train.save+'/gap.png', dpi=300)
    print(args_train.save+'/gap.png')