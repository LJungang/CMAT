#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test

from mmocr.apis.test import single_gpu_attack_test
from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector
from mmocr.utils import revert_sync_batchnorm, setup_multi_processes


def evaluate(patch,eval_cfg):

    cfg = Config.fromfile(eval_cfg["script_path"])
    
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.model.get('pretrained'):
        cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = (cfg.data.get('test_dataloader', {})).get(
        'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
    if samples_per_gpu > 1:
        cfg = disable_text_recog_aug_test(cfg)
        cfg = replace_image_to_tensor(cfg)

    # init distributed env first, since logger depends on the dist info.
    cfg.gpu_ids = [0]
    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    default_loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           ))
    }
    default_loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **default_loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {}),
        **dict(samples_per_gpu=samples_per_gpu)
    }

    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = revert_sync_batchnorm(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, eval_cfg["weight_path"], map_location='cpu')

    
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    is_kie = cfg.model.type in ['SDMGR']
    clean_outputs,adv_outputs = single_gpu_attack_test(model, data_loader, patch)
    
    rank, _ = get_dist_info()
    kwargs = {}
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='hmean-iou', **kwargs))
    #eval_kwargs['adv_results'] = adv_outputs
    results = dataset.evaluate(adv_outputs, **eval_kwargs)
    return results['0_hmean-iou:recall']

    
