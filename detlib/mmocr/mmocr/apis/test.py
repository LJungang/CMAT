# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import torch
from torchvision import transforms
from mmcv.image import tensor2imgs
from mmcv.parallel import DataContainer
from mmdet.core import encode_mask_results

from .utils import tensor2grayimgs


def retrieve_img_tensor_and_meta(data):
    """Retrieval img_tensor, img_metas and img_norm_cfg.

    Args:
        data (dict): One batch data from data_loader.

    Returns:
        tuple: Returns (img_tensor, img_metas, img_norm_cfg).

            - | img_tensor (Tensor): Input image tensor with shape
                :math:`(N, C, H, W)`.
            - | img_metas (list[dict]): The metadata of images.
            - | img_norm_cfg (dict): Config for image normalization.
    """

    if isinstance(data['img'], torch.Tensor):
        # for textrecog with batch_size > 1
        # and not use 'DefaultFormatBundle' in pipeline
        img_tensor = data['img']
        img_metas = data['img_metas'].data[0]
    elif isinstance(data['img'], list):
        if isinstance(data['img'][0], torch.Tensor):
            # for textrecog with aug_test and batch_size = 1
            img_tensor = data['img'][0]
        elif isinstance(data['img'][0], DataContainer):
            # for textdet with 'MultiScaleFlipAug'
            # and 'DefaultFormatBundle' in pipeline
            img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
    elif isinstance(data['img'], DataContainer):
        # for textrecog with 'DefaultFormatBundle' in pipeline
        img_tensor = data['img'].data[0]
        img_metas = data['img_metas'].data[0]

    must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape', 'ori_shape']
    for key in must_keys:
        if key not in img_metas[0]:
            raise KeyError(
                f'Please add {key} to the "meta_keys" in the pipeline')

    img_norm_cfg = img_metas[0]['img_norm_cfg']
    if max(img_norm_cfg['mean']) <= 1:
        img_norm_cfg['mean'] = [255 * x for x in img_norm_cfg['mean']]
        img_norm_cfg['std'] = [255 * x for x in img_norm_cfg['std']]

    return img_tensor, img_metas, img_norm_cfg


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    is_kie=False,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if is_kie:
                img_tensor = data['img'].data[0]
                if img_tensor.shape[0] != 1:
                    raise KeyError('Visualizing KIE outputs in batches is'
                                   'currently not supported.')
                gt_bboxes = data['gt_bboxes'].data[0]
                img_metas = data['img_metas'].data[0]
                must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape']
                for key in must_keys:
                    if key not in img_metas[0]:
                        raise KeyError(
                            f'Please add {key} to the "meta_keys" in config.')
                # for no visual model
                if np.prod(img_tensor.shape) == 0:
                    imgs = []
                    for img_meta in img_metas:
                        try:
                            img = mmcv.imread(img_meta['filename'])
                        except Exception as e:
                            print(f'Load image with error: {e}, '
                                  'use empty image instead.')
                            img = np.ones(
                                img_meta['img_shape'], dtype=np.uint8)
                        imgs.append(img)
                else:
                    imgs = tensor2imgs(img_tensor,
                                       **img_metas[0]['img_norm_cfg'])
                for i, img in enumerate(imgs):
                    h, w, _ = img_metas[i]['img_shape']
                    img_show = img[:h, :w, :]
                    if out_dir:
                        out_file = osp.join(out_dir,
                                            img_metas[i]['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        gt_bboxes[i],
                        show=show,
                        out_file=out_file)
            else:
                img_tensor, img_metas, img_norm_cfg = \
                    retrieve_img_tensor_and_meta(data)

                if img_tensor.size(1) == 1:
                    imgs = tensor2grayimgs(img_tensor, **img_norm_cfg)
                else:
                    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
                assert len(imgs) == len(img_metas)

                for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    img_shape, ori_shape = img_meta['img_shape'], img_meta[
                        'ori_shape']
                    img_show = img[:img_shape[0], :img_shape[1]]
                    img_show = mmcv.imresize(img_show,
                                             (ori_shape[1], ori_shape[0]))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_attack_test(model,
                    data_loader,
                    perturbation,
                    show=False,
                    out_dir=None,
                    is_kie=False,
                    show_score_thr=0.3):
    model.eval()
    clean_results = []
    adv_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:

        Trans = transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229, 0.224,0.225])

        invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                       transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]),])
        #random crop
        perturbation,_,_ = random_crop(perturbation,[data['img'][0].shape[2],data['img'][0].shape[3]])
        perturbation = perturbation.detach()
        #print(perturbation.shape)
        #perturbation = torch.nn.functional.interpolate(perturbation,size=(data['img'][0].shape[2], data['img'][0].shape[3]), mode='bilinear')

        #record clean result
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        clean_results.extend(result)

        #record adv result
        #因为data里面的数据已经被normalize，所以先反normalize一下
        data['img']= [invTrans(i) for i in data['img']]

        #加上扰动
        data['img']= [i +perturbation for i in data['img']]

        #clamp到[0,1]范围
        data['img']= [torch.clamp(i,0,1) for i in data['img']]
        
        #再重新normalize
        data['img']= [Trans(i) for i in data['img']]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        if show or out_dir:
            if is_kie:
                img_tensor = data['img'].data[0]
                if img_tensor.shape[0] != 1:
                    raise KeyError('Visualizing KIE outputs in batches is'
                                   'currently not supported.')
                gt_bboxes = data['gt_bboxes'].data[0]
                img_metas = data['img_metas'].data[0]
                must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape']
                for key in must_keys:
                    if key not in img_metas[0]:
                        raise KeyError(
                            f'Please add {key} to the "meta_keys" in config.')
                # for no visual model
                if np.prod(img_tensor.shape) == 0:
                    imgs = []
                    for img_meta in img_metas:
                        try:
                            img = mmcv.imread(img_meta['filename'])
                        except Exception as e:
                            print(f'Load image with error: {e}, '
                                  'use empty image instead.')
                            img = np.ones(
                                img_meta['img_shape'], dtype=np.uint8)
                        imgs.append(img)
                else:
                    imgs = tensor2imgs(img_tensor,
                                       **img_metas[0]['img_norm_cfg'])
                for i, img in enumerate(imgs):
                    h, w, _ = img_metas[i]['img_shape']
                    img_show = img[:h, :w, :]
                    if out_dir:
                        out_file = osp.join(out_dir,
                                            img_metas[i]['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        gt_bboxes[i],
                        show=show,
                        out_file=out_file)
            else:
                img_tensor, img_metas, img_norm_cfg = \
                    retrieve_img_tensor_and_meta(data)

                if img_tensor.size(1) == 1:
                    imgs = tensor2grayimgs(img_tensor, **img_norm_cfg)
                else:
                    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
                assert len(imgs) == len(img_metas)

                for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    img_shape, ori_shape = img_meta['img_shape'], img_meta[
                        'ori_shape']
                    img_show = img[:img_shape[0], :img_shape[1]]
                    img_show = mmcv.imresize(img_show,
                                             (ori_shape[1], ori_shape[0]))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        adv_results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return clean_results,adv_results


def random_crop(cloth, crop_size, pos=None, crop_type='recursive', fill=0):
    w = cloth.shape[2]
    h = cloth.shape[3]
    if crop_size is 'equal':
        crop_size = [w, h]
    if crop_type is None:
        d_w = w - crop_size[0]
        d_h = h - crop_size[1]
        if pos is None:
            r_w = np.random.randint(d_w + 1)
            r_h = np.random.randint(d_h + 1)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
        else:
            r_w = pos[0]
            r_h = pos[1]

        p1 = max(0, 0 - r_h)
        p2 = max(0, r_h + crop_size[1] - h)
        p3 = max(0, 0 - r_w)
        p4 = max(0, r_w + crop_size[1] - w)
        cloth_pad = F.pad(cloth, [p1, p2, p3, p4], value=fill)
        patch = cloth_pad[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    elif crop_type == 'recursive':
        if pos is None:
            r_w = np.random.randint(w)
            r_h = np.random.randint(h)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
            if r_w < 0:
                r_w = r_w % w
            if r_h < 0:
                r_h = r_h % h
        else:
            r_w = pos[0]
            r_h = pos[1]
        expand_w = (w + crop_size[0] - 1) // w + 1
        expand_h = (h + crop_size[1] - 1) // h + 1
        cloth_expanded = cloth.repeat([1, 1, expand_w, expand_h])
        patch = cloth_expanded[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    else:
        raise ValueError
    return patch, r_w, r_h