# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import cv2
import torch
from torchvision import transforms
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmocr.models import build_detector
from mmocr.utils import is_2dlist
import mmocr.utils as utils
from .utils import disable_text_recog_aug_test


from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont



def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if config.model.get('pretrained'):
        config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def model_inference(model,
                    imgs,
                    ann=None,
                    batch_mode=False,
                    return_data=False):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        batch_mode (bool): If True, use batch mode for inference.
        ann (dict): Annotation info for key information extraction.
        return_data: Return postprocessed data.
    Returns:
        result (dict): Predicted results.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
        if len(imgs) == 0:
            raise Exception('empty imgs provided, please check and try again')
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
        is_batch = False
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    is_ndarray = isinstance(imgs[0], np.ndarray)

    cfg = model.cfg
    if batch_mode:
        cfg = disable_text_recog_aug_test(cfg, set_types=['test'])

    device = next(model.parameters()).device  # model device

    if cfg.data.test.get('pipeline', None) is None:
        if is_2dlist(cfg.data.test.datasets):
            cfg.data.test.pipeline = cfg.data.test.datasets[0][0].pipeline
        else:
            cfg.data.test.pipeline = cfg.data.test.datasets[0].pipeline
    if is_2dlist(cfg.data.test.pipeline):
        cfg.data.test.pipeline = cfg.data.test.pipeline[0]

    if is_ndarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if is_ndarray:
            # directly add img
            data = dict(
                img=img,
                ann_info=ann,
                img_info=dict(width=img.shape[1], height=img.shape[0]),
                bbox_fields=[])
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename=img),
                img_prefix=None,
                ann_info=ann,
                bbox_fields=[])
        if ann is not None:
            data.update(dict(**ann))

        # build the data pipeline
        data = test_pipeline(data)
        # get tensor from list to stack for batch mode (text detection)
        if batch_mode:
            if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
                for key, value in data.items():
                    data[key] = value[0]
        datas.append(data)

    if isinstance(datas[0]['img'], list) and len(datas) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(datas)}')

    data = collate(datas, samples_per_gpu=len(imgs))

    # process img_metas
    if isinstance(data['img_metas'], list):
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
    else:
        data['img_metas'] = data['img_metas'].data

    if isinstance(data['img'], list):
        data['img'] = [img.data for img in data['img']]
        if isinstance(data['img'][0], list):
            data['img'] = [img[0] for img in data['img']]
    else:
        data['img'] = data['img'].data

    # for KIE models
    if ann is not None:
        data['relations'] = data['relations'].data[0]
        data['gt_bboxes'] = data['gt_bboxes'].data[0]
        data['texts'] = data['texts'].data[0]
        data['img'] = data['img'][0]
        data['img_metas'] = data['img_metas'][0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        if not return_data:
            return results[0]
        return results[0], datas[0]
    else:
        if not return_data:
            return results
        return results, datas


def text_model_inference(model, input_sentence):
    """Inference text(s) with the entity recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        input_sentence (str): A text entered by the user.

    Returns:
        result (dict): Predicted results.
    """

    assert isinstance(input_sentence, str)

    cfg = model.cfg
    if cfg.data.test.get('pipeline', None) is None:
        if is_2dlist(cfg.data.test.datasets):
            cfg.data.test.pipeline = cfg.data.test.datasets[0][0].pipeline
        else:
            cfg.data.test.pipeline = cfg.data.test.datasets[0].pipeline
    if is_2dlist(cfg.data.test.pipeline):
        cfg.data.test.pipeline = cfg.data.test.pipeline[0]
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {'text': input_sentence, 'label': {}}

    # build the data pipeline
    data = test_pipeline(data)
    if isinstance(data['img_metas'], dict):
        img_metas = data['img_metas']
    else:
        img_metas = data['img_metas'].data

    assert isinstance(img_metas, dict)
    img_metas = {
        'input_ids': img_metas['input_ids'].unsqueeze(0),
        'attention_masks': img_metas['attention_masks'].unsqueeze(0),
        'token_type_ids': img_metas['token_type_ids'].unsqueeze(0),
        'labels': img_metas['labels'].unsqueeze(0)
    }
    # forward the model
    with torch.no_grad():
        result = model(None, img_metas, return_loss=False)
    return result



def model_attack(model,
                    imgs,
                    perturbation,
                    ann=None,
                    batch_mode=False,
                    return_data=False):
    """Attack image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        batch_mode (bool): If True, use batch mode for inference.
        ann (dict): Annotation info for key information extraction.
        return_data: Return postprocessed data.
    Returns:
        result (dict): Predicted results.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
        if len(imgs) == 0:
            raise Exception('empty imgs provided, please check and try again')
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
        is_batch = False
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    #print('perturbation.shape', perturbation.shape)
    
    is_ndarray = isinstance(imgs[0], np.ndarray)
    #print('perturbation.shape', perturbation.shape)
    #print('imgs[0]', imgs[0].shape)

    cfg = model.cfg
    if batch_mode:
        cfg = disable_text_recog_aug_test(cfg, set_types=['test'])

    device = next(model.parameters()).device  # model device

    if cfg.data.test.get('pipeline', None) is None:
        if is_2dlist(cfg.data.test.datasets):
            cfg.data.test.pipeline = cfg.data.test.datasets[0][0].pipeline
        else:
            cfg.data.test.pipeline = cfg.data.test.datasets[0].pipeline
    if is_2dlist(cfg.data.test.pipeline):
        cfg.data.test.pipeline = cfg.data.test.pipeline[0]

    if is_ndarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if is_ndarray:
            # directly add img
            data = dict(
                img=img,
                ann_info=ann,
                img_info=dict(width=img.shape[1], height=img.shape[0]),
                bbox_fields=[])
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename=img),
                img_prefix=None,
                ann_info=ann,
                bbox_fields=[])
        if ann is not None:
            data.update(dict(**ann))

        # build the data pipeline
        data = test_pipeline(data)
        # get tensor from list to stack for batch mode (text detection)
        if batch_mode:
            if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
                for key, value in data.items():
                    data[key] = value[0]
        datas.append(data)

    if isinstance(datas[0]['img'], list) and len(datas) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(datas)}')

    data = collate(datas, samples_per_gpu=len(imgs))

    # process img_metas
    if isinstance(data['img_metas'], list):
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
    else:
        data['img_metas'] = data['img_metas'].data

    if isinstance(data['img'], list):
        data['img'] = [img.data for img in data['img']]
        if isinstance(data['img'][0], list):
            data['img'] = [img[0] for img in data['img']]
    else:
        data['img'] = data['img'].data

    # for KIE models
    if ann is not None:
        data['relations'] = data['relations'].data[0]
        data['gt_bboxes'] = data['gt_bboxes'].data[0]
        data['texts'] = data['texts'].data[0]
        data['img'] = data['img'][0]
        data['img_metas'] = data['img_metas'][0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
    Trans = transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229, 0.224,0.225])

    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]),])

    #reserve clean results
    clean_boundaries, clean_outs = model.attack(return_loss=False, rescale=True, **data)
    # clear grad
    model.zero_grad()

    #random crop
    perturbation,_,_ = random_crop(perturbation,[data['img'][0].shape[2],data['img'][0].shape[3]])

    #因为data里面的数据已经被normalize，所以先反normalize一下
    data['img'][0] = invTrans(data['img'][0])
    
    #加上扰动
    data['img'][0] = data['img'][0] + perturbation

    #clamp到[0,1]范围
    torch.clamp_(data['img'][0],0,1)
    
    #再重新normalize
    data['img'][0] = Trans(data['img'][0])

    #perturbation = torch.nn.functional.interpolate(perturbation,size=(data['img'][0].shape[2], data['img'][0].shape[3]), mode='bilinear')
    
    
    # forward the model
    boundaries, outs = model.attack(return_loss=False, rescale=True, **data)
    
    # clear grad
    model.zero_grad()
    

    #inverse the normalization for better visualization
    adv_tensor = data['img'][0].clone().detach().squeeze(0)
    adv_tensor = invTrans(adv_tensor)

    #print(imgs[0].type())
    #print(imgs[0].shape)
    clean_img = imgs[0].copy()
    adv_with_bbox,clean_with_bbox = imshow_pred_boundary(clean_img,clean_boundaries,boundaries)
    #print("no effect:",(adv_img == adv_img_with_bbox).all())
    return boundaries, outs, adv_tensor, adv_with_bbox,clean_with_bbox


def imshow_pred_boundary(img,
                        clean_boundaries,
                        boundaries_with_scores,
                        score_thr=0.5,
                        boundary_color='green',
                        text_color='green',
                        thickness=1,
                        font_scale=1,
                        win_name='',
                        show=False,
                        wait_time=0,
                        out_file=None,
                        show_score=False):
    clean_img = img
    adv_img = img.copy()
    if len(boundaries_with_scores) == 0:
        warnings.warn('0 text found in ' + out_file)
        return None

    utils.valid_boundary(boundaries_with_scores[0])
    clean_with_scores = clean_boundaries[0]['boundary_result']
    boundaries_with_scores = boundaries_with_scores[0]['boundary_result']

    scores = np.array([b[-1] for b in boundaries_with_scores])
    inds = scores > score_thr
    labels = [0] * len(boundaries_with_scores)
    boundaries = [boundaries_with_scores[i][:-1] for i in np.where(inds)[0]]
    scores = [scores[i] for i in np.where(inds)[0]]
    labels = [labels[i] for i in np.where(inds)[0]]

    clean_scores = np.array([b[-1] for b in clean_with_scores])
    clean_inds = clean_scores > score_thr
    clean_labels = [0] * len(clean_with_scores)
    clean = [clean_with_scores[i][:-1] for i in np.where(clean_inds)[0]]
    clean_scores = [clean_scores[i] for i in np.where(clean_inds)[0]]
    clean_labels = [clean_labels[i] for i in np.where(clean_inds)[0]]

    boundary_color = mmcv.color_val(boundary_color)
    clean_color = mmcv.color_val('red')
    text_color = mmcv.color_val(text_color)
    font_scale = 0.5

    for boundary, score in zip(boundaries, scores):
        boundary_int = np.array(boundary).reshape(-1, 2).astype(np.int32)
        #print(boundary_int," ",score)

        cv2.polylines(
            adv_img, [boundary_int.reshape(-1, 1, 2)],
            True,
            color=boundary_color,
            thickness=thickness)

        if show_score:
            label_text = f'{score:.02f}'
            cv2.putText(adv_img, label_text,
                        (boundary_int[0], boundary_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    
    for boundary in clean:
        boundary_int = np.array(boundary).reshape(-1, 2).astype(np.int32)
        #print(boundary_int," ",score)

        cv2.polylines(
            clean_img, [boundary_int.reshape(-1, 1, 2)],
            True,
            color=clean_color,
            thickness=thickness)

    return adv_img,clean_img


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