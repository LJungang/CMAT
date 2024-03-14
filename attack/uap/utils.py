import torch
import torch.nn.functional as F

"""
TODO: This file is not used now.
"""
def attach_patch(img_tensor, adv_patch, all_preds, cfg):
    height, width = cfg.DETECTOR.INPUT_SIZE
    scale = cfg.ATTACKER.PATCH_ATTACK.SCALE
    aspect_ratio = cfg.ATTACKER.PATCH_ATTACK.ASPECT_RATIO
    for i in range(img_tensor.shape[0]):
        boxes = rescale_patches(all_preds[i], height, width, scale, aspect_ratio)
        for j, bbox in enumerate(boxes):
            # for jth bbox in ith-img's bboxes
            p_x1, p_y1, p_x2, p_y2 = bbox[:4]
            height = p_y2 - p_y1
            width = p_x2 - p_x1
            if height <= 0 or width <= 0:
                continue
            adv = F.interpolate(adv_patch, size=(height, width), mode='bilinear')[0]
            # Warning: This is an inplace operation, it changes the value of the outer variable
            img_tensor[i][:, p_y1:p_y2, p_x1:p_x2] = adv
    return img_tensor


def rescale_patches(bboxes, image_height, image_width, scale, aspect_ratio):
    def compute(bwidth, bheight, scale, aspect_ratio):
        # fix aspect area ratio
        if aspect_ratio > 0:
            target_y = torch.sqrt(bwidth * bheight * scale / aspect_ratio)
            target_x = aspect_ratio * target_y
        # oiginal' scale
        elif aspect_ratio == -1:
            target_x = (torch.sqrt(scale) * bwidth)
            target_y = (torch.sqrt(scale) * bheight)
        # natural's scale
        elif aspect_ratio == -2:
            target_x = torch.sqrt((scale * bheight) ** 2 + (scale * bwidth) ** 2)
            # adjust patch height as rectangle
            target_y = target_x * 1.5
        else:
            assert False, 'aspect ratio undefined!'
        return target_x / 2, target_y / 2

    image_height = torch.cuda.FloatTensor([image_height])
    image_width = torch.cuda.FloatTensor([image_width])
    scale = torch.cuda.FloatTensor([scale])
    # print(image_width, image_height)
    bboxes = bboxes[:, :4].cuda()
    bwidth = bboxes[:, 2] - bboxes[:, 0]
    bheight = bboxes[:, 3] - bboxes[:, 1]
    xc = (bboxes[:, 2] + bboxes[:, 0]) / 2
    yc = (bboxes[:, 3] + bboxes[:, 1]) / 2

    target_x, target_y = compute(bwidth, bheight, scale, aspect_ratio)

    target = torch.stack((-target_x, -target_y, target_x, target_y)).T
    # print('target: ', target.size())
    len = torch.stack((image_width, image_height, image_width, image_height)).T
    # print('len: ', len.size())
    bcenter = torch.stack((xc, yc, xc, yc)).T
    # print('bc', bcenter.size())

    bboxes = (bcenter + target).clamp(0, 1) * len
    # print('rescaled bbox: ', bboxes)
    return bboxes.to(torch.int32)