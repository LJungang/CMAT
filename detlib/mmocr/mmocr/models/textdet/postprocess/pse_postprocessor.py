# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import torch
from mmcv.ops import contour_expand

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor


@POSTPROCESSOR.register_module()
class PSEPostprocessor(BasePostprocessor):
    """Decoding predictions of PSENet to instances. This is partially adapted
    from https://github.com/whai362/PSENet.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_kernel_area (int): The minimal text kernel area.
        min_text_area (int): The minimal text instance region area.
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_kernel_confidence=0.5,
                 min_text_avg_confidence=0.85,
                 min_kernel_area=0,
                 min_text_area=16,
                 **kwargs):
        super().__init__(text_repr_type)

        assert 0 <= min_kernel_confidence <= 1
        assert 0 <= min_text_avg_confidence <= 1
        assert isinstance(min_kernel_area, int)
        assert isinstance(min_text_area, int)

        self.min_kernel_confidence = min_kernel_confidence
        self.min_text_avg_confidence = min_text_avg_confidence
        self.min_kernel_area = min_kernel_area
        self.min_text_area = min_text_area

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        assert preds.dim() == 3

        preds = torch.sigmoid(preds)  # text confidence

        score = preds[0, :, :]
        masks = preds > self.min_kernel_confidence
        text_mask = masks[0, :, :]
        kernel_masks = masks[0:, :, :] * text_mask

        score = score.data.cpu().numpy().astype(np.float32)

        kernel_masks = kernel_masks.data.cpu().numpy().astype(np.uint8)

        region_num, labels = cv2.connectedComponents(
            kernel_masks[-1], connectivity=4)

        labels = contour_expand(kernel_masks, labels, self.min_kernel_area,
                                region_num)
        labels = np.array(labels)
        label_num = np.max(labels)
        boundaries = []
        for i in range(1, label_num + 1):
            points = np.array(np.where(labels == i)).transpose((1, 0))[:, ::-1]
            area = points.shape[0]
            score_instance = np.mean(score[labels == i])
            if not self.is_valid_instance(area, score_instance,
                                          self.min_text_area,
                                          self.min_text_avg_confidence):
                continue

            vertices_confidence = points2boundary(points, self.text_repr_type,
                                                  score_instance)
            if vertices_confidence is not None:
                boundaries.append(vertices_confidence)
        
        # return boundaries
        if len(boundaries)==0:
            return boundaries
        results=merge_boxes(boundaries)
        return results
def merge_boxes(boundaries, min_distance=1):
    """
    Merge text boxes that are on the same line.
    
    Args:
        boundaries (list[list[float]]): A list of boundaries, each containing the vertices of a text box.
        min_distance (float): The minimum distance (in pixels) to consider two text boxes to be on the same line.
        
    Returns:
        list[list[float]]: The updated list of boundaries.
    """
    # 按y的中心坐标排序
    boundaries = sorted(boundaries, key=lambda x: np.mean([x[2*i+1] for i in range(int(len(x)/2))]))# 多边形求各顶点y坐标的均值
    
    merged_boundaries = []
    curr_line_boundaries = []
    curr_line_y = []
    for k in boundaries:
        curr_line_y.append(np.mean([k[2*i+1] for i in range(int(len(k)/2))]))
    
    for i, boundary in enumerate(boundaries):
        if i == 0:
            curr_line_boundaries.append(boundary)
            continue
        
        # 计算前后两个边框的垂直距离
        distance = curr_line_y[i]-curr_line_y[i-1]
        # distance = boundaries[i][1]-boundaries[i-1][1]
        
        # 小于阈值则说明在同一行
        if distance < min_distance:
            curr_line_boundaries.append(boundary)
        else:
            # 将前面认为在同一行的且小于阈值的两个边框合并
            merged_boundaries.extend(merge_line_boxes(curr_line_boundaries))
            curr_line_boundaries = [boundary]
    
    # 最后一个合并上
    merged_boundaries.extend(merge_line_boxes(curr_line_boundaries))
    
    return merged_boundaries

def merge_line_boxes(line_boundaries,threshold = 5):
    """
    Merge text boxes in the same line.

    Args:
        line_boundaries (list[list[float]]): The text boundaries in the same
            line.

    Returns:
        list[list[float]]: The merged text boundaries.
    """
    line_boundaries = sorted(line_boundaries, key=lambda x: np.mean([x[2*i] for i in range(int(len(x)/2))]))# 多边形求各顶点y坐标的均值
    boxes = np.array(line_boundaries)

    boxes = boxes[boxes[:, 0].argsort()]

    merged_boxes = []
    while boxes.shape[0] > 0:
        box = boxes[0]
        merged_box = box.copy()
        merged_box = adjust_pos(merged_box)
        num_boxes = 1
        for i in range(1, boxes.shape[0]):
            boxes[i] = adjust_pos(boxes[i])
            if boxes[i, 0] - merged_box[2] < threshold: 
                merged_box[2] = boxes[i, 2]
                merged_box[4] = boxes[i, 2]
                merged_box[8] = (boxes[i,8]+merged_box[8])/2
                num_boxes += 1
            else:
                break
        merged_box[2] += 1  
        merged_boxes.append(merged_box.tolist())
        boxes = boxes[num_boxes:]

    return merged_boxes

def adjust_pos(ori_box):
    x_min = np.min(np.array([ori_box[0],ori_box[2],ori_box[4],ori_box[6]]))
    x_max = np.max(np.array([ori_box[0],ori_box[2],ori_box[4],ori_box[6]]))
    y_min = np.min(np.array([ori_box[1],ori_box[3],ori_box[5],ori_box[7]]))
    y_max = np.max(np.array([ori_box[1],ori_box[3],ori_box[5],ori_box[7]]))
    return np.array([x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max,ori_box[8]])