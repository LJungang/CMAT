# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import box_score_fast, unclip


@POSTPROCESSOR.register_module()
class DBPostprocessor(BasePostprocessor):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        epsilon_ratio (float): The epsilon ratio for approximation accuracy.
        max_candidates (int): The maximum candidate number.
    """

    def __init__(self,
                 text_repr_type='poly',
                 mask_thr=0.3,
                 min_text_score=0.3,
                 min_text_width=5,
                 unclip_ratio=1.5,
                 epsilon_ratio=0.01,
                 max_candidates=3000,
                 **kwargs):
        super().__init__(text_repr_type)
        self.mask_thr = mask_thr
        self.min_text_score = min_text_score
        self.min_text_width = min_text_width
        self.unclip_ratio = unclip_ratio
        self.epsilon_ratio = epsilon_ratio
        self.max_candidates = max_candidates

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The predicted text boundaries.
        """
        assert preds.dim() == 3

        prob_map = preds[0, :, :]
        text_mask = prob_map > self.mask_thr

        score_map = prob_map.data.cpu().numpy().astype(np.float32)
        text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

        contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boundaries = []
        for i, poly in enumerate(contours):
            if i > self.max_candidates:
                break
            epsilon = self.epsilon_ratio * cv2.arcLength(poly, True)
            approx = cv2.approxPolyDP(poly, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = box_score_fast(score_map, points)
            if score < self.min_text_score:
                continue
            poly = unclip(points, unclip_ratio=self.unclip_ratio)
            if len(poly) == 0 or isinstance(poly[0], list):
                continue
            poly = poly.reshape(-1, 2)

            if self.text_repr_type == 'quad':
                poly = points2boundary(poly, self.text_repr_type, score,
                                       self.min_text_width)
            elif self.text_repr_type == 'poly':
                poly = poly.flatten().tolist()
                print(poly)
                if score is not None:
                    poly = poly + [score]
                if len(poly) < 8:
                    poly = None
            
            if poly is not None:
                boundaries.append(poly)
        # return boundaries
        if len(boundaries)==0:
            return boundaries
        results = merge_boxes(boundaries)
        return results
        
def merge_boxes(boundaries, min_distance=8):
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

def merge_line_boxes(line_boundaries,threshold = 14):
    """
    Merge text boxes in the same line.

    Args:
        line_boundaries (list[list[float]]): The text boundaries in the same
            line.

    Returns:
        list[list[float]]: The merged text boundaries.
    """
    
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
                merged_box[8] = max(boxes[i,8],merged_box[8])
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
    
    