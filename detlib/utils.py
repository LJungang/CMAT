import numpy as np
import torchvision
import torch
import os

from .mmocr.mmocr.utils.ocr import MMOCR


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_LIB = os.path.join(PROJECT_DIR, 'detlib')

def init_detector_weight(cfg):
    """
    return a dict containing all text_detectors and their weights.
    """
    weight = {}

    for i in range(len(cfg.NAME)):
        weight[cfg.NAME[i]] = cfg.WEIGHT[i]
    
    return weight

def init_text_detectors(detector_names):
    """
    return a dict containing all text_detectors and their names.
    """
    detectors = {}

    for name in detector_names:
        name = name.lower()
        if name == 'ps_ic15':
            ps_ic15 = MMOCR(det='PS_IC15', recog=None)
            detectors["PS_IC15"] = ps_ic15
        elif name == 'ps_ctw':
            ps_ctw = MMOCR(det='PS_CTW', recog=None)
            detectors["PS_CTW"] = ps_ctw
        elif name == 'panet_ic15':
            panet_ic15 = MMOCR(det='PANet_IC15', recog=None)
            detectors["PANET_IC15"] = panet_ic15
        elif name == 'panet_ctw':
            panet_ctw = MMOCR(det='PANet_CTW',recog=None)
            detectors["PANET_CTW"] = panet_ctw
        elif name == 'textsnake':
            textsnake = MMOCR(det='TextSnake',recog=None)
            detectors["TextSnake"] = textsnake
        elif name == "db_r50":
            db_r50 = MMOCR(det= 'DB_r50',recog = None)
            detectors['DB_r50'] = db_r50
        elif name == "dbpp_r50":
            dbpp = MMOCR(det = 'DBPP_r50',recog = None)
            detectors['DBPP_r50'] = dbpp
    return detectors



def inter_nms(all_predictions, conf_thres=0.25, iou_thres=0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    max_det = 300  # maximum number of detections per image
    out = []
    for predictions in all_predictions:
        # for each img in batch
        # print('pred', predictions.shape)
        if not predictions.shape[0]:
            out.append(predictions)
            continue
        if type(predictions) is np.ndarray:
            predictions = torch.from_numpy(predictions)
        # print(predictions.shape[0])
        try:
            scores = predictions[:, 4]
        except Exception as e:
            print(predictions.shape)
            assert 0==1
        i = scores > conf_thres

        # filter with conf threshhold
        boxes = predictions[i, :4]
        scores = scores[i]

        # filter with iou threshhold
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # print('i', predictions[i].shape)
        out.append(predictions[i])
    return out


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names