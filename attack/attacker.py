import torch
import numpy as np

from attack.base import BaseAttacker
from tools import DataTransformer, pad_lab
from attack.uap import PatchManager, PatchRandomApplier
from tools.det_utils import inter_nms


class UniversalAttacker(BaseAttacker):
    """
    An attacker agent to coordinate the detect & base attack methods for universal attacks.
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.vlogger = None
        self.gates = {'jitter': False, 'median_pool': False, 'rotate': True, 'shift': False, 'p9_scale': False}
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH_ATTACK, device)
        self.patch_apply = PatchRandomApplier(device, scale_rate=cfg.ATTACKER.PATCH_ATTACK.SCALE)
        self.max_boxes = 15
        if '3' in cfg.DATA.AUGMENT:
            self.data_transformer = DataTransformer(device, rand_rotate=0)

    @property
    def universal_patch(self):
        return self.patch_obj.patch

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds, target_cls=None):
        # FIXME: To be a more universal op fn
        if len(preds) == 0:
            return preds
        # if cls_array is None: cls_array = preds[:, -1]
        # filt = [cls in self.cfg.attack_list for cls in cls_array]
        # preds = preds[filt]
        target_cls = self.cfg.attack_cls if target_cls is None else target_cls
        return preds[preds[:, -1] == target_cls]

    def get_patch_pos_batch(self, all_preds):
        # get all bboxs of setted target. If none target bbox is got, return has_target=False
        self.all_preds = all_preds
        batch_boxes = None
        target_nums = []
        for i_batch, preds in enumerate(all_preds):
            if len(preds) == 0:
                preds = torch.cuda.FloatTensor([[0, 0, 0, 0, 0, 0]])
            preds = self.filter_bbox(preds)
            padded_boxs = pad_lab(preds, self.max_boxes).unsqueeze(0)
            batch_boxes = padded_boxs if batch_boxes is None else torch.vstack((batch_boxes, padded_boxs))
            target_nums.append(len(preds))
        self.all_preds = batch_boxes
        return np.array(target_nums)

    def uap_apply(self, img_tensor, adv_patch=None, gates=None):
        '''
        UAP: universal adversarial patch
        :param img_tensor:
        :param adv_patch:
        :param gates: The patch augmentation gates(dict: True or False).
        :return:
        '''
        if adv_patch is None: adv_patch = self.universal_patch
        if gates is None: gates = self.gates

        img_tensor = self.patch_apply(img_tensor, adv_patch, self.all_preds, gates=gates)

        if '2' in self.cfg.DATA.AUGMENT:
            img_tensor = self.data_transformer(img_tensor)

        return img_tensor

    def merge_batch(self, all_preds, preds):
        if all_preds is None:
            return preds
        for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
            if pred.shape[0]:
                all_preds[i] = torch.cat((all_pred, pred), dim=0)
                continue
            all_preds[i] = all_pred if all_pred.shape[0] else pred
        return all_preds

    def detect_bbox(self, img_batch, detectors=None):
        if detectors is None:
            detectors = self.detectors

        all_preds = None
        for detector in detectors:
            preds = detector(img_batch)['bbox_array']
            all_preds = self.merge_batch(all_preds, preds)

        # nms among detectors
        if len(detectors) > 1: all_preds = inter_nms(all_preds)
        return all_preds

    def attack(self, img_tensor_batch, mode='sequential'):
        '''
        given batch input, return loss, and optimize patch
        '''
        detectors_loss = []
        if mode == 'optim' or mode == 'sequential':
            for detector in self.detectors:
                loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
                detectors_loss.append(loss)
        elif mode == 'parallel':
            detectors_loss = self.parallel_attack(img_tensor_batch)
        return torch.tensor(detectors_loss).mean()

    def parallel_attack(self, img_tensor_batch):
        detectors_loss = []
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            patch_tmp, loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
            patch_update = patch_tmp - self.universal_patch
            patch_updates += patch_update
            detectors_loss.append(loss)
        self.patch_obj.update_((self.universal_patch + patch_updates / len(self.detectors)).detach_())
        return detectors_loss
