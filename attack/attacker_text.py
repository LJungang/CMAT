import torch
import numpy as np

from attack.base import BaseAttacker
from tools import DataTransformer, pad_lab
from attack.uap import PatchManager
from tools.det_utils import inter_nms
#from detlib.utils import init_text_detectors


class UniversalAttacker(BaseAttacker):
    """
    An attacker agent to coordinate the detect & base attack methods for universal attacks.
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.vlogger = None
        self.weight = None #weight for detectors
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH_ATTACK, device)
        print(self.detectors)
        self.max_boxes = 15
        

    @property
    def universal_patch(self):
        return self.patch_obj.patch

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def attack(self, img_names, mode='sequential'):
        '''
        given batch input, return loss, and optimize patch
        '''
        detectors_loss = []
        if mode == 'optim' or mode == 'sequential':
            #for attacking single model
            for name, detector in self.detectors.items():
                loss = self.attacker.non_targeted_attack_for_text(img_names, self.universal_patch, detector,name)
                detectors_loss.append(loss)
        elif mode == 'parallel':
            # for attacking multiple models
            detectors_loss = self.parallel_attack(img_names)
        return torch.tensor(detectors_loss).mean()

    def parallel_attack(self, img_names):
        detectors_loss = None
        for name, detector in self.detectors.items():
            loss = self.attacker.parallel_non_targeted_attack_for_text(img_names, self.universal_patch, detector,name)
            loss = loss if self.weight is None else loss * self.weight[name]
            if detectors_loss is None:
                detectors_loss =  loss
            else:
                detectors_loss = loss + detectors_loss
        
        detectors_loss = detectors_loss/len(self.detectors)
        detectors_loss.backward()
        self.attacker.patch_update(patch_clamp_=self.patch_obj.clamp_)
        return detectors_loss
