import numpy as np
import torch
from .base import BaseAttacker


class OptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        patch_clamp_ = kwargs['patch_clamp_']
        self.optimizer.step()
        patch_clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    def attack_loss(self, confs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        if 'obj-tv' in self.cfg.LOSS_FUNC:
            tv_loss, obj_loss = loss.values()
            tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
            # print(obj_loss)
            obj_loss = obj_loss * self.cfg.obj_eta
            loss = tv_loss + obj_loss
        elif self.cfg.LOSS_FUNC == 'obj':
            loss = loss['obj_loss'] * self.cfg.obj_eta
            tv_loss = torch.cuda.FloatTensor([0])
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out