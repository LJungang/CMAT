from bisect import bisect_right
import torch
from torch import optim


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class _WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        """

        :param optimizer:
        :param milestones:
        :param gamma:
        :param warmup_factor:
        :param warmup_iters:
        :param warmup_method:
        :param last_epoch:
        """
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(_WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class _ExponentStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=0.999, step_size=1, last_epoch=1):
        self.gamma = gamma
        self.step_size = step_size
        super(_ExponentStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 200 or self.last_epoch % self.step_size:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


def Expo_lr_scheduler(optimizer):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, update_step=5)


def Cosine_lr_scheduler(optimizer, total_epoch=1000):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)


def Warmup_lr_scheduler(milestone, optimizer):
    return _WarmupMultiStepLR(optimizer, milestone)


def Plateau_lr_scheduler(optimizer, patience=100):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)


class ALRS():
    '''
    proposer: Huanran Chen
    theory: landscape
    Bootstrap Generalization Ability from Loss Landscape Perspective
    '''

    def __init__(self, optimizer, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.97):
        self.optimizer = optimizer

        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta/self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.decay_rate
                now_lr = group['lr']
                print(f'now lr = {now_lr}')

        self.last_loss = loss
