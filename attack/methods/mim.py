from .base import BaseAttacker

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import copy
from tqdm import tqdm
import cv2


class LinfMIMAttack(BaseAttacker):
    """MI-FGSM attack (arxiv: https://arxiv.org/pdf/1710.06081.pdf)
    """

    def __init__(self, loss_func, cfg, device, detector_attacker, norm='L_infty', momentum=1, perturbation=None):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)
        self.momentum = momentum
        self.grad = None
     
    def non_targeted_attack(self, x, y, is_universal = False, x_min=-1, x_max=1, *model_args):
        """the main attack method of MIM

        Args:
            x ([torch.tensor]): [input of model]
            y ([torch.tensor]): [expected or unexpected outputs]
            x_min (int, optional): [the minimum value of input x]. Defaults to -1.
            x_max (int, optional): [the maximum value of input x]. Defaults to 1.

        Returns:
            [tensor]: [the adversarial examples crafted by MIM]
        """
        x_ori = x.clone().detach_()
        x_adv = x.clone().detach_()
        for iter in range(self.max_iters):
            x_adv.requires_grad = True
            output = self.model(x_adv, model_args)
            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            if self.grad is None:
                self.grad = x_adv.grad
            else:
                self.grad += self.grad * self.momentum +  x_adv.grad / torch.norm(x_adv.grad, p=1)
            X_adv = x_adv + self.step_size * self.grad.sign()
            eta = torch.clamp(X_adv - x_ori, min=-self.epsilon, max=self.epsilon)
            if is_universal:
                eta = torch.mean(eta.detach_(),dim=0)
            x_adv = torch.clamp(x_ori + eta, min=x_min, max=x_max).detach_()
        self.model.zero_grad()
        self.perturbation = eta
        return x_adv

    def non_targeted_attack_mmdet(self, data, img_cv2, img, img_resized, test_pipeline, prase_func, is_prase=True, is_universal = False, x_min=-255, x_max=255):
        """the main attack method of BIM

        Args:
            data ([torch.tensor]): [input of model]
            img_cv2 ([np.array]): [image readed by opencv-python]
            img ([torch.tensor]): [tensor transformed from img_cv2]
            img_resized ([torch.tensor]): [resized tensor transformed from img_cv2]
            x_min (int, optional): [the minimum value of input x]. Defaults to -1.
            x_max (int, optional): [the maximum value of input x]. Defaults to 1.

        Returns:
            [tensor]: [the adversarial examples crafted by BIM]
        """
        x_ori = copy.deepcopy(img)
        x_adv = img
        x_adv.requires_grad = True
        original_boxes_number = -1
        pbar = tqdm(range(self.max_iters), desc='attacking image ......')
        for iter in pbar:
            results, cls_scores, det_bboxes = self.model(return_loss=False, rescale=True, **data)
            if is_prase:
                bboxes, labels, bbox_num = prase_func(self.model, img_cv2, results[0])
                if original_boxes_number==-1:
                    original_boxes_number = bbox_num
                pbar.set_description('attacking image... original boxes: {} / now boxes: {}'.format(original_boxes_number, bbox_num))
            self.model.zero_grad()
            loss = self.loss_fn(det_bboxes[0][:,-1])
            loss.backward()
            if self.grad is None:
                self.grad = x_adv.grad
            else:
                self.grad += self.grad * self.momentum +  x_adv.grad / torch.norm(x_adv.grad, p=1)
            X_adv = x_adv + self.step_size * self.grad.sign()
            eta = torch.clamp(X_adv - x_ori, min=-self.epsilon, max=self.epsilon)
            if is_universal:
                eta = torch.mean(eta.detach_(),dim=0)
            x_adv = torch.clamp(x_ori + eta, min=x_min, max=x_max).detach_()
            x_adv.requires_grad = True
            data['img'] = x_adv
            data = test_pipeline(data)
        self.model.zero_grad()
        self.perturbation = eta
        x_adv = x_adv.cpu().detach().clone().squeeze(0).numpy().transpose(1, 2, 0)
        x_adv_cv2 = cv2.cvtColor(x_adv, cv2.COLOR_RGB2BGR)
        return x_adv, x_adv_cv2

    def non_targeted_patch_attack_mmdet(self, data, img_cv2, img, img_resized, test_pipeline, prase_func, mask, is_prase=True, is_universal = False, x_min=-255, x_max=255):
        """the main attack method of BIM

        Args:
            data ([torch.tensor]): [input of model]
            img_cv2 ([np.array]): [image readed by opencv-python]
            img ([torch.tensor]): [tensor transformed from img_cv2]
            img_resized ([torch.tensor]): [resized tensor transformed from img_cv2]
            x_min (int, optional): [the minimum value of input x]. Defaults to -1.
            x_max (int, optional): [the maximum value of input x]. Defaults to 1.

        Returns:
            [tensor]: [the adversarial examples crafted by BIM]
        """
        x_ori = copy.deepcopy(img)
        x_adv = img
        x_adv.requires_grad = True
        original_boxes_number = -1
        pbar = tqdm(range(self.max_iters), desc='attacking image ......')
        mask = torch.from_numpy(cv2.resize(mask[:,:,0], (800, 800))).cuda()
        mask[mask>0] = 1

        for iter in pbar:
            results, cls_scores, det_bboxes = self.model(return_loss=False, rescale=True, **data)
            if is_prase:
                bboxes, labels, bbox_num = prase_func(self.model, img_cv2, results[0])
                if original_boxes_number==-1:
                    original_boxes_number = bbox_num
                pbar.set_description('attacking image... original boxes: {} / now boxes: {}'.format(original_boxes_number, bbox_num))
            self.model.zero_grad()
            loss = self.loss_fn(det_bboxes[0][:,-1])
            loss.backward()
            if self.grad is None:
                self.grad = x_adv.grad
            else:
                self.grad += self.grad * self.momentum +  x_adv.grad / torch.norm(x_adv.grad, p=1)
            X_adv = x_adv + self.step_size * self.grad.sign()
            eta = X_adv - x_ori
            if is_universal:
                eta = torch.mean(eta.detach_(),dim=0)
            x_adv = torch.clamp(x_ori + mask * eta, min=x_min, max=x_max).detach_()
            x_adv.requires_grad = True
            data['img'] = x_adv
            data = test_pipeline(data)
        self.model.zero_grad()
        self.perturbation = eta
        x_adv = x_adv.cpu().detach().clone().squeeze(0).numpy().transpose(1, 2, 0)
        x_adv_cv2 = cv2.cvtColor(x_adv, cv2.COLOR_RGB2BGR)
        return x_adv, x_adv_cv2