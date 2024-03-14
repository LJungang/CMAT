import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseAttacker(ABC):
    """An Attack Base Class"""

    def __init__(self, loss_func, norm: str, cfg, device: torch.device, detector_attacker):
        """

        :param loss_func:
        :param norm: str, [L0, L1, L2, L_infty]
        :param cfg:
        :param detector_attacker: this attacker should have attributes vlogger

        Args:
            loss_func ([torch.nn.Loss]): [a loss function to calculate the loss between the inputs and expeced outputs]
            norm (str, optional): [the attack norm and the choices are [L0, L1, L2, L_infty]]. Defaults to 'L_infty'.
            epsilons (float, optional): [the upper bound of perturbation]. Defaults to 0.05.
            max_iters (int, optional): [the maximum iteration number]. Defaults to 10.
            step_size (float, optional): [the step size of attack]. Defaults to 0.01.
            device ([type], optional): ['cpu' or 'cuda']. Defaults to None.
        """
        self.loss_fn = loss_func
        self.cfg = cfg
        self.detector_attacker = detector_attacker
        self.device = device
        self.norm = norm
        self.min_epsilon = -cfg.EPSILON
        self.max_epsilon = cfg.EPSILON
        self.max_iters = cfg.MAX_EPOCH
        self.iter_step = cfg.ITER_STEP
        self.step_size = cfg.STEP_SIZE
        self.class_id = cfg.TARGET_CLASS
        self.attack_class = cfg.ATTACK_CLASS

    def logger(self, loss_dict, adv_img, det_with_bbox, ori_with_bbox,name,conf_max,conf_min):
        #detector, adv_tensor_batch, bboxes, 
        vlogger = self.detector_attacker.vlogger
        # TODO: this is manually appointed logger iter
        if vlogger:
            vlogger.note_loss(loss_dict['det_loss'],conf_max,conf_min,name)
            if vlogger.iter % 37 == 0:
            
                vlogger.write_tensor(adv_img,"adv_img")
                vlogger.write_cv2(np.hstack([det_with_bbox,ori_with_bbox]),"{}_adv_and_original_results".format(name))
                #for i in range(conf_map.shape[1]):
                #    vlogger.write_tensor(conf_map[:,i,:,:],"conf_map_{}".format(i))

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()
            if hasattr(self.cfg, 'class_specify'):
                # TODO: only support filtering a single cls now
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(([conf[cls==attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()
            
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update(patch_clamp_=self.detector_attacker.patch_obj.clamp_)
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(loss_dict)
        return torch.tensor(losses).mean()
    
    def non_targeted_attack_for_text(self, ori_images, patch, detector,name):
        """
        attack function for text detection
        : param ori_images: clean image filenames
        : param patch: universal patch tensor with grad (shape:[3, height, width])
        : param detector: text detector 
        """
        losses = []
        for iter in range(self.iter_step):

            #input image name & patch, return conf_map (shape: [1,7,336,560])
            bound, conf_map, adv_img, adv_with_bbox, ori_with_bbox= detector.attack_text(ori_images, perturbation=patch)[0]
            #bbox = detector.readtext(ori_images)
            #print(bbox[0]['boundary_result'])
            #print(conf_map.shape)
            if name == "PS_IC15":
                confs=torch.sigmoid(conf_map[:,0,:,:])
            else:#db needs no sigmoid
                confs=conf_map[:,0,:,:]
            conf_max = confs.max()
            conf_min = confs.min()
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()
            losses.append(float(loss))
            #print("patch.grad",patch.grad)

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update(patch_clamp_=self.detector_attacker.patch_obj.clamp_)
        
        #tensorboard logger
        self.logger(loss_dict, adv_img,adv_with_bbox,ori_with_bbox,name,conf_max,conf_min)
        return torch.tensor(losses).mean()
    
    def parallel_non_targeted_attack_for_text(self, ori_images, patch, detector,name):
        """
        attack function for text detection
        : param ori_images: clean image filenames
        : param patch: universal patch tensor with grad (shape:[3, height, width])
        : param detector: text detector 
        """

        #input image name & patch, return conf_map (shape: [1,7,336,560])
        bound, conf_map, adv_img, adv_with_bbox, ori_with_bbox= detector.attack_text(ori_images, perturbation=patch)[0]
        #bbox = detector.readtext(ori_images)
        #print(bbox[0]['boundary_result'])
        #print(conf_map.shape)
        if name == "PS_IC15":
            confs=torch.sigmoid(conf_map[:,0,:,:])
        else:#db needs no sigmoid
            confs=conf_map[:,0,:,:]
        conf_max = confs.max()
        conf_min = confs.min()
        loss_dict = self.attack_loss(confs=confs)
        loss = loss_dict['loss']
        #loss.backward()
        self.logger(loss_dict, adv_img,adv_with_bbox,ori_with_bbox,name,conf_max,conf_min)
        return loss

    @abstractmethod
    def patch_update(self, **kwargs):
        pass

    @abstractmethod
    def attack_loss(self, confs):
        pass
