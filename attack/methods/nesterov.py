import torch
import numpy as np


class LinfNesterovAttack:
    def __init__(self):
        pass

    def sequential_non_targeted_attack(self, ori_tensor_batch, detector_attacker, detector, confs_thresh=None):

        print('conf thresh: ', confs_thresh)
        # interative attack
        # self.optim.zero_grad()
        losses = []
        for iter in range(detector_attacker.cfg.ATTACKER.ITER_STEP):
            # num_iter += 1
            adv_tensor_batch, patch_tmp = detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            preds, detections_with_grad = detector(adv_tensor_batch, confs_thresh=confs_thresh)
            bbox_num = torch.FloatTensor([len(pred) for pred in preds])
            # print('bbox num: ', bbox_num)
            if torch.sum(bbox_num) == 0: break
            detector.zero_grad()

            disappear_loss, update_func = self.loss_fn(detections_with_grad * bbox_num)
            disappear_loss.backward()
            grad = patch_tmp.grad

            if hasattr(detector_attacker.cfg.ATTACKER, 'nesterov'):
                momentum_step = detector_attacker.cfg.ATTACKER.nesterov
                # momentum初期修正
                # if num_iter < 100:
                #     update /= (1 - momentum_step)
                # patch_tmp = patch_tmp - momentum_step * update * self.step_size
                update = self.step_size * grad
                l2 = torch.sqrt(torch.sum(torch.pow(update, 2))) / update.numel()
                print(l2)
                update /= l2
                # update = momentum_step * update_pre + self.step_size * grad
                # update_pre = copy.deepcopy(update)

            patch_tmp = update_func(patch_tmp, update)
            losses.append(float(disappear_loss))
            # min_max epsilon clamp of different channels
            patch_tmp = torch.clamp(patch_tmp, min=0, max=1)

        # patch_tmp = detector.unnormalize_tensor(patch_tmp.detach())

        return patch_tmp, np.mean(losses)