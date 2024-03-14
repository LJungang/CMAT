import torch
import os

from .convertor import FormatConverter


def save_tensor(target_tensor, save_name, save_path='./'):
    os.makedirs(save_path, exist_ok=True)
    save_target = os.path.join(save_path, save_name)

    if save_name.split('.')[-1] == 'pth':
        torch.save(target_tensor, save_target)
    else:
        if target_tensor.ndim == 4:
            target_tensor = target_tensor.squeeze(0)
        FormatConverter.tensor2PIL(target_tensor).save(save_target)


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)