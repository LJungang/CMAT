import numpy as np
import cv2
import torch


def imnormalize(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    # print(img)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.squeeze(0)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor.unsqueeze(0)

# def random_transform(img_tensor, u=32):

#     if np.random.random()>0.5:
#         alpha=np.random.uniform(-u, u)/255
#         img_tensor+=alpha
#         img_tensor=img_tensor.clamp(min=-10, max=10)
    
#     if np.random.random()>0.5:
#         alpha=np.random.uniform(0.8, 1.2)
#         img_tensor*=alpha
#         img_tensor=img_tensor.clamp(min=-10, max=10)
    
#     if np.random.random()>0.5:
#         noise=torch.normal(0, 0.5, img_tensor.shape).cuda()
#         img_tensor+=noise
#         img_tensor=img_tensor.clamp(min=-10, max=10)

#     return img_tensor

def random_transform(img_tensor, u=8):

    if np.random.random()>0.5:
        alpha=np.random.uniform(-u, u)/255
        img_tensor+=alpha
        img_tensor=img_tensor.clamp(min=-10, max=10)
    
    if np.random.random()>0.5:
        alpha=np.random.uniform(0.9, 1.1)
        img_tensor*=alpha
        img_tensor=img_tensor.clamp(min=-10, max=10)
    
    if np.random.random()>0.5:
        noise=torch.normal(0, 0.15, img_tensor.shape).cuda()
        img_tensor+=noise
        img_tensor=img_tensor.clamp(min=-10, max=10)

    return img_tensor

