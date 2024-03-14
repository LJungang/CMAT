from mmocr.utils.ocr import MMOCR
import torch
# Load models into memory
ocr = MMOCR(det='PS_IC15', recog=None)

# Attack
perturbation = torch.zeros((3, 256, 256), requires_grad=True)
boundaries, outs = ocr.attack_text('demo/demo_text_det.jpg', output='demo/', export='demo/', perturbation=perturbation)[0]
print(outs.shape)
