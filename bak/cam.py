import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

from detector.utils import init_detector
from tools.loader import read_img_np_batch
from tools.det_utils import plot_boxes_cv2
from tools.parser import load_class_names
from tools.parser import ConfigParser
from tools.loader import dataLoader
from evaluate import UniversalPatchEvaluator

from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def draw_cam(img_tensor_batch, plot, save_dir, save_name):
    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = [ClassifierOutputTarget(80)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = cam(img_tensor_batch, targets=targets)[0, ...]

    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    plot_img = (plot / 255.).astype(np.float32)
    # plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
    visualization = show_cam_on_image(plot_img, grayscale_cam, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, save_name), visualization)


def detect(attacker, img_tensor_batch, model):
    all_preds = None
    preds, _ = model(img_tensor_batch)
    all_preds = attacker.merge_batch_pred(all_preds, preds)
    attacker.get_patch_pos_batch(all_preds)
    return preds


def draw_detection(attacker, img_tensor_batch, model, save_dir, save_name=None):
    preds = detect(attacker, img_tensor_batch, model)

    savename = None
    if save_name is not None:
        savename = os.path.join(save_dir, save_name)

    img_numpy, img_numpy_int8 = model.unnormalize(img_tensor_batch[0])
    plot = plot_boxes_cv2(img_numpy_int8, np.array(preds[0]), cls,
                          savename=savename)
    return plot


if __name__ == '__main__':
    class_file = '../configs/namefiles/coco.names'
    img_dir = './data/INRIAPerson/Test/pos/'
    patch_path = '../results/inria/gap/aug/v5/patch/1000_v5-aug.png'
    cfg = ConfigParser('./configs/cam/v5.yaml')
    print(cfg.DETECTOR.NAME)
    model = init_detector(cfg.DETECTOR.NAME[0], cfg.DETECTOR)
    # print(model.device, model.detector.model)
    target_layers = [model.detector.model[-2]]
    cls = load_class_names(class_file, trim=False)
    imgs =  os.listdir(img_dir)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = dataLoader(img_dir, batch_size=1, is_augment=False, input_size=cfg.DETECTOR.INPUT_SIZE)
    attacker = UniversalPatchEvaluator(cfg, patch_path, device)

    cur = 0
    model.requires_grad_(True)
    cam = EigenCAM(model.detector, target_layers, use_cuda=True)
    save_dir = f'./data/cam/{model.name}'
    os.makedirs(save_dir, exist_ok=True)

    for index, img_tensor_batch in tqdm(enumerate(loader)):
        img_tensor_batch = img_tensor_batch.to(device)
        # ----------------clean-----------------
        # plot = draw_detection(attacker, img_tensor_batch, model, save_dir, f'{cur}-clean-detect.png')
        plot = draw_detection(attacker, img_tensor_batch, model, save_dir)
        # cam = GradCAM(model=model.detector, target_layers=target_layers, use_cuda=True)
        draw_cam(img_tensor_batch, plot, save_dir, f'./{cur}-clean-cam.png')

        # ----------------adv-----------------
        img_adv_tensor, _ = attacker.uap_apply(img_tensor_batch)
        # plot = draw_detection(attacker, img_adv_tensor, model, save_dir, f'{cur}-adv-detect.png')
        plot = draw_detection(attacker, img_adv_tensor, model, save_dir)
        draw_cam(img_adv_tensor, plot, save_dir, f'./{cur}-adv-cam.png')
        cur += 1
        # break