import copy
import os
import numpy as np
from tools.loader import dataLoader
from tqdm import tqdm
from tools import FormatConverter
import cv2

fp = './data/INRIAPerson/Train/labels/yolo-label'
save_path = './data/INRIAPerson/Train/labels/yolo-labels'
im_path = './data/INRIAPerson/Train/pos'
os.makedirs(save_path, exist_ok=True)
labs = os.listdir(fp)
for lab in labs:
    s_path = os.path.join(save_path, lab)
    path = os.path.join(fp, lab)

    if not os.path.getsize(path):
        continue
    # im = cv2.imread(os.path.join(im_path, lab.replace('txt', 'png')))
    s = np.loadtxt(path)
    if s.ndim == 1:
        s = s[np.newaxis, :]
    x = copy.deepcopy(s)
    print(s)
    x[:, 1] = np.clip(s[:, 1] - s[:, 3]/2, a_min=0, a_max=1)
    x[:, 2] = np.clip(s[:, 2] - s[:, 4]/2, a_min=0, a_max=1)
    x[:, 3] = np.clip(s[:, 1] + s[:, 3]/2, a_min=0, a_max=1)
    x[:, 4] = np.clip(s[:, 2] + s[:, 4]/2, a_min=0, a_max=1)
    print(x)
    np.savetxt(s_path, x)
# data_root = './data/INRIAPerson/Train/pos'
# data_sampler = None
# data_loader = dataLoader(data_root, [416, 416], is_augment=True,
#                              batch_size=1, sampler=data_sampler)
#
# for index, img_tensor_batch in tqdm(enumerate(data_loader)):
#     bgr_im_int8 = FormatConverter.tensor2numpy_cv2(img_tensor_batch[0])
#     cv2.imwrite(f'./data/test/aug/{index}.png', bgr_im_int8)
