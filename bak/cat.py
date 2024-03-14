import os

import cv2
import numpy as np

fp = './results/inria/natural/'
targets = [os.path.join(fp, f) for f in os.listdir(fp)]
patch_size = 128
size = int(patch_size/2)
# res = np.zeros((patch_size*2, patch_size, 3))
res = None
test = None
for ind, target in enumerate(targets):
    print(target)
    if 'inria2' in target or 'inria0' in target:
        patch = cv2.imread(target)
        res = np.r_[res, patch] if res is not None else patch
        test = patch if test is None else test*0.5 + patch *0.5
#     print(ind+1, size)
#     res[ind*size:(ind+1)*size, ...] = patch[:size, ...]

cv2.imwrite('./target.png', res)
cv2.imwrite('./test.png', test)