import copy
import os
import re
import matplotlib.pyplot as plt
import numpy as np
fp = './data/inria/gap/aug/coco'
x = np.load(f'{fp}/x.npy')
y_test = np.load(f'{fp}/y_test.npy', allow_pickle=True).item()
y_train = np.load(f'{fp}/y_train.npy', allow_pickle=True).item()

print(len(y_test.values()), len(y_train.values()), len(x))

plt.figure()
for train_y, test_y in zip(y_train.values(), y_test.values()):
    print(x, train_y, len(x), len(train_y))
    # plt.plot(x, train_y)
    # print(x, train_y, test_y)
    plt.scatter(x, train_y, c='b', label='train')
    plt.scatter(x, test_y, c='r', label='test')
plt.legend()
plt.ylabel('mAP(%)')
plt.xlabel('# iteration')
plt.savefig(f'{fp}/gap.png', dpi=300)


def saveAPs():
    def readAP(p):
        with open(p, 'r') as f:
            mAP = f.readlines()[1].split('%')[0]
            # print(mAP)
        return float(mAP)

    y_test = {'yolov3': []}
    y_train = copy.deepcopy(y_test)
    x = []
    fp = './data/inria/gap/aug/inria0/all_data'
    all_dirs = os.listdir(fp)
    for edir in all_dirs:
        try:
            x.append(int(edir.split('_')[0]))
        except:
            continue

        y_test['yolov3'].append(readAP(os.path.join(*[fp, edir, 'test/yolov3/det-results/results.txt'])))
        y_train['yolov3'].append(readAP(os.path.join(*[fp, edir, 'yolov3/det-results/results.txt'])))

    np.save('../data/inria/gap/aug/x.npy', x)
    np.save('../data/inria/gap/aug/y_test.npy', y_test)
    np.save('../data/inria/gap/aug/y_train.npy', y_train)
    print(x, y_test['yolov3'], y_train['yolov3'])


# saveAPs()