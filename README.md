# Install 

- step 1. please follow the installation instructions of T-SEA to create a conda environment

  ```bash
  conda create -n text-attack python=3.7
  conda activate text-attack
  pip install -r requirements.txt
  ```

- Step 2. please follow the build-from-source instructions to install mmocr

  ```bash
  pip install -U openmim
  mim install mmcv-full
  pip install mmdet
  cd detlib/mmocr
  pip install -r requirements.txt
  pip install -v -e .
  ```

  然后请回到项目根目录，将./base.py替换到conda环境的lib/python3.7/site-packages/mmdet/models/detectors/

- Step 3. please install the following packets

```python
 pip install Image
 pip install jupyter

#  if error about PIL exist, please uninstall pillow and re-install it with lower version
pip uninstall pillow
pip install "pillow<7"
```

## 攻击代码

首先，请修改configs/parallel.yaml的15行，在name中填上所需的检测器：

```bash
DETECTOR: #在这里填入需要的检测器
 NAME: ["PS_IC15"] #,"PS_CTW","PANET_IC15","PANET_CTW"]
 WEIGHT: [1.0, 1.0, 1.0] #Model loss Weight
```

#### 单模型 / 多模型 - 不综合模型loss：

请将weight设为全1.0

请运行train.sh，在tensorboard中，以及results/crop.log检查结果，训练得到的扰动保存在/results/crop。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_optim_text.py \
-cfg=parallel.yaml -s=./results/crop \
-np >./results/crop.log 2>&1 &
```

#### 多模型 - 综合模型loss：

请运行train_parallel.sh

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_parallel_text.py \
-cfg=parallel.yaml -s=./results/parallel \
-np >./results/parallel.log 2>&1 &
```

## 搜索权重

使用nnictl搜索最好的Model loss weight：

```
nnictl create --config ./nni_config.yaml 
```

## Evaluation

首先，修改detlib/mmocr/configs/_base_/det_datasets/icdar2015.py中的data_root为数据集目录，修改test部分的pipeline：

```python
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/[json文件名]',
    img_prefix=f'{data_root}/[Image目录名]',
    pipeline=None)
```

接下来，就可以在项目根目录下运行test.sh，得到Psenet-IC15上的测试结果：

```bash
CUDA_VISIBLE_DEVICES=0 python detlib/mmocr/tools/test_attack.py \
detlib/mmocr/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_adv.py \
https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth \
--eval hmean-iou --perturbation [扰动路径]
--show --show-dir [保存路径]
```

其中，perturbation修改为**扰动文件的路径。**运行结束后，可以在show-dir下看到测试集图片的检测结果。
