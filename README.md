
<div align="center">
  <h2>
    <a href="https://ljungang.github.io/CMAT/">
      CMAT: Cross-Model Adversarial Texture for Scanned Document Privacy Protection
    </a>
  </h2>
</div>

<p align="center">
  <a href="https://openreview.net/profile?id=~Xiaoyu_Ye1">Xiaoyu Ye</a><sup>1</sup>,
  <a href="https://jrx.cueb.edu.cn//szll/jsml/2adaa433df324c97a51d0b547256d7b1.htm">Jingjing Yu</a><sup>2*</sup>,
  <a href="https://github.com/LJungang">Jungang Li</a><sup>1</sup>,
    <a href="https://tsukasane.github.io/">Yiwen Zhao</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Peking University&nbsp;&nbsp;
  <sup>2</sup>Capital University of Economics and Business
</p>

<p align="center">
  <em>* Corresponding Author</em>
</p>

<div align="center">
  <p align="center">
    &nbsp&nbsp📑 <a href="https://doi.org/10.1016/j.asoc.2025.114353"><b>Paper (Applied Soft Computing 2025)</b></a>&nbsp&nbsp |
    &nbsp&nbsp📂 <a href="https://drive.google.com/file/d/1r4scB0HvLdz1NLinWI0KClqyyF8UrWnN/view?usp=drive_link"><b>Dataset (AdvDocument)</b></a>&nbsp&nbsp |
    &nbsp&nbsp🏠 <a href="https://ljungang.github.io/CMAT/"><b>Project Page</b></a>&nbsp&nbsp
  </p>

  <p align="center">
    This repository provides the official implementation of CMAT, a cross-model adversarial texture designed to protect scanned document privacy by degrading OCR text detection performance while maintaining human readability.
  </p>

  <p align="center">
    If you find this work useful, please consider giving us a ⭐ to support our research.
  </p>
</div>

---

## 📌 Method Overview

<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S1568494625016667-gr2_lrg.jpg" width="85%">
</p>

<p align="center">
  <em>Our Cross-Model Adversarial Texture (CMAT) generation process consists of three steps: (1) Preparation (dataset processing & attack initialization); (2) Cross-Model Attack (toroidal cropping + weighted loss fusion across detectors); (3) Protection (applying CMAT to documents so that detectors fail to locate text).</em>
</p>

---

## 📚 Contents

- [Installation](#installation)
- [Attack Usage](#attack-usage)
- [Loss Weight Search (NNI)](#loss-weight-search-nni)
- [Evaluation](#evaluation)
- [AdvDocument Dataset](#advdocument-dataset)
- [Citation](#citation)

---

## Installation

### Step 1: Create Conda Environment (T-SEA)

```bash
conda create -n text-attack python=3.7
conda activate text-attack
pip install -r requirements.txt
```

---

### Step 2: Install MMOCR from Source

```bash
pip install -U openmim
mim install mmcv-full
pip install mmdet

cd detlib/mmocr
pip install -r requirements.txt
pip install -v -e .
```

Then replace:

- `lib/python3.7/site-packages/mmdet/models/detectors/base.py`

with:

- `./base.py` (provided in the repository)

---

### Step 3: Additional Python Packages

```bash
pip install Image
pip install jupyter

# If PIL errors occur:
pip uninstall pillow
pip install "pillow<7"
```

---

##  Attack Usage

### Configure Detectors

Edit `configs/parallel.yaml`:

```yaml
DETECTOR:
  NAME: ["PS_IC15"]  # ,"PS_CTW","PANET_IC15","PANET_CTW"]
  WEIGHT: [1.0, 1.0, 1.0]
```

---

### Single / Multi-Model (Equal Weights)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_optim_text.py \
  -cfg=parallel.yaml -s=./results/crop \
  -np > ./results/crop.log 2>&1 &
```

Outputs stored in:

- `results/crop.log`
- `results/crop/` (perturbation)

---

### Multi-Model (Weighted Loss Fusion)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_parallel_text.py \
  -cfg=parallel.yaml -s=./results/parallel \
  -np > ./results/parallel.log 2>&1 &
```

---

##  Loss Weight Search (NNI)

```bash
nnictl create --config ./nni_config.yaml
```

NNI automatically explores detector loss weight configurations and reports optimal settings.

---

##  Evaluation

Modify:

```
detlib/mmocr/configs/_base_/det_datasets/icdar2015.py
```

Set dataset paths and update:

```python
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/[json file name]',
    img_prefix=f'{data_root}/[Image dir name]',
    pipeline=None
)
```

Run evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python detlib/mmocr/tools/test_attack.py \
  detlib/mmocr/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_adv.py \
  https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth \
  --eval hmean-iou \
  --perturbation [perturbation file path] \
  --show --show-dir [output directory]
```

---

##  AdvDocument Dataset

We release the **AdvDocument** dataset used in our experiments (COCO format):

 **Download:**  
https://drive.google.com/file/d/1r4scB0HvLdz1NLinWI0KClqyyF8UrWnN/view?usp=drive_link

### Example Display

![AdvDocument Example](image/README/1710387307709.png)

---

##  Citation

If you use CMAT or AdvDocument in your research, please cite:

```latex
@article{YE2025114353,
  title   = {CMAT: A cross-model adversarial texture for scanned document privacy protection},
  journal = {Applied Soft Computing},
  pages   = {114353},
  year    = {2025},
  issn    = {1568-4946},
  doi     = {https://doi.org/10.1016/j.asoc.2025.114353},
  url     = {https://www.sciencedirect.com/science/article/pii/S1568494625016667},
  author  = {Xiaoyu Ye and Jingjing Yu and Jungang Li and Yiwen Zhao},
  keywords = {Document active privacy protection, Adversarial texture, Cross-model texture},
}
```
