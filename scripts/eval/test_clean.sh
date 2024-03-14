
CUDA_VISIBLE_DEVICES=9 python detlib/mmocr/tools/test.py \
detlib/mmocr/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py \
https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth \
--eval hmean-iou

