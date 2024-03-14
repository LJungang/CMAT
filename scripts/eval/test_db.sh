
CUDA_VISIBLE_DEVICES=0 nohup python -W ignore detlib/mmocr/tools/test_attack.py \
detlib/mmocr/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015_adv.py \
https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth \
--eval hmean-iou --perturbation results/parallel_weight_0213_1/parallel.pth \
>eval_db_custom.log 2>&1 &