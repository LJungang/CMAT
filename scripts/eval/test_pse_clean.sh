CUDA_VISIBLE_DEVICES=1 python -W ignore detlib/mmocr/tools/test.py \
detlib/mmocr/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py \
https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth \
--eval hmean-iou \
--show --show-dir vis_test/custom
#>eval_pse_clean.log 2>&1 &

