CUDA_VISIBLE_DEVICES=0 python -W ignore detlib/mmocr/tools/test_attack.py \
detlib/mmocr/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_adv.py \
https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth \
--eval hmean-iou --perturbation results/parallel_weight_0213_1/parallel.pth \
--show --show-dir vis_test/custom_pse
#>eval_pse_custom.log 2>&1 &
