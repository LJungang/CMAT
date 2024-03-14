CUDA_VISIBLE_DEVICES=1 nohup python detlib/mmocr/tools/test.py \
detlib/mmocr/configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py \
https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth \
--eval hmean-iou \
>eval_dbpp_clean.log 2>&1 &