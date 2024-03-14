CUDA_VISIBLE_DEVICES=1 nohup python train_weighted_text.py \
-cfg=parallel.yaml -s=./results/custom_dataset_smaller \
>./results/custom_dataset_smaller.log 2>&1 &
