CUDA_VISIBLE_DEVICES=0 nohup python train_parallel_text.py \
-cfg=parallel.yaml -s=./results/parallel \
-np >./results/parallel.log 2>&1 &