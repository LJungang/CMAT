CUDA_VISIBLE_DEVICES=7 nohup python train_optim_text.py \
-cfg=text.yaml -s=./results/no_parallel_0216 \
-np >./results/no_parallel_2016.log 2>&1 &