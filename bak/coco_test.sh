#!/bin/bash
# nohup bash train.sh 1 ps4 $(date '+%m-%d') >./results/inria/07-28/ps4.log 2>&1 &
gg="-g"
device=$1
patch_name=$2
save=$3
A=$4
targets=$5
ifng=$6
if [ $ifng == "-ng" ] ;then
    gg=""
fi
# 测试模型自身+迁移的效果
for config in ${A[@]}
do
  echo $config
  echo "./results/${patch_name}.png"
  for target in ${targets[@]}
  do
    cmd="CUDA_VISIBLE_DEVICES=${device} nohup python evaluate.py $gg\
    -p ./results/${patch_name}.png \
    -cfg ./configs/${config}.yaml \
    -lp /home/chenziyan/work/BaseDetectionAttack/data/coco/${target}/${target}2017/labels \
    -dr /home/chenziyan/work/BaseDetectionAttack/data/coco/${target}/${target}2017/pos \
    -s /home/chenziyan/work/BaseDetectionAttack/data/${save}/${target} \
    -e 0 &"
    echo $cmd
    eval $cmd
  done
done
