#!/bin/bash

# 先在家目录下启动visdom server
count=$(ps -ef | grep visdom | grep -v "grep" | wc -l)
if [ ${count} -eq 0 ]; then
	pushd ~
	python -m visdom.server
	pushd
fi

# 训练集目录
traindir=dataset/dstl-train

# 训练模型的名字
model_name=pspnet-densenet-dstl

# 此次训练可见的GPUID
CUDA_VISIBLE_DEVICES=0

echo "Start training..."
date
python src/processing/train.py --arch ${model_name} \
				--img_rows 256 \
				--img_cols 256 \
				--n_epoch 150 \
				--l_rate 1e-3 \
				--batch_size 16 \
				--gpu 0 \
				--step 50 \
				--traindir ${traindir}
				#--snapshot snapshot/${model_name}/0.pkl \
				#--split "trainval"
date
echo "Training ended."