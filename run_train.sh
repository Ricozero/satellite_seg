#!/bin/bash
count=$(ps -ef | grep visdom | grep -v "grep" | wc -l)
if [ ${count} -eq 0 ]; then
    python -m visdom.server
fi

traindir="dataset/dstl-train"
model_name=pspnet-densenet-dstl
CUDA_VISIBLE_DEVICES=0

echo "start training..."
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
echo "training ended."