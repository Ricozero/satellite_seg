#!/bin/bash

# 使用的模型的名字
model_name=pspnet-densenet-dstl

# 使用第多少epoch的模型
start_epoch=90
end_epoch=150
interval=20

# 预测的图片，可以多个
img_paths=dataset/Tairi_4cm_small1_down.png

# 此次预测可见的GPUID
CUDA_VISIBLE_DEVICES=0

echo "Start testing..."
date
for i in $(seq ${start_epoch} ${interval} ${end_epoch})
do
	model_path=snapshot/${model_name}/${i}.pkl
	save_dir=results/${model_name}/epoch${i}
	mkdir -p ${save_dir}
	mkdir -p ${save_dir}/temp
	for img_path in $img_paths
	do
		echo "Testing: $img_path"
		file_name=$(basename $img_path)
		# remove any suffix
		img_name=${file_name%.*}
		if [ -f ${save_dir}/${img_name}_pred.png ] ;then
			echo "File exists: ${save_dir}/${img_name}_pred.png"
			continue
		fi
		python src/processing/test.py --img_path $img_path \
						--out_path ${save_dir}/${img_name}_pred.png \
						--vis_out_path ${save_dir}/${img_name}_pred_vis.png \
						--gpu 0 \
						--batch_size 8 \
						--stride 64 \
						--model_path $model_path \
						--input_size 256 \
						--crop_scales 192 224 256 288 \
						--tempdir ${save_dir}/temp
	done
done
date
echo "Testing ended."