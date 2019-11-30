#!/bin/bash
model_name=pspnet-densenet-dstl
start_epoch=90
end_epoch=150
interval=20

echo "Start testing..."
date
for i in $(seq ${start_epoch} ${interval} ${end_epoch})
do
	model_path=snapshot/${model_name}/${i}.pkl
	save_dir=results/${model_name}/epoch${i}
	mkdir -p ${save_dir}
	mkdir -p ${save_dir}/temp
	img_paths="dataset/Tairi_Google_20_small1.png
		dataset/Tairi_4cm_small1_small.png
		dataset/Tairi_2cm_small1_small.png"
	for img_path in $img_paths
	do
		file_name=$(basename $img_path)
		# remove any suffix
		img_name=${file_name%.*}
		echo $img_name
		if [ -f ${save_dir}/${img_name}_pred.png ] ;then
			echo ${save_dir}/${img_name}_pred.png
			echo "File exists: $img_path"
			continue
		fi
		echo "Testing $img_path"
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