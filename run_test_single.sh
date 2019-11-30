#!/bin/bash
img_name=1
test_img=dataset/CCF-testing/${img_name}.png
model_name=pspnet-densenet-dstl
start_epoch=90
end_epoch=150
interval=20

echo "start testing..."
date
for i in $(seq ${start_epoch} ${interval} ${end_epoch})
do
	model=snapshot/${model_name}/${i}.pkl
	save_dir=results/${model_name}/epoch${i}
	mkdir -p ${save_dir}
	mkdir -p ${save_dir}/temp
	if [ -f ${save_dir}/${img_name}_pred.png ] ;then
		echo file exists:${save_dir}/${img_name}_pred.png
		break
	fi
	python src/processing/test.py --img_path $test_img \
					--out_path ${save_dir}/${img_name}_pred.png \
					--vis_out_path ${save_dir}/${img_name}_pred_vis.png \
					--gpu 0 \
					--batch_size 8 \
					--stride 64 \
					--model_path $model \
					--input_size 256 \
					--crop_scales 192 224 256 288 320 \
					--tempdir ${save_dir}/temp
done
date
echo "testing ended."