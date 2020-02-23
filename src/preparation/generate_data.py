import os
import skimage.io as io
import numpy as np
import sys
from tqdm import tqdm
from collections import Counter

sys.path.append('./src')
from utils.imgformat import save_visual_gt

def auto_stride(patch_label):
	#x0:x1:x2:x3:x4
	#most_count_label= Counter(patch_label.flatten().tolist()).most_common(1)[0][0]
	most_count_label = np.argmax(np.bincount(patch_label.flatten().tolist()))
	if(most_count_label==0):
		stride = 256
	elif(most_count_label==1):
		stride = 32 #oversampling the most class in the testset(plant)
	elif(most_count_label==2):
		stride = 32
	elif(most_count_label==3):
		stride = 256
	elif(most_count_label==4):
		stride = 32
	else:
		print("Unknown label")
	return stride

def crop(image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir,crop_label=False):
	raw_img = io.imread(image_path)
	raw_img_class = io.imread(img_class_path)
	h,w = raw_img.shape[0],raw_img.shape[1]
	#stride_h = 15
	#stride_w = 15
	#n_rows = int(np.ceil((h - crop_size_h)/stride_h)) + 1
	#n_cols = int(np.ceil((w - crop_size_w)/stride_w)) + 1
	index = 0
	x2,y2 = 0,0
	x0,y0 = 0,0
	while(y2<h):
		while(x2<w):
			x1 = x0
			x2 = x1 + crop_size_w
			y1 = y0
			y2 = y1 +crop_size_h

			#print(x1,y1,x2,y2)

			if(x2>w or y2>h):
				x2 = min(x2,w)
				y2 = min(y2,h)
				if((x2-x1)>10 and (y2-y1)>10):
					backgroud = np.zeros((crop_size_h,crop_size_w,raw_img.shape[2]),dtype=np.uint8)
					backgroud[:y2-y1,:x2-x1] = raw_img[y1:y2,x1:x2]
					patch = backgroud

					backgroud_label = np.zeros((crop_size_h,crop_size_w),dtype=np.uint8)
					backgroud_label[:y2-y1,:x2-x1] = raw_img_class[y1:y2,x1:x2]
					patch_label = backgroud_label
				else:
					break
			else:
				patch = raw_img[y1:y2,x1:x2]
				patch_label = raw_img_class[y1:y2,x1:x2]
			#stride_h = auto_stride(patch_label)
			stride_h = crop_size_h
			stride_w = crop_size_w
			#print "current stride: ",stride_h
			x0 = x1 + stride_w

			io.imsave(os.path.join(save_dir,'img',prefix+"_%05d.png"%(index)),patch,check_contrast=False)
			io.imsave(os.path.join(save_dir,'label',prefix+"_%05d.png"%(index)),patch_label,check_contrast=False)
			save_visual_gt(os.path.join(save_dir,'visualize_gt'),patch_label,prefix,index)
			index = index + 1
		x0,x1,x2 = 0,0,0
		y0 = y1 + stride_h

def generate_trainval_list(pathdir):
	labels_img_paths = os.listdir(os.path.join(pathdir,'label'))
	labels_count_list=dict()
	print('Generating file list...')
	for labels_img_path in tqdm(labels_img_paths):
		label = io.imread(os.path.join(pathdir,'label',labels_img_path))
		most_count_label= np.argmax(np.bincount(label.flatten().tolist()))
		labels_count_list[labels_img_path] = most_count_label
	values= list(labels_count_list.values())
	count_dict= Counter(values)
	print('Label count: ' + count_dict)

def write_train_list(pathdir):
	labels_img_paths = os.listdir(os.path.join(pathdir,'label'))
	num_sets = len(labels_img_paths)
	indexs = list(range(num_sets))
	np.random.shuffle(indexs)
	train_set_num = 0.95 * num_sets
	train_f = open(os.path.join(pathdir,'train.txt'),'w')
	val_f = open(os.path.join(pathdir,'val.txt'),'w')
	trainval_f = open(os.path.join(pathdir,'trainval.txt'),'w')
	for index in range(num_sets):
		if(index<train_set_num):
			print(labels_img_paths[indexs[index]], file=train_f)
		else:
			print(labels_img_paths[indexs[index]], file=val_f)
		print(labels_img_paths[indexs[index]], file=trainval_f)
	train_f.close()
	val_f.close()
	trainval_f.close()

def generate_stat(label_file_list):
	print('Generating statistics...')
	label_list=[]
	for label_file in tqdm(label_file_list):
		label = io.imread(label_file)
		label_list = label_list + label.flatten().tolist()
	count_label = np.bincount(label_list)
	return count_label

def generate_dataset(dataset_dir,crop_size,img_list,label_list):
	img_path=os.path.join(dataset_dir,'img')
	label_path =os.path.join(dataset_dir,'label')
	visualize_gt_path = os.path.join(dataset_dir,'visualize_gt')

	if(not os.path.exists(img_path)):
		os.mkdir(img_path)
	if(not os.path.exists(label_path)):
		os.mkdir(label_path)
	if(not os.path.exists(visualize_gt_path)):
		os.mkdir(visualize_gt_path)

	for i in range(len(img_list)):
		print('Cropping ' + img_list[i] + ' and its annotation...')
		crop(img_list[i],label_list[i],crop_size,crop_size,prefix='%d'%(i+1),save_dir=dataset_dir,crop_label=True)

	generate_trainval_list(dataset_dir)
	write_train_list(dataset_dir)