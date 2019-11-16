from preprocess import write_train_list,crop,generate_trainval_list
import os
import skimage.io as io
import numpy as np
from tqdm import tqdm

def generate_stat(label_file_list):
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

	for i in tqdm(range(len(img_list))):
		crop(img_list[i],label_list[i],crop_size,crop_size,prefix='%d'%(i+1),save_dir=dataset_dir,crop_label=True)

	generate_trainval_list(dataset_dir)
	write_train_list(dataset_dir)