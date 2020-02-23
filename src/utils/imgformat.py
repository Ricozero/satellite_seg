import skimage.io as io
import os
import numpy as np

def get_color_labels():
	return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128],[255,255,255]])

def encode_segmap(mask):
	mask = mask.astype(int)
	label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
	for i, label in enumerate(get_color_labels()):
		label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
	label_mask = label_mask.astype(int)
	return label_mask

def segmap(mask):
	label_colours = get_color_labels()
	r = mask.copy()
	g = mask.copy()
	b = mask.copy()
	for l in range(0, label_colours.shape[0]):
		r[mask == l] = label_colours[l, 0]
		g[mask == l] = label_colours[l, 1]
		b[mask == l] = label_colours[l, 2]

		rgb = np.zeros((mask.shape[0], mask.shape[1], 3),dtype=np.uint8)
		rgb[:, :, 0] = r
		rgb[:, :, 1] = g
		rgb[:, :, 2] = b
	return rgb

# gt = ground truth
def save_visual_gt(save_dir,mask,prefix,index):
	color_mask = segmap(mask)
	io.imsave(os.path.join(save_dir,prefix+"_%05d.png"%(index)),color_mask, check_contrast=False)

def save_gt_vis(input_path,output_path):
	raw_img_class = io.imread(input_path)
	color_mask = segmap(raw_img_class)
	io.imsave(output_path,color_mask)

def convert_label_to_vis(src_path,dst_path):
	label = io.imread(src_path)
	io.imsave(dst_path,segmap(label))
	
def convert_vis_to_label(src_path,dst_path):
	vis = io.imread(src_path)
	io.imsave(dst_path,encode_segmap(vis))
