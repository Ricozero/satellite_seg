import os
import sys
import threading
import numpy as np

from generate_data import generate_stat, generate_dataset
from preprocess import convert_label_to_vis, convert_vis_to_label
from inference import inference_gray

train_stage1_dir = "dataset/CCF-training"
train_stage2_dir = "dataset/CCF-training-Semi"


def do_crf(gt_prob=0.92, sz=15):
    print('CRF processing...')
    paths = []
    for i in range(1, 3):
        paths.append((os.path.join(train_stage1_dir, str(i) + '.png'),
                      os.path.join(train_stage1_dir, str(i) + '_class.png'),
                      os.path.join(train_stage1_dir, str(i) + '_class_crf.png'),
                      os.path.join(train_stage1_dir, str(i) + '_class_crf_vis.png')))
    for i in range(1, 4):
        paths.append((os.path.join(train_stage2_dir, str(i) + '.png'),
                      os.path.join(train_stage2_dir, str(i) + '_class.png'),
                      os.path.join(train_stage2_dir, str(i) + '_class_crf.png'),
                      os.path.join(train_stage2_dir, str(i) + '_class_crf_vis.png')))
    # Python不是真正的多线程，用多线程不会更快，反而会OOM
    for path in paths:
        print('Inferencing ' + path[0])
        inference_gray(path[0], path[1], path[2], gt_prob, sz)
    for path in paths:
        convert_label_to_vis(path[2], path[3])


def label_to_vis(use_crf):
    print('Visualizing...')
    for i in range(1, 3):
        convert_label_to_vis(os.path.join(train_stage1_dir, str(i) + '_class.png'), os.path.join(
            train_stage1_dir, str(i) + '_class_vis.png'))
        if use_crf:
            convert_label_to_vis(os.path.join(train_stage1_dir, str(i) + '_class_crf.png'), os.path.join(
                train_stage1_dir, str(i) + '_class_crf_vis.png'))
    for i in range(1, 4):
        convert_label_to_vis(os.path.join(train_stage2_dir, str(i) + '_class.png'), os.path.join(
            train_stage2_dir, str(i) + '_class_vis.png'))
        if use_crf:
            convert_label_to_vis(os.path.join(train_stage2_dir, str(i) + '_class_crf.png'), os.path.join(
                train_stage2_dir, str(i) + '_class_crf_vis.png'))


def generate_ccf(use_crf):
    img_list1 = [os.path.join(train_stage1_dir, '1.png'),
                  os.path.join(train_stage1_dir, '2.png'),]
    label_list1 = [os.path.join(train_stage1_dir, '1_class.png'),
                    os.path.join(train_stage1_dir, '2_class.png'),]
    label_crf_list1 = [os.path.join(train_stage1_dir, '1_class_crf.png'),
                        os.path.join(train_stage1_dir, '2_class_crf.png')]

    img_list2 = [os.path.join(train_stage2_dir, '1.png'),
                  os.path.join(train_stage2_dir, '2.png'),
                  os.path.join(train_stage2_dir, '3.png')]
    label_list2 = [os.path.join(train_stage2_dir, '1_class.png'),
                    os.path.join(train_stage2_dir, '2_class.png'),
                    os.path.join(train_stage2_dir, '3_class.png')]
    label_crf_list2 = [os.path.join(train_stage2_dir, '1_class_crf.png'),
                        os.path.join(train_stage2_dir, '2_class_crf.png'),
                        os.path.join(train_stage2_dir, '3_class_crf.png')]

    print('Generating...')
    if use_crf:
        stat = generate_stat(label_crf_list1 + label_crf_list2)
        print("crf stage1&stage2 rate: ", np.array(stat) * 1.0 / np.min(stat[np.nonzero(stat)]))
        dataset_dir = "dataset/stage1-stage2-train-crf" # [4 4 6 1 1]
        if(not os.path.exists(dataset_dir)):
            os.mkdir(dataset_dir)
            print("create dataset stage1-stage2-train-crf...")
            # {0: 654, 2: 568, 1: 499, 4: 91, 3: 68})
            generate_dataset(dataset_dir, 320, img_list1 + img_list2, label_crf_list1 + label_crf_list2)
        else:
            print("dataset stage1-stage2-train-crf exists, pass!")
    else:
        stat = generate_stat(label_list1 + label_list2)
        print("stage1&stage2 rate: ", np.array(stat) * 1.0 / np.min(stat[np.nonzero(stat)]))
        dataset_dir = "dataset/stage1-stage2-train"
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
            print("create dataset stage1-stage2-train...")
            generate_dataset(dataset_dir, 320, img_list1 + img_list2, label_list1 + label_list2)
        else:
            print("dataset stage1-stage2-train exists, pass!")


use_crf = (sys.argv[1] == '1')
if use_crf:
    if len(sys.argv) == 4:
        do_crf(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        do_crf()
    else:
        print("CRF: Wrong argument amount, stop.")
        exit()

label_to_vis(use_crf)
generate_ccf(use_crf)
