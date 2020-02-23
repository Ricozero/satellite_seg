import os
import sys
import threading
import numpy as np

from generate_data import generate_stat, generate_dataset
from inference import inference_gray

sys.path.append('./src')
from utils.imgformat import convert_label_to_vis

train_stage1_dir = "dataset/CCF-training"
train_stage2_dir = "dataset/CCF-training-Semi"

paths = []
for i in range(1, 3):
    paths.append((os.path.join(train_stage1_dir, str(i) + '.png'),
                    os.path.join(train_stage1_dir, str(i) + '_class.png'),
                    os.path.join(train_stage1_dir, str(i) + '_class_vis.png'),
                    os.path.join(train_stage1_dir, str(i) + '_class_crf.png'),
                    os.path.join(train_stage1_dir, str(i) + '_class_crf_vis.png')))
for i in range(1, 4):
    paths.append((os.path.join(train_stage2_dir, str(i) + '.png'),
                    os.path.join(train_stage2_dir, str(i) + '_class.png'),
                    os.path.join(train_stage2_dir, str(i) + '_class_vis.png'),
                    os.path.join(train_stage2_dir, str(i) + '_class_crf.png'),
                    os.path.join(train_stage2_dir, str(i) + '_class_crf_vis.png')))

def do_crf(gt_prob=0.92, sz=15):
    print('CRF processing...')
    # Python不是真正的多线程，用多线程不会更快，反而会OOM
    for path in paths:
        if not os.path.exists(path[3]):
            print('Inferencing ' + path[0] + '...')
            inference_gray(path[0], path[1], path[3], gt_prob, sz)
        else:
            print('Skip: ' + path[3])
    print()


def label_to_vis(use_crf):
    print('Visualizing...')
    for path in paths:
        # Ground truth
        if not os.path.exists(path[2]):
            print('Visualizing ' + path[1] + '...')
            convert_label_to_vis(path[1], path[2])
        else:
            print('Skip: ' + path[2])
        # CRF
        if use_crf:
            if not os.path.exists(path[4]):
                print('Visualizing ' + path[3] + '...')
                convert_label_to_vis(path[3], path[4])
            else:
                print('Skip: ' + path[4])
    print()


def generate_ccf(use_crf):
    print('Generating...')
    stat = generate_stat([path[3 if use_crf else 1] for path in paths])
    print(np.array(stat) * 1.0 / np.min(stat[np.nonzero(stat)]))
    dataset_dir = "dataset/s1s2-train" + ('-crf' if use_crf else '')
    if(not os.path.exists(dataset_dir)):
        print('Dataset directory: ' + dataset_dir)
        os.mkdir(dataset_dir)
        generate_dataset(dataset_dir, 320, [path[0] for path in paths], [path[3 if use_crf else 1] for path in paths])
    else:
        print('Skip: ' + dataset_dir)


if len(sys.argv) == 1:
    print('Please specify whether to use CRF!')
    exit()
use_crf = (sys.argv[1] == '1')
if use_crf:
    if len(sys.argv) == 4:
        do_crf(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        do_crf()
    else:
        print('CRF: Wrong argument amount, stop.')
        exit()

label_to_vis(use_crf)
generate_ccf(use_crf)
