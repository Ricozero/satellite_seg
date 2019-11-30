import os
import numpy as np

from generate_data import generate_stat, generate_dataset
from preprocess import convert_label_to_vis,convert_vis_to_label

training_data_stage1_dir="dataset/CCF-training"
training_data_stage2_dir="dataset/CCF-training-Semi"

def generate_ccf():
    convert_label_to_vis(os.path.join(training_data_stage1_dir,'1_class.png'),os.path.join(training_data_stage1_dir,'1_class_vis.png'))
    convert_label_to_vis(os.path.join(training_data_stage1_dir,'2_class.png'),os.path.join(training_data_stage1_dir,'2_class_vis.png'))

    convert_label_to_vis(os.path.join(training_data_stage2_dir,'1_class.png'),os.path.join(training_data_stage2_dir,'1_class_vis.png'))
    convert_label_to_vis(os.path.join(training_data_stage2_dir,'2_class.png'),os.path.join(training_data_stage2_dir,'2_class_vis.png'))
    convert_label_to_vis(os.path.join(training_data_stage2_dir,'3_class.png'),os.path.join(training_data_stage2_dir,'3_class_vis.png'))

    ##save crf label
    #convert_vis_to_label(os.path.join(training_data_stage1_dir,'1_visual_crf.png'),os.path.join(training_data_stage1_dir,'1_class_crf.png'))
    #convert_vis_to_label(os.path.join(training_data_stage1_dir,'2_visual_crf.png'),os.path.join(training_data_stage1_dir,'2_class_crf.png'))

    convert_vis_to_label(os.path.join(training_data_stage2_dir,'1_visual_crf.png'),os.path.join(training_data_stage2_dir,'1_class_crf.png'))
    convert_vis_to_label(os.path.join(training_data_stage2_dir,'2_visual_crf.png'),os.path.join(training_data_stage2_dir,'2_class_crf.png'))
    convert_vis_to_label(os.path.join(training_data_stage2_dir,'3_visual_crf.png'),os.path.join(training_data_stage2_dir,'3_class_crf.png'))

    img_list_1=[os.path.join(training_data_stage1_dir,'1.png'),
                os.path.join(training_data_stage1_dir,'2.png'),
    ]
    label_list_1=[os.path.join(training_data_stage1_dir,'1_class.png'),
                os.path.join(training_data_stage1_dir,'2_class.png'),
    ]

    img_list_2=[os.path.join(training_data_stage2_dir,'1.png'),
                os.path.join(training_data_stage2_dir,'2.png'),
                os.path.join(training_data_stage2_dir,'3.png')
    ]
    label_list_2=[os.path.join(training_data_stage2_dir,'1_class.png'),
                os.path.join(training_data_stage2_dir,'2_class.png'),
                os.path.join(training_data_stage2_dir,'3_class.png')
    ]
    label_crf_list_2=[os.path.join(training_data_stage2_dir,'1_class_crf.png'),
                os.path.join(training_data_stage2_dir,'2_class_crf.png'),
                os.path.join(training_data_stage2_dir,'3_class_crf.png')
    ]

    #dataset s2
    stat=generate_stat(label_list_2)
    print("dataset s2 rate: ",np.array(stat)*1.0/np.min(stat[np.nonzero(stat)]))
    dataset_dir="dataset/stage2-train"
    if(not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)
        generate_dataset(dataset_dir,320,img_list_2,label_list_2)
        print("create dataset s2...")
    else:
        print("dataset s2 exists, pass!")

    #dataset s1s2
    stat=generate_stat(label_list_1+label_list_2)
    print("stage1&stage2 rate: ",np.array(stat)*1.0/np.min(stat[np.nonzero(stat)]))
    dataset_dir="dataset/stage1-stage2-train"
    if(not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)
        print("create dataset s1s2...")
        generate_dataset(dataset_dir,320,img_list_1+img_list_2,label_list_1+label_list_2)
    else:
        print("dataset s1s2 exists, pass!")

    #dataset s1s2-crf
    stat=generate_stat(label_list_1+label_crf_list_2)
    print("crf2 stage1&stage2 rate: ",np.array(stat)*1.0/np.min(stat[np.nonzero(stat)]))
    dataset_dir="dataset/stage1-stage2-train-crf2" #[4 4 6 1 1]
    if(not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)
        print("create dataset s1s2-crf2...")
        generate_dataset(dataset_dir,320,img_list_1+img_list_2,label_list_1+label_crf_list_2)#{0: 654, 2: 568, 1: 499, 4: 91, 3: 68})
    else:
        print("dataset s1s2-crf2 exists, pass!")

generate_ccf()