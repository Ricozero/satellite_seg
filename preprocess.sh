#!/bin/bash

ProjectDir=.

echo "preprocess the training dataset with crf..."

#stage2
train_dir2=${ProjectDir}/dataset/CCF-training-Semi
python src/preparation/inference.py ${train_dir2}/1.png ${train_dir2}/1_visual.png ${train_dir2}/1_visual_crf.png 0.95 5 &
python src/preparation/inference.py ${train_dir2}/2.png ${train_dir2}/2_visual.png ${train_dir2}/2_visual_crf.png 0.95 5 &
python src/preparation/inference.py ${train_dir2}/3.png ${train_dir2}/3_visual.png ${train_dir2}/3_visual_crf.png 0.95 5 &

wait
echo "generating dataset..."
python src/processing/generate_data.py
echo "Done"