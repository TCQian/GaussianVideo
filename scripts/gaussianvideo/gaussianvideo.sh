#!/bin/bash

data_path=$1
data_name=$2
num_points=$3
iterations=$4
num_frames=$5
start_frame=$6

CUDA_VISIBLE_DEVICES=0 python train_quantize_video.py -d $data_path \
--data_name $data_name --model_name GaussianVideo --num_points $num_points \
--iterations $iterations --num_frames $num_frames --start_frame $start_frame
