#!/bin/bash -x
set -e

output_dir=$1_$(hostname)
num_videos=$2
video_gen_config=$3
split=$4
file_prefix=$5

module load cuda/11.8
cd image_generation

old_blender="/home/lydakis/work/programs/\
blender-2.78c-linux-glibc219-x86_64/blender"

srun $old_blender --background --python render_videos.py -- --num_videos $num_videos \
    --use_gpu 1 --output_video_dir=$output_dir --output_gt_jsonl \
    movement_gt.jsonl --render_num_samples 64 --video_gen_config \
    $video_gen_config --split $split --filename_prefix $file_prefix