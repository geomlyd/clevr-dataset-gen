#!/bin/bash -x
set -e

output_dir=$1
slurm_dir=${output_dir}/slurm
echo $slurm_dir
mkdir -p $slurm_dir

desired_partitions=1080-hi,1080-lo,a40-lo,a40-hi,3090-lo,veltins,tsingtao,\
kriek,astra,schlunz,arctic

num_videos=10
num_jobs=4
video_gen_config=/home/lydakis/work/clevr-dataset-gen/video_gen_cfgs/gen_easiest.json
split="testing"
prefix="more"
runtime=01:00:00
date_time=$(date +"%Y_%m_%d_%Hh_%Mm_%Ss")

for i in $(seq 1 $num_jobs);
do
    sbatch --partition=$desired_partitions --mem=16G -c 4 --time=$runtime \
        --gres=gpu:1 --output="$slurm_dir/slurm_log_job_${i}_${date_time}.txt" \
        --job-name=video_gen_$prefix_$split_$i \
        scripts/sbatch_video_gen.sh $output_dir/$i \
        $num_videos $video_gen_config $split $prefix
done