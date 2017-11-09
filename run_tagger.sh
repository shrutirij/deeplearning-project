#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -t 0

# Training
module load singularity
singularity shell /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img
python bilstm.py --dynet-gpu --dynet-mem 11500 --dynet-autobatch 1

