#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -t 0
#SBATCH --nodelist=compute-0-19
#SBATCH --gres=gpu:1

source ~/.bashrc
export LD_LIBRARY_PATH=/projects/tir2/users/srijhwan/dynet-base/dynet/build/dynet/:$LD_LIBRARY_PATH

# Training
module load singularity
singularity shell --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img
source activate dynetenv
python bilstm.py --dynet-gpu --dynet-mem 11000 --dynet-autobatch 1
