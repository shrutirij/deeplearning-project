#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -t 0
#SBATCH --gres=gpu:1

source ~/.bashrc

# Training
source activate dynetenv
python bilstm.py --dynet-gpu --dynet-mem 11000 --dynet-autobatch 1
