#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=15000
#SBATCH -t 0
#SBATCH --gres=gpu:1

source ~/.bashrc

# Training
python fixed_neurons.py --dynet-gpu --dynet-mem 10000 --dynet-autobatch 1
