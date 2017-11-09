#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -t 0

# Training
python bilstm.py --dynet-gpu --dynet-mem 11500 --dynet-autobatch 1

