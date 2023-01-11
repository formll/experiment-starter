#!/bin/bash
# training on the stronger GPU's in the killable partition
#SBATCH --output=%j.out	# redirect stdout
#SBATCH --error=%j.out	# redirect stderr
#SBATCH --partition=killable
#SBATCH --time=1-0:00:00       # max time (minutes)
#SBATCH --nodes=1              # number of machines
#SBATCH --ntasks=1             # number of processes
#SBATCH --mem=32G               # memory
#SBATCH --cpus-per-task=4      # CPU cores per process
#SBATCH --gpus=1               # GPUs in total
#SBATCH --constraint="tesla_v100|geforce_rtx_3090|a5000"
source ~/.bashrc
conda activate dev

python train.py $1