#!/bin/bash
# training on the weaker GPU's in the killable partition
#SBATCH --output=%j.out	# redirect stdout
#SBATCH --error=%j.out	# redirect stderr
#SBATCH --partition=killable
#SBATCH --time=1-0:00:00       # max time (minutes)
#SBATCH --nodes=1              # number of machines
#SBATCH --ntasks=1             # number of processes
#SBATCH --mem=8G               # memory
#SBATCH --cpus-per-task=4      # CPU cores per process
#SBATCH --gpus=1               # GPUs in total
#SBATCH --constraint="geforce_rtx_2080"
source ~/.bashrc
conda activate dev

python train.py $1