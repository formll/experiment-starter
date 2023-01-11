#!/bin/bash
# training on CPU's (essentially unlimited)
#SBATCH --output=%j.out	# redirect stdout
#SBATCH --error=%j.out	# redirect stderr
#SBATCH --partition=cpu-killable
#SBATCH --time=2-0:00:00       # max time (minutes)
#SBATCH --nodes=1              # number of machines
#SBATCH --ntasks=1             # number of processes
#SBATCH --mem=8G               # memory
#SBATCH --cpus-per-task=2      # CPU cores per process
#SBATCH --gpus=0               # GPUs in total
source ~/.bashrc
conda activate dev

OMP_NUM_THREADS=2 python train.py $1