#!/bin/bash -l
#SBATCH --job-name="sw_dir_cor"
#SBATCH --account="pr133"
#SBATCH --time=03:58:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread
#SBATCH --output=sw_dir_cor.o
#SBATCH --error=sw_dir_cor.e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate raytracing
srun -u python compute_sw_dir_cor.py
