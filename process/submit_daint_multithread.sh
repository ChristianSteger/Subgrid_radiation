#!/bin/bash -l
#SBATCH --job-name="subrad"
#SBATCH --account="pr133"
#SBATCH --time=01:58:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread
#SBATCH --output=subrad.o
#SBATCH --error=subrad.e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate raytracing
srun -u python compute_sw_dir_cor.py
# srun -u python compute_sky_view_factor.py
