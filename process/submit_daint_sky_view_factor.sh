#!/bin/bash -l
#SBATCH --job-name="sky_view_factor"
#SBATCH --account="pr133"
#SBATCH --time=01:57:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread
#SBATCH --output=sky_view_factor.o
#SBATCH --error=sky_view_factor.e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate raytracing
srun -u python compute_sky_view_factor.py
