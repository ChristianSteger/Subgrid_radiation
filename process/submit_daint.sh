#!/bin/bash -l
#SBATCH --job-name="swsg_0"
#SBATCH --account="pr133"
#SBATCH --time=01:58:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=swsg_0.o
#SBATCH --error=swsg_0.e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate raytracing
srun -u python compute_subsolar_lookup.py
~                                        
