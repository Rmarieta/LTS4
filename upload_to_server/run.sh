#!/bin/bash
#SBATCH --chdir /scratch/rmarieta
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 00:30:00 

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
python ../../home/rmarieta/upload_data.py # arg1 arg2