#!/bin/bash
#SBATCH --chdir /scratch/rmarieta/cluster_output
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 00:30:00 
#SBATCH --output test.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
python ../LTS4/data_preparation/test.py # arg1 arg2

# To run : sbatch run.sh