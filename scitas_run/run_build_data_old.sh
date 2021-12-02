#!/bin/bash
#SBATCH --chdir /scratch/rmarieta/cluster_output
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 00:30:00 
#SBATCH --output build_data_test.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../../../home/rmarieta/rma_env/bin/activate
python ../LTS4/data_preparation/build_data_test.py

# To run : sbatch run_build_data.sh