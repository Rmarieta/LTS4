#!/bin/bash
#SBATCH --chdir /scratch/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4000
#SBATCH --time 00:30:00 
#SBATCH --output ./cluster_output/build_data_test.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../../../home/rmarieta/rma_env/bin/activate
python data_preparation/build_data_test.py --output_dir '../../../home/rmarieta/LTS4/data'

# To run : sbatch scitas_run/build_data_test.sh (in scratch/rmarieta/LTS4)