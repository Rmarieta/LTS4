#!/bin/bash
#SBATCH --chdir /scratch/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4000
#SBATCH --time 00:30:00 
#SBATCH --output ../cluster_output/test.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../../../home/rmarieta/rma_env/bin/activate
python data_preparation/build_data.py

# To run : sbatch scratch_build_data.sh (in scratch/rmarieta/cluster_output)
# Output in home/rmarieta/LTS4/data/v1.5.2/raw_samples