#!/bin/bash
#SBATCH --chdir /scratch/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 1
#SBATCH --mem 10000
#SBATCH --time 07:00:00 
#SBATCH --output ./cluster_output/build_data.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../../../home/rmarieta/rma_env/bin/activate
python data_preparation/build_data.py --output_dir '../../../home/rmarieta/LTS4/data'

# To run : sbatch scitas_run/scratch_build_data.sh (in scratch/rmarieta/LTS4)
# Output in home/rmarieta/LTS4/data/v1.5.2/raw_samples
# To read the output file on the terminal (live) : tail -f cluster_output/build_data.out