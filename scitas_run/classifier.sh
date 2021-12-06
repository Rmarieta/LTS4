#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 40000
#SBATCH --time 01:30:00 
#SBATCH --output ./cluster_output/classifier.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate
python classifier/graph_classifier.py --graph_dir './data/v1.5.2/graph_1_5_60000' --algo 'logit'

# To run : sbatch scitas_run/classifier.sh (in home/rmarieta/LTS4)