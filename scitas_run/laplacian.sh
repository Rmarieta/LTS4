#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 40000
#SBATCH --time 20:30:00 
#SBATCH --output ./cluster_output/lapl.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate
python learn_graph_laplacian/laplacian.py --graph_dir './data/v1.5.2/graph_1_5_60000'

# To run : sbatch scitas_run/laplacian.sh (in home/rmarieta/LTS4)
# 7h30 with 40000 of RAM and 1 node/task/cpu was just enough for 'dev' dataset (1/6 of 'train')