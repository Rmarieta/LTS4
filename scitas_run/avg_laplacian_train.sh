#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 90000
#SBATCH --time 08:30:00 
#SBATCH --output ./cluster_output/nonorm_dev.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate
python -u learn_graph_laplacian/avg_laplacian_dev.py --graph_dir './data/v1.5.2/graph_unnormal_dev' --restrict_size 'False' --seizure_types 'FNSZ' 'GNSZ'
# To run : sbatch scitas_run/avg_laplacian_train.sh (in home/rmarieta/LTS4)
# 7h30 with 40000 of RAM and 1 node/task/cpu was just enough for 'dev' dataset (with restrict_size = True), worked until 1/6 of 'train'