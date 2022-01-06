#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80000
#SBATCH --time 01:30:00
#SBATCH --output ./cluster_output/cov.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate
python -u learn_graph_laplacian/covariance.py --graph_dir './data/v1.5.2/graph_cov_low' --seizure_types 'FNSZ' 'GNSZ'
#python -u learn_graph_laplacian/test.py
# To run : sbatch scitas_run/cov.sh (in home/rmarieta/LTS4)