#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80000
#SBATCH --time 12:30:00 
#SBATCH --output ./cluster_output/lapl_low_true.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate
python -u learn_graph_laplacian/laplacian_low_pass.py --graph_dir './data/v1.5.2/graph_lapl_low_50_true' --restrict_size 'False' --low_pass 'True' --seizure_types 'FNSZ' 'GNSZ'