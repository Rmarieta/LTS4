#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100000
#SBATCH --time 20:30:00 
#SBATCH --output ./cluster_output/lapl_train.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate
python -u learn_graph_laplacian/test.py --graph_dir './data/v1.5.2/graph_lapl_nolow_1s_train' --low_pass 'False' --seizure_types 'FNSZ' 'GNSZ' --chop 'True'