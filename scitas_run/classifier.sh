#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100000
#SBATCH --time 10:50:00 
#SBATCH --output ./cluster_output/1s_FC_NN.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate # To activate the virtual environment

# To run the bash script : sbatch scitas_run/classifier.sh (in home/rmarieta/LTS4)

# To run simple models, one in ['bayes','kNN','SVM','tree','logit']
# python -u classifier/graph_classifier.py --graph_dir './data/v1.5.2/graph_cov_1s' --algo 'logit' --seizure_types 'FNSZ' 'GNSZ' --cross_val 'False' --is_cov 'True' --plot 'False'

# To run the FC_NN :
python -u classifier/FC_NN.py --input_dir './data/v1.5.2/graph_lapl_nolow_1s' --nb_epochs 200 --save_model 'True' --batch_size 30

# To run the CNN :
#python -u classifier/CNN.py --input_dir './data/v1.5.2/graph_lapl_nolow_1s' --nb_epochs 200 --save_model 'True' --batch_size 30 --upper 'True' --is_cov 'False'

# To run the dual CNN :
#python -u classifier/dual_CNN.py --input_cov './data/v1.5.2/graph_cov_low_100' --input_lapl './data/v1.5.2/graph_lapl_low_50' --nb_epochs 200 --save_model 'False' --upper 'True'

