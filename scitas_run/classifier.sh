#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 60000
#SBATCH --time 00:50:00 
#SBATCH --output ./cluster_output/classifier.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate
# To run simple models, one in ['bayes','kNN','SVM','tree','logit']
#python -u classifier/graph_classifier.py --graph_dir './data/v1.5.2/graph_cov_1s' --algo 'logit' --seizure_types 'FNSZ' 'GNSZ' --cross_val 'False' --is_cov 'True' --plot 'False'
# To run the CNN
python -u classifier/CNN.py --input_dir './data/v1.5.2/graph_lapl_low_50' --is_cov 'False' --nb_epochs 500 --save_model 'True' --upper 'True'
# To run : sbatch scitas_run/classifier.sh (in home/rmarieta/LTS4)