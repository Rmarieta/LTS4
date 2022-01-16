#!/bin/bash
#SBATCH --chdir /home/rmarieta/LTS4
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 120000
#SBATCH --time 03:50:00 
#SBATCH --output ./cluster_output/classifier_3.out

# Use modules to set the software environment
module load gcc/8.4.0
module load python/3.7.7
source ../rma_env/bin/activate # To activate the virtual environment

# To run the bash script : sbatch scitas_run/classifier.sh (in home/rmarieta/LTS4)

# To train simple models, one in ['bayes','kNN','SVM','tree','logit']
#python -u classifier/graph_classifier.py --graph_dir './data/v1.5.2/graph_lapl_low_50' --algo 'logit' --seizure_types 'FNSZ' 'GNSZ' --cross_val 'False' --is_cov 'False' --over_conn 'True'

python -u classifier/test.py --graph_dir './data/v1.5.2/graph_cov_1s' --algo 'logit' --seizure_types 'FNSZ' 'GNSZ' --cross_val 'False' --is_cov 'True' --over_conn 'True'

# To train the FC_NN :
#python -u classifier/FC_NN.py --input_dir './data/v1.5.2/graph_lapl_low_50' --nb_epochs 50 --save_model 'False' --batch_size 30 --over_conn 'True' --save_model 'False'

# To train the CNN :
#python -u classifier/CNN.py --input_dir './data/v1.5.2/graph_lapl_low_50' --nb_epochs 50 --save_model 'False' --batch_size 30 --upper 'True' --is_cov 'False' --revert 'False' --over_conn 'True'

# To train the dual CNN :
#python -u classifier/dual_CNN.py --input_cov './data/v1.5.2/graph_cov_low_100' --input_lapl './data/v1.5.2/graph_lapl_low_50' --nb_epochs 50 --save_model 'False' --upper 'True' --revert 'False' --batch_size 10 --over_conn 'True'

# To run explainability on previously trained dual CNN (issue with import shap on the cluster)
#python -u classifier/dual_CNN.py --input_cov './data/v1.5.2/graph_cov_low_100' --input_lapl './data/v1.5.2/graph_lapl_low_50' --upper 'True' --revert 'False' --over_conn 'True' --explain_model 'True' --input_model './classifier/model_dual_CNN.pt'
