# Classification and Explainability Analysis

## Classification

Here, scripts are provided to run simple classifiers and neural networks on previously computed graphs. An example of an explainability analysis is then computed in the case of the dual CNN as it is the model that showed the best performance. Examples of python commands that can be used to run all of the classifiers can be found in <code>scitas_run/classifier.sh</code>.

### Simple classifiers

<code>graph_classifier.py</code> implements various simple models such as kNN, Bayes, Logistic Regression, Decision Tree and SVM using the <code>scikit-learn</code> library. For all the classifiers, you can select the graphs that you want to use as an input for the classification. Two examples of graphs, <code>data/v1.5.2/graph_cov_low_100</code> and <code>data/v1.5.2/graph_lapl_low_50</code> (computed on full recordings, only for FNSZ and GNSZ and with low-pass filtering the signal with cut-off frequency 100Hz and 50Hz) are included on Github to be able to run the explainability without having to compute graphs. When running the script, the following options can be selected :
- <code>--graph_dir</code> : path to the graphs to be classified.
- <code>--seizure_types</code> : types of seizures to use as classes.
- <code>--algo</code> : classification algorithm in 'bayes', 'kNN', 'SVM', 'tree', 'logit'.
- <code>--balanced</code> : to balance the training set (to be adapted if used on more than 2 classes).
- <code>--plot</code> : to plot the confusion matrix (only on local).
- <code>--is_cov</code> : set to True if using covariance graphs to avoid removing the over-connected adjacency matrices.
- <code>--over_conn</code> : to remove over-connected adjacency graphs.

A typical run would involve running the following command :
```
python classifier/graph_classifier.py --graph_dir './data/v1.5.2/graph_lapl_low_50' --algo 'logit' --seizure_types 'FNSZ' 'GNSZ' --cross_val 'False' --is_cov 'False' --over_conn 'True' --balanced 'True'
```

### Neural Networks

Here, 3 different networks are provided. A first feed-forward neural network, using the flattened upper triangular part of the graphs as input. This network is implemented in <code>FC_NN.py</code> and can be run with the following parameters :
- <code>--input_dir</code> : path to the graphs to be classified.
- <code>--seizure_types</code> : types of seizures to use as classes.
- <code>--is_cov</code> : set to True if using covariance graphs to avoid removing the over-connected adjacency matrices.
- <code>--plot</code> : to plot the training loss and the confusion matrix (only on local).
- <code>--over_conn</code> : to remove over-connected adjacency graphs.
- <code>--nb_epochs</code> : number of epochs for the training of the classifier.
- <code>--l_rate</code> : learning rate for the training.
- <code>--batch_size</code> : batch size for the training.
- <code>--save_model</code> : set to True to save the trained classifier (saved in <code>classifier/test_FC_NN.pt</code>, can be changed at the bottom of the script).

A typical run would involve running the following command :
```
python classifier/FC_NN.py --input_dir './data/v1.5.2/graph_lapl_low_50' --nb_epochs 40 --save_model 'False' --batch_size 30 --over_conn 'True' --save_model 'False' --is_cov 'False'
```

The second NN is a convolutional NN and uses the graphs in the form of a 2D image as input for the convolutions. This network is implemented in <code>CNN.py</code> and involves the same options as <code>FC_NN.py</code> except for a few additional options :
- <code>--upper</code> : set to True to set the lower triangular matrix to 0, as the graphs computed here are all symmetric.
- <code>--revert</code> : to compute the Laplacian matrix back from the adjacency and use it as a graph instead.

Here, only the implementation with the classes FNSZ and GNSZ is implemented, but it is straightforward to use the current implementation for more, if the output layer of the network is adapted as well as a few other conditions. A typical run would involve running the following command :
```
python classifier/CNN.py --input_dir './data/v1.5.2/graph_lapl_low_50' --nb_epochs 50 --save_model 'False' --batch_size 30 --upper 'True' --is_cov 'False' --revert 'False' --over_conn 'True'
```

The last NN is a slight adaptation of the CNN, with changes in the dimensions of the layers to use both the covariance and the adjacency matrices as inputs. This dual CNN is implemented in <code>dual_CNN.py</code> and uses the same options as <code>CNN.py</code> except that two input graph directories have to be provided :
- <code>--input_cov</code> : path to the covariance graphs to be classified.
- <code>--input_lapl</code> : path to the adjacency graphs to be classified.

The best performance is obtained with the dual CNN when using <code>data/v1.5.2/graph_lapl_low_50</code> and <code>data/v1.5.2/graph_cov_low_100</code> as 2-channel input. A typical run would involve running the following command :
```
python classifier/dual_CNN.py --input_cov './data/v1.5.2/graph_cov_low_100' --input_lapl './data/v1.5.2/graph_lapl_low_50' --nb_epochs 50 --save_model 'False' --upper 'True' --revert 'False' --batch_size 10 --over_conn 'True'
```

## Explainability Analysis

An explainability analysis is provided using the output of the best
performing model is explained, using <a href="https://github.com/slundberg/shap">SHAP</a>'s DeepExplainer tool. This analysis is also computed in <code>dual_CNN.py</code> using the two additional options provided :
- <code>--explain_model</code> : set to True to avoid the training and compute the explainability analysis.
- <code>--input_model</code> : path to the trained model whose output will be explained. 

Already trained <code>classifier/model_dual_CNN.pt</code> is provided here to avoid having to train the model. Make sure you use it with the same graphs it's been trained with (<code>data/v1.5.2/graph_lapl_low_50</code> and <code>data/v1.5.2/graph_cov_low_100</code> here). The explainability analysis might have to be run on the local device to allow for the plotting of the SHAP values. <code>dual_CNN.ipynb</code> is provided here to experiment with the explainability analysis if running on local to avoid having to recompute the SHAP values on all graphs for each try (this notebook might not use the latest version of the explainability analysis so you might have to adapt it with the functions of <code>dual_CNN.py</code>). A typical run would involve running the following command :
```
python classifier/dual_CNN.py --input_cov './data/v1.5.2/graph_cov_low_100' --input_lapl './data/v1.5.2/graph_lapl_low_50' --upper 'True' --revert 'False' --over_conn 'True' --explain_model 'True' --input_model './classifier/model_dual_CNN.pt'
```
You might encounter issues when importing SHAP on the cluster due to the version of numpy installed on the SCITAS cluster. To deal with that, try upgrading your numpy version on the cluster with the following :
```
pip install --no-cache-dir --upgrade numpy
```


