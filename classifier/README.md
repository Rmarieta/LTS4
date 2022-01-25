# Classification and Explainability Analysis

Here, scripts are provided to run simple classifiers and neural networks on previously computed graphs. An example of an explainability analysis is then computed in the case of the dual CNN as it is the model that showed the best performance. Examples of python commands that can be used to run all of the classifiers can be found in <code>scitas_run/classifier.sh</code>.

<code>graph_classifier.py</code> implements various simple models such as kNN, Bayes, Logistic Regression, Decision Tree and SVM using the <code>scikit-learn</code> library. For all the classifiers, you can select the graphs that you want to use as an input for the classification. Two examples of graphs, <code>data/v1.5.2/graph_cov_low_100</code> and <code>data/v1.5.2/graph_lapl_low_50</code> (computed on full recordings, only for FNSZ and GNSZ and with low-pass filtering the signal with cut-off frequency 100Hz and 50Hz) are included on Github to be able to run an explainability without having to compute graphs.

Multiple classifiers can be tried here, simple classifiers (kNN, Bayes, Logistic Regression, Decision Tree, SVM) can be trained running <code>classifier/graph_classifier.py</code>. The Feed-forward Neural Network running <code>classifier/FC_NN.py</code>, the Convolutional Neural Network using <code>classifier/CNN.py</code>, and the 2-channel Convolutional Neural Network running <code>classifier/dual_CNN.py</code>.

The model that yielded the best accuracy after training was obtained using the dual CNN, combining both the covariance and the adjacency matrix graph representations into the same input. The output of this model is explained using <a href="https://github.com/slundberg/shap">SHAP</a>'s DeepExplainer tool is also computed in <code>classifier/dual_CNN.py</code> using the appropriate arguments.

```
code here
```

