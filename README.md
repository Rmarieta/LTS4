# Building and Explainability Analysis of a Seizure Classifier from EEG Recordings Using Graph Signal Processing

This repository contains the code for the implementation of the classifier and its interpretation described in the [project report](report.pdf). The different parts of the implementation involve the following steps :

<ul>
<li>Extraction and formatting of the seizures from the dataset. Run <code>data_preparation/build_data.py</code> to execute this step.</li>
<li>Computation of the graphs from the signals. Run <code>learn_graph_laplacian/laplacian.py</code> to compute the adjacency matrix, and <code>learn_graph_laplacian/covariance.py</code> to compute the covariance matrix.</li>
<li>Training of the classifier. All the classifiers can be found in <code>classifier/</code>.</li>
<li>The explainability analysis.</li>
</ul>

## Dataset

Read <code>sh_scripts/README.md</code> to find out how to download the dataset.

The current implementation is based on the TUH EEG Corpus dataset v1.5.2 released in 2020. The dataset can be downloaded following the instructions on <a href="https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml">TUH EEG dataset</a>. 

The dataset can be easily downloaded using <em>Rsync</em> on Linux. The shell scripts containing the commands to execute the download can be found in sh_scripts/. If trying to download from Windows, <em>Rsync</em> is unfortunately not available, and you might need the help of <em>MobaXterm</em> to be able to run <em>Rsync</em>. A tutorial explaining how to do this can be found <a href="https://isip.piconepress.com/courses/temple/ece_1111/resources/tutorials/tips_mobaxterm/02_install_and_rsync_nedc_v00.mp4">here</a>. 

## Set-up and Requirements

Due to the high amount of data, high computational power might be required for some steps of the implementation, in particular the computation of the adjacency matrix, and the training of the classifier. For that reason, the SCITAS EPFL cluster was used to run the heavy scripts. 
Find <a href="https://scitas-data.epfl.ch/confluence/display/DOC/Connecting+to+the+clusters#Connectingtotheclusters-Step-by-stepguide">here</a> how to set-up your access to the cluster (<code>fidis.epfl.ch</code> works perfectly) and <a href="https://scitas-data.epfl.ch/confluence/display/DOC/Using+the+clusters">here</a> how to use the clusters. If using VS Code, you can use the <em>Remote Explorer</em> extension to set-up the SSH connection and access the interface and your scripts on the cluster without having to navigate through them with the terminal. When running a python script on the cluster, you first need to allocate the power and time desired for the run, therefore you will have to run the appropriate scripts in <code>scitas_run/</code> to run the python scripts instead of using the terminal command. The cluster does not offer the option to directly plot an output with <em>Matplotlib</em>, so you might need to transfer your outputs of the graph computation step on your local device to plot graphs or plot the available plots in the explainability step.

To download packages on the cluster, you will have to create your virtual environment (<a>tutorial</a>) and download the required packages on your activated virtual environment. For example for numpy, run (in <code>home/LTS4/</code>) : 
```
# Activate your virtual environment (adapt with your env name)
source ../rma_env/bin/activate
# Download the package (no_cache_dir option required on the cluster)
pip install --no_cache_dir numpy
```

The file <code>requirements.txt</code> can be run with the command below to install the dependencies on your local device (some might have been forgotten).
```
pip install requirements.txt
```

## Seizure Extraction

Read <code>data_preparation/README.md</code> to find out how to extract the seizure types of interest.

This step is an adaptation of the implementation of the work found on this <a href= "https://github.com/IBM/seizure-type-classification-tuh">Github repository</a>, which itself is the implementation of the the work in <em>SeizureNet: Multi-Spectral Deep Feature Learning for Seizure Type Classification</em> (Asif
et al., 2020). It provides a way to extract the chops of recordings during which a seizure event occurred and retrieve them by type.

## Graph Computation

Read <code>learn_graph_laplacian/README.md</code> to find out how to compute the different graphs.

A graph representation of the connectivity between the 20-channel signals is computed with 2 different techniques :
<ul>
<li>Covariance matrix</li>
<li>Adjacency matrix by computing it back from the learned Laplacian matrix. Implementation of the framework described in <em>Learning Laplacian Matrix in Smooth Graph Signal Representations</em> (Dong
et al., 2016). Their code can be found <a href="https://github.com/TheShadow29/Learn-Graph-Laplacian">here</a>.
</ul>

## Classification and Explainability

Read <code>classifier/README.md</code> to find out how to use the multiple classifiers and explain their output.

Multiple classifiers can be tried here, simple classifiers (kNN, Bayes, Logistic Regression, Decision Tree, SVM) can be trained running <code>classifier/graph_classifier.py</code>. The Feed-forward Neural Network running <code>classifier/FC_NN.py</code>, the Convolutional Neural Network using <code>classifier/CNN.py</code>, and the 2-channel Convolutional Neural Network running <code>classifier/dual_CNN.py</code>.

The model that yielded the best accuracy after training was obtained using the dual CNN, combining both the covariance and the adjacency matrix graph representations into the same input. The output of this model is explained using <a href="https://github.com/slundberg/shap">SHAP</a>'s DeepExplainer tool is also computed in <code>classifier/dual_CNN.py</code> using the appropriate arguments.
