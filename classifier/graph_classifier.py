import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sn
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
import math
import random

def over_connected(graph, upper, is_cov, revert) :

    G = graph.flatten()
    cross_thr, full_thr = 90, 90
    # No such over-connected graphs in covariance matrices and not same thresholds with Laplacian (revert==True)
    if is_cov or revert : 
        return False
    # If on full symmetric matrix, the threshold count of pixels has to be doubled
    if not upper :
        cross_thr = 2*cross_thr
        full_thr = 2*full_thr
    if (G > 0.6).sum() >= cross_thr :
        return True
    elif (G > 0.4).sum() >= full_thr : 
        return True
    else : 
        return False

def load_graphs(input_dir, class_dict, is_covariance, over_conn) :

    data, data_labels = [], [] # data containing the graphs and data_labels the associated seizure type labels

    for szr_type in class_dict.keys() :
        szr_label = class_dict[szr_type]
        for _, _, files in os.walk(os.path.join(input_dir,szr_type)) :
            for npy_file in files :
                graph = np.load(os.path.join(input_dir,szr_type,npy_file))

                # To convert to Laplacian (if intended), diagonal would need to be included in flattened output
                # graph = np.diag(np.sum(graph,axis=1))-graph

                graph = graph/np.amax(graph.flatten())
                
                if over_conn : is_over_conn = over_connected(graph, upper=False, is_cov=is_covariance, revert=False)
                else : is_over_conn = False

                if not is_over_conn :
                    graph = graph[np.triu_indices(20, k = 1)]

                    data.append(graph.flatten()) # graph has to be flattened to be fed to the classifier
                    data_labels.append(szr_label)

    return np.array(data), np.array(data_labels)

def train_test_data(input_dir, class_dict, is_covariance, over_conn) :

    train, train_labels = load_graphs(os.path.join(input_dir,'train'), class_dict, is_covariance, over_conn)
    test, test_labels = load_graphs(os.path.join(input_dir,'dev'), class_dict, is_covariance, over_conn)

    return train, test, train_labels, test_labels

def oversample(data, labels) :

    os_data, os_labels = [], []

    # Oversampling (train set only) to have balanced classification without dropping information
    PD = pd.DataFrame(labels,columns=['label'])
    no_0, no_1 = len(PD[PD['label']==0]), len(PD[PD['label']==1])

    # Multiply the dataset by this ratio, then add (no_0 - R*no_1) randomly selected entries from the smallest dataset
    R = math.floor(no_0/no_1)

    trainset = []
    for i in range(len(data)) :
        if labels[i] == 1 : # Under-represented class (here the one with element 1):
            for r in range(R) : # Add each element R times
                os_data.append(data[i])
                os_labels.append(labels[i])
        else : # Only add once each element of the over-represented class
            os_data.append(data[i])
            os_labels.append(labels[i])

    # Compensate the remaining imbalance : draw (no_0 - R*no_1) elements from already present elements
    Add = random.sample(PD[PD['label']==1].index.to_list(),no_0 - R*no_1)
    for idx in Add :
        os_data.append(data[idx])
        os_labels.append(labels[idx])

    PD = pd.DataFrame(os_labels,columns=['label'])
    no_0, no_1 = len(PD[PD['label']==0]), len(PD[PD['label']==1])

    return os_data, os_labels

def classify(input_dir, szr_types, algo, cross_val, is_covariance, plot, balanced, over_conn) :

    class_dict = {}
    for i, szr_type in enumerate(szr_types) :
        class_dict[szr_type] = i
    
    train, test, train_labels, test_labels = train_test_data(input_dir, class_dict, is_covariance, over_conn)
    
    # Oversample the under-represented class (only relevant on training data)
    if balanced : 
        train, train_labels = oversample(train, train_labels)

    # Shuffle the datasets
    np.random.seed(2) # For reproducibility
    train, train_labels = shuffle(train, train_labels)
    test, test_labels = shuffle(test, test_labels)

    # Initialisation of the selected classification model

    if algo == 'bayes' : # Gaussian Naive Bayes
        # Supposes that all the features are independent (prediction can be poor when they're not which might be the case here)
        model = GaussianNB()

    elif algo == 'kNN' : # k-Nearest Neighbours
        model = KNeighborsClassifier(n_neighbors=3)

    elif algo == 'SVM' : # Support vector machine
        model = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    elif algo == 'tree' : # Decision tree
        model = DecisionTreeClassifier()

    else : # Multinomial logistic regression (algo == 'logit')
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    
    if not cross_val : # We train and test the model normally
    
        # Training of the classifier
        model.fit(train, train_labels)
    
        # Prediction of the classes
        train_preds = model.predict(train)
        test_preds = model.predict(test)
        
        F1 = 100 * f1_score(test_labels, test_preds, average='weighted')

        # Evaluate accuracy of the classifier
        print(f"({algo} classifier) \
            \nWeigthed F1-score : {round(F1,2)} %")

        acc_dict = {str(j): 0 for j in range(len(szr_types))}

        for i in range(len(test_preds)) :
            if test_preds[i] == test_labels[i] : acc_dict[str(test_labels[i])] += 1

        # Print accuracy for each class
        for i, szr_type in enumerate(szr_types) :

            acc = int(acc_dict[str(i)])/len([x for x in test_labels if x==i])
            print(f'Accuracy for {szr_type} : {100*round(acc,2)} %')

        C = confusion_matrix(test_labels,test_preds)
        print(f'\nConfusion matrix :\n{C}')

        df_cm = pd.DataFrame(C, index=szr_types, columns=szr_types)
        
        if plot :
            plt.figure(figsize=(4.3,4))
            sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', cbar=False)
            plt.title(f'Confusion ({algo}, train/test : {100*round(accuracy_score(train_labels, train_preds),2)}/{100*round(accuracy_score(test_labels, test_preds),2)} %)\nWeighted F1-score : {round(F1,2)} %')
            plt.ylabel('True label'); plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.show()

    else : # We compute the accuracy of the model with K-fold cross-validation
        
        k = 5
        kf = KFold(n_splits=k, shuffle=True)
        result = cross_val_score(model , train, train_labels, cv=kf)
        
        print(f"{k}-Fold Cross-Validation\n\nAccuracy of each split on training data with '{algo}' :\n{result}\n")
        print(f"Avg accuracy: {result.mean()}")
  
if __name__ == '__main__':
    
    # Run : python .\classifier\graph_classifier.py --graph_dir './data/v1.5.2/graph_avg_1_5' --seizure_types 'FNSZ' 'GNSZ' --algo 'logit' --cross_val False

    print('\n\nSTART\n\n')

    implemented_algos = ['bayes','kNN','SVM','tree','logit']

    parser = argparse.ArgumentParser(description='Build the graph classifier')
    parser.add_argument('--data_dir', default='./data', help='path to the dataset')
    known_args, _ = parser.parse_known_args()
    data_dir = known_args.data_dir

    parser.add_argument('--graph_dir', default=os.path.join(data_dir,'v1.5.2/graph_lapl_low_50'), help='path to the computed graphs')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ'], help="types of seizures to include in the classification, in the form --seizure_types 'BG' 'FNSZ' 'GNSZ'", nargs="+")
    parser.add_argument('--algo',default='bayes', help="pick the classification algorithm in \
         the following : "+str(implemented_algos))
    parser.add_argument('--cross_val',default=False, help="set to True (or 1) to perform a cross-validation", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--is_cov',default=False, help="set to True (or 1) if the graphs used are cov matrices", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--plot',default=False, help="set to True if not on the cluster to plot the confusion matrix", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--balanced',default=True, help="set to True to oversample under-represented class", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--over_conn',default=False, help="set to True to remove over-connected graphs", type=lambda x: (str(x).lower() in ['true','1']))

    args = parser.parse_args()

    graph_dir = args.graph_dir
    szr_types = args.seizure_types
    algo = args.algo
    cross_val = args.cross_val
    is_covariance = args.is_cov
    plot = args.plot
    balanced = args.balanced
    over_conn = args.over_conn

    if algo not in implemented_algos :
        print(f"The selected classification algorithm ('"+algo+"') is not available")
        exit()

    classify(graph_dir, szr_types, algo, cross_val, is_covariance, plot, balanced, over_conn)

    print('\n\nDONE\n\n')
