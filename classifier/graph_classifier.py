import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap
import seaborn as sn
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def load_graphs(input_dir, class_dict, is_covariance) :

    data, data_labels = [], [] # data containing the graphs and data_labels the associated seizure type labels

    for szr_type in class_dict.keys() :
        szr_label = class_dict[szr_type]
        for _, _, files in os.walk(os.path.join(input_dir,szr_type)) :
            for npy_file in files :
                graph = np.load(os.path.join(input_dir,szr_type,npy_file))

                graph = graph[np.triu_indices(20, k = 1)]
                if is_covariance : graph = graph/np.amax(graph.flatten())

                data.append(graph.flatten()) # graph has to be flattened to be fed to the classifier
                data_labels.append(szr_label)

    return np.array(data), np.array(data_labels)

def train_test_data(input_dir, class_dict, is_covariance) :

    train, train_labels = load_graphs(os.path.join(input_dir,'train'), class_dict, is_covariance)
    test, test_labels = load_graphs(os.path.join(input_dir,'dev'), class_dict, is_covariance)

    return train, test, train_labels, test_labels

def classify(input_dir, szr_types, algo, cross_val, is_covariance, plot) :

    class_dict = {}
    for i, szr_type in enumerate(szr_types) :
        class_dict[szr_type] = i
    
    train, test, train_labels, test_labels = train_test_data(input_dir, class_dict, is_covariance)

    # Shuffle the datasets (if of any use ?)
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
        # Evaluate accuracy of the classifier
        print('\nPredictions (test dataset) :\n',test_preds[:10],'\nGround truth labels :\n',test_labels[:10],'\n')
        print(f"Accuracy of the '{algo}' classifier :\n- training dataset : {100*round(accuracy_score(train_labels, train_preds),2)} % \
            \n- test dataset : {100*round(accuracy_score(test_labels, test_preds),2)} %")

        C = confusion_matrix(test_labels,test_preds)
        print(f'Confusion matrix :\n{C}\n')
        #disp = ConfusionMatrixDisplay(confusion_matrix=C)
        #disp.plot()

        df_cm = pd.DataFrame(C, index=szr_types, columns=szr_types)
        
        if plot :
            plt.figure(figsize=(5,4))
            #sn.set(font_scale=1.4) # for label size
            sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g') # font size
            plt.title(f'Confusion matrix ({algo}, train/test : {100*round(accuracy_score(train_labels, train_preds),2)}/{100*round(accuracy_score(test_labels, test_preds),2)} %)')
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
    
    ################################################################

    # WATCH OUT, MIGHT NEED TO CONVERT LAPLACIAN TO ADJACENCY MATRIX
    # OR, COMPUTE THE ADJACENCY MATRIX BACK FROM THE LAPLACIAN

    # A = -(L - np.diag(np.diag(L)))
    # A = A/np.amax(A.flatten())

    # Run : python .\classifier\graph_classifier.py --graph_dir './data/v1.5.2/graph_avg_1_5' --seizure_types 'FNSZ' 'GNSZ' --algo 'logit' --cross_val True

    ################################################################

    print('\n\nSTART\n\n')

    implemented_algos = ['bayes','kNN','SVM','tree','logit']

    parser = argparse.ArgumentParser(description='Build the graph classifier')
    parser.add_argument('--data_dir', default='./data', help='path to the dataset')
    known_args, _ = parser.parse_known_args()
    data_dir = known_args.data_dir

    parser.add_argument('--graph_dir', default=os.path.join(data_dir,'v1.5.2/graph_output'), help='path to the computed graphs')
    parser.add_argument('--seizure_types',default=['BG','FNSZ','GNSZ'], help="types of seizures to include in the classification, in the form --seizure_types 'BG' 'FNSZ' 'GNSZ'", nargs="+")
    parser.add_argument('--algo',default='bayes', help="pick the classification algorithm in \
         the following : "+str(implemented_algos))
    parser.add_argument('--cross_val',default=False, help="set to True (or 1) to perform a cross-validation", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--is_cov',default=False, help="set to True (or 1) if the graphs used are cov matrices", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--plot',default=False, help="set to True if not on the cluster to plot the confusion matrix", type=lambda x: (str(x).lower() in ['true','1']))

    args = parser.parse_args()
    #parser.print_help()

    graph_dir = args.graph_dir
    szr_types = args.seizure_types
    algo = args.algo
    cross_val = args.cross_val
    is_covariance = args.is_cov
    plot = args.plot

    if algo not in implemented_algos :
        print(f"The selected classification algorithm ('"+algo+"') is not available")
        exit()

    classify(graph_dir, szr_types, algo, cross_val, is_covariance, plot)

    print('\n\nDONE\n\n')
