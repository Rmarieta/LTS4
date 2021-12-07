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

def load_graphs(input_dir, class_dict) :

    data, data_labels = [], [] # data containing the graphs and data_labels the associated seizure type labels

    for szr_type in class_dict.keys() :
        szr_label = class_dict[szr_type]
        for _, _, files in os.walk(os.path.join(input_dir,szr_type)) :
            for npy_file in files :
                graph = np.load(os.path.join(input_dir,szr_type,npy_file))

                graph = graph[np.triu_indices(20, k = 1)]

                data.append(graph.flatten()) # graph has to be flattened to be fed to the classifier
                data_labels.append(szr_label)

    """
    for root, dir, files in os.walk(input_dir) :
        print('Root :',root)
        szr_type = root.split("/")[-1]
        print('Type : ',szr_type)
        if szr_type in class_dict.keys() : # Only consider the selected classes for the classification
            print('\nIN\n')
            szr_label = class_dict[szr_type]
            for npy_file in files :
                graph = np.load(os.path.join(input_dir,szr_type,npy_file))
                print('Graph :\n',graph.shape,'\n')
                data.append(graph.flatten()) # graph has to be flattened to be fed to the classifier
                data_labels.append(szr_label)
    """
    return np.array(data), np.array(data_labels)

def train_test_data(input_dir, class_dict) :

    train, train_labels = load_graphs(os.path.join(input_dir,'train'), class_dict)
    test, test_labels = load_graphs(os.path.join(input_dir,'dev'), class_dict)

    return train, test, train_labels, test_labels

def classify(input_dir, szr_types, algo) :

    class_dict = {}
    for i, szr_type in enumerate(szr_types) :
        class_dict[szr_type] = i
    
    train, test, train_labels, test_labels = train_test_data(input_dir, class_dict)

    # Shuffle the datasets (if of any use ?)
    np.random.seed(2) # For reproducibility
    train, train_labels = shuffle(train, train_labels)
    test, test_labels = shuffle(test, test_labels)

    if True :
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
        #disp = ConfusionMatrixDisplay(confusion_matrix=C)
        #disp.plot()

        df_cm = pd.DataFrame(C, index=szr_types, columns=szr_types)

        plt.figure(figsize=(5,4))
        #sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g') # font size
        plt.title(f'Confusion matrix ({algo}, train/test : {100*round(accuracy_score(train_labels, train_preds),2)}/{100*round(accuracy_score(test_labels, test_preds),2)} %)')
        plt.ylabel('True label'); plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
    
    else :
        
        L = list(range(0,21,5))
        """
        n, m = 7, 7
        fig, axes = plt.subplots(nrows=m,ncols=n)
        
        for i, graph in enumerate(train[:n*m]) :
            im = axes.flatten()[i].imshow(np.reshape(graph,(20,20)),cmap='Reds')
            axes.flatten()[i].set_title(szr_types[train_labels[i]]+str(i))
            axes.flatten()[i].set_xticks(L)
            axes.flatten()[i].set_yticks(L)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        """

        """
        n, m = 3, 1
        fig, axes = plt.subplots(nrows=m,ncols=n)
        idx = [21,34,38]#[6,32,48]
        for i, graph in enumerate(train[idx]) :
            im = axes.flatten()[i].imshow(np.reshape(graph,(20,20)),cmap='Reds')
            axes.flatten()[i].set_title(szr_types[train_labels[idx[i]]])
            axes.flatten()[i].set_xticks(L)
            axes.flatten()[i].set_yticks(L)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.293, 0.05, 0.4])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.show()
        """
    
if __name__ == '__main__':
    
    ################################################################

    # WATCH OUT, MIGHT NEED TO CONVERT LAPLACIAN TO ADJACENCY MATRIX
    # OR, COMPUTE THE ADJACENCY MATRIX BACK FROM THE LAPLACIAN

    # A = -(L - np.diag(np.diag(L)))
    # A = A/np.amax(A.flatten())

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
    args = parser.parse_args()
    #parser.print_help()

    graph_dir = args.graph_dir
    szr_types = args.seizure_types
    algo = args.algo

    if algo not in implemented_algos :
        print(f"The selected classification algorithm ('"+algo+"') is not available")
        exit()

    classify(graph_dir, szr_types, algo)

    print('\n\nDONE\n\n')
