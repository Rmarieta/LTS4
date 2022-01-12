import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score

def load_graphs(input_dir, class_dict, is_cov) :

    data, data_labels = [], [] # data contains the graphs as tensors and data_labels the associated seizure type labels
    i = 0

    for szr_type in class_dict.keys() :

        szr_label = class_dict[szr_type]
        for _, _, files in os.walk(os.path.join(input_dir,szr_type)) :
            
            for npy_file in files :
                A = np.load(os.path.join(input_dir,szr_type,npy_file))
                # Normalise A (already normalised depending on the input)
                A = A/np.amax(A.flatten())
                
                if is_cov : 
                    L = A
                else :
                    L = np.diag(A*np.ones((A.shape[0],1)))-A

                L = L[np.triu_indices(20, k = 1)].flatten()
                # Change to tensor and reshape for dataloader
                # L = torch.tensor(L).view(1,190)

                data.append(L)
                data_labels.append(szr_label)

    #return np.array(data, dtype=object), np.array(data_labels)
    return np.array(data), np.array(data_labels)

def train_test_data(input_dir, class_dict, is_cov) :

    train, train_labels = load_graphs(os.path.join(input_dir,'train'), class_dict, is_cov)
    test, test_labels = load_graphs(os.path.join(input_dir,'dev'), class_dict, is_cov)

    return train, test, train_labels, test_labels

def to_set(train, test, train_labels, test_labels) :

    # Oversampling (train set only) to have balanced classification without dropping information
    PD = pd.DataFrame(train_labels, columns=['label'])
    no_0, no_1 = len(PD[PD['label']==0]), len(PD[PD['label']==1])
    R = math.floor(no_0/no_1) # Multiply the dataset by this ratio, then add (no_0 - R*no_1) randomly selected entries from the smallest dataset

    trainset, testset = [], []
    for i in range(len(train)) :
        if train_labels[i] == 1 : # Under-represented class :
            # The dataloader later shuffles the data
            for r in range(R) :
                trainset.append((train[i],train_labels[i]))
        else :
            trainset.append((train[i],train_labels[i]))
    
    # Compensate the remaining imbalance => draw (no_0 - R*no_1) elements from already present elements
    Add = random.sample(PD[PD['label']==1].index.to_list(),no_0 - R*no_1)
    for idx in Add :
        trainset.append((train[idx],train_labels[idx]))

    for j in range(len(test)) :
        testset.append((test[j],test_labels[j]))

    return trainset, testset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(190, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x,dim=1)

def train_model(CNN, trainloader, batch_size, optimizer, loss_criterion, gamma, nb_epochs, plot) :

    total_L = []
    print('Batch_size : ',batch_size,'\nLearning rate : ',gamma,'\n')
    for epoch in range(nb_epochs): 
        i = 0
        temp_L = []
        for data in trainloader:
            X, y = data
            optimizer.zero_grad()
            X, y = X.float(), y.type(torch.LongTensor)
            output = CNN(X)

            loss = loss_criterion(output, y)
            loss.backward()
            optimizer.step() 
            temp_L.append(loss)

            i += 1
        total_L.append(sum(temp_L)/float(len(temp_L)))
        print(f"Epoch : {epoch}, training loss : {round(total_L[-1].item(),6)}")

    if plot :
        loss_plot = np.array([T.detach().numpy() for T in total_L])
        plt.figure(figsize=(4.3,4))
        sns.set()
        plt.plot(loss_plot)
        plt.title('Evolution of the loss')
        plt.xlabel('Epoch');plt.ylabel('Loss');

    return round(total_L[-1].item(),6)

def compute_accuracy(testloader, CNN, last_loss, classes, plot) :

    correct = 0
    total = 0

    y_pred, y_true = [], []

    # Prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            X_test, labels = data
            X_test, labels = X_test.float(), labels.type(torch.LongTensor)
            # Calculate outputs by running images through the network
            outputs = CNN(X_test)

            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            _, predictions = torch.max(outputs, 1)
            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    TOT_ACC = 100 * correct / total
    F1 = 100 * f1_score(y_true, y_pred, average='weighted')

    print('\nFinal loss : ',last_loss)
    print(f'Unweighted total accuracy on test : {round(TOT_ACC,1)} %')
    print(f'Weighted F1-score on test : {round(F1,1)} %')

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for {:5s} is: {:.1f} %".format(classname, accuracy))
    
    C = confusion_matrix(y_true, y_pred)
    print(f'\nConfusion matrix :\n{C}\n')

    if plot :
        df_cm = pd.DataFrame(C, index=classes, columns=classes)
        plt.figure(figsize=(4.3,4))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', cbar=False) 
        plt.title('Confusion matrix')
        plt.ylabel('True label'); plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    print('\nStart...\n')

    parser = argparse.ArgumentParser(description='Build the graph CNN for classification')
    parser.add_argument('--input_dir', default='./data/v1.5.2/graph_avg_1_5', help='path to input graphs')
    parser.add_argument('--is_cov',default=False, help="set to True (or 1) if the graphs used are cov matrices", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--plot',default=False, help="set to True if not on the cluster to plot", type=lambda x: (str(x).lower() in ['true','1']))
    parser.add_argument('--nb_epochs',default=10, help="number of epochs for CNN training", type=lambda x: int(str(x)))
    parser.add_argument('--batch_size',default=50, help="batch size for the CNN dataloader", type=lambda x: int(str(x)))
    parser.add_argument('--l_rate',default=0.0001, help="learning rate for the CNN", type=lambda x: float(str(x)))
    parser.add_argument('--save_model',default=False, help="set to True to save the CNN", type=lambda x: (str(x).lower() in ['true','1']))

    args = parser.parse_args()
    input_dir = args.input_dir
    is_cov = args.is_cov
    plot = args.plot
    nb_epochs = args.nb_epochs
    batch_size = args.batch_size
    gamma = args.l_rate
    save_model = args.save_model

    classes = ['FNSZ','GNSZ']

    class_dict = {}
    for i, szr_type in enumerate(classes) :
        class_dict[szr_type] = i

    train, test, train_labels, test_labels = train_test_data(input_dir, class_dict, is_cov)
    # Turn into a set with the label to feed the dataloader and oversample the least represented class
    trainset, testset = to_set(train, test, train_labels, test_labels)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialise convolutional neural network
    FC_NN = Net()
    FC_NN = FC_NN.float()
    print(FC_NN)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(FC_NN.parameters(), lr=gamma)

    print('\nStart of training...\n')
    last_loss = train_model(FC_NN, trainloader, batch_size, optimizer, loss_criterion, gamma, nb_epochs, plot)
    print('\n...Training done\n\nComputation of accuracy on test data...')

    compute_accuracy(testloader, FC_NN, last_loss, classes, plot)

    if save_model : 
        torch.save(FC_NN, 'classifier/test_FC_NN.pt')

    print('\n...Done\n')