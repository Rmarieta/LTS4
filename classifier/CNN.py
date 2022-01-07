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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score


def load_graphs(input_dir, class_dict, is_cov) :

    data, data_labels = [], [] # data containing the graphs and data_labels the associated seizure type labels
    i=0
    for szr_type in class_dict.keys() :
        szr_label = class_dict[szr_type]
        for _, _, files in os.walk(os.path.join(input_dir,szr_type)) :
            for npy_file in files :
                A = np.load(os.path.join(input_dir,szr_type,npy_file))

                # Normalise A (already normalised depending on the input)
                A = A/np.amax(A.flatten())
                if is_cov : 
                    L = torch.tensor(A).view(1,20,20)
                else : 
                    L = torch.tensor(np.diag(A*np.ones((A.shape[0],1)))-A).view(1,20,20)
                    #L = torch.tensor(A).view(1,20,20)

                data.append(L)
                data_labels.append(szr_label)

    return np.array(data), np.array(data_labels)

def train_test_data(input_dir, class_dict) :

    train, train_labels = load_graphs(os.path.join(input_dir,'train'), class_dict)
    test, test_labels = load_graphs(os.path.join(input_dir,'dev'), class_dict)

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

    for j in range(len(test)) :
        testset.append((test[j],test_labels[j]))

    return trainset, testset

class Net(nn.Module) :

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 3 * 3, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x,dim=1)

def train_model(CNN, trainloader, batch_size, optimizer, loss_criterion, gamma, nb_epochs) :

    total_L = []
    print('Batch_size : ',batch_size,'\nLearning rate : ',gamma)
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
        print(f"Epoch : {epoch}, Loss : {round(total_L[-1].item(),6)}")

    return round(total_L[-1].item(),6)

def compute_accuracy(testloader, CNN, last_loss) :

    correct = 0
    total = 0

    y_pred, y_true = [], []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            X_test, labels = data
            X_test, labels = X_test.float(), labels.type(torch.LongTensor)
            # calculate outputs by running images through the network
            outputs = CNN(X_test)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    TOT_ACC = 100 * correct / total
    F1 = 100 * f1_score(y_true, y_pred, average='weighted')

    # prepare to count predictions for each class
    classes = ('FNSZ','GNSZ')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            X_test, labels = data
            X_test, labels = X_test.float(), labels.type(torch.LongTensor)
            outputs = CNN(X_test)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print('Final loss : ',last_loss)
    print(f'Unweighted total accuracy on test : {round(TOT_ACC,1)} %')
    print(f'Weighted F1-score on test : {round(F1,1)} %')

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for {:5s} is: {:.1f} %".format(classname, accuracy))



if __name__ == '__main__':

    # Change this if using cov matrix (or to keep the adjacency matrix)
    is_cov = True

    # Need to put it as a torch.Size([1, 20, 20])
    #input_dir = '../data/v1.5.2/graph_unnormal'
    input_dir = '../data/v1.5.2/graph_cov_low'
    #input_dir = '../data/v1.5.2/graph_avg_1_5'
    szr_types = ['FNSZ','GNSZ']

    class_dict = {}
    for i, szr_type in enumerate(szr_types) :
        class_dict[szr_type] = i

    train, test, train_labels, test_labels = train_test_data(input_dir, class_dict)

    batch_size = 50

    classes = ('FNSZ','GNSZ')

    trainset, testset = to_set(train, test, train_labels, test_labels)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Run forward pass once and check flattened output dimensions
    CNN = Net()
    CNN = CNN.float()
    print(CNN)

    nb_epochs = 100
    gamma = 1e-4
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN.parameters(), lr=gamma)

    last_loss = train_model(CNN, trainloader, batch_size, optimizer, loss_criterion, gamma, nb_epochs)

    compute_accuracy(testloader, CNN, last_loss)