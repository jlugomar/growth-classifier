import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super(MLP, self).__init__()
        self.linears = [nn.Linear(in_features, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            self.linears.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.linears = nn.Sequential(*self.linears)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.linears(x)
        x = self.act(x)
        return x

class PatientDataset(Dataset):
    def __init__(self, X, y = None):
        self.X = X
        self.y = y 

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y != None:
            return self.X[idx, :], self.y[idx, 0]
        else:
            return self.X[idx, :]

class NeuralNet():
    def __init__(self, in_features, hidden_dims, learning_rate = 0.001, batch_size = 16, momentum = 0.9):
        self.model = MLP(in_features, hidden_dims).cuda()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss().cuda()

    def fit(self, train_X, train_y, num_epochs = 100):
        dset = PatientDataset(train_X, train_y)
        loader = DataLoader(dset, batch_size = self.batch_size, shuffle = True, num_workers = 16)
        for epoch in range(num_epochs):
            train_acc = 0.0
            train_loss = 0.0
            total_num = 0
            for x, y in loader:
                x = Variable(x, requires_grad = True).cuda()
                y = Variable(y).cuda()
                self.optimizer.zero_grad()
                self.model.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss)
                true_labels = y.cpu().numpy()
                prediction = output.cpu().detach().numpy()
                prediction = prediction.argmax(axis = 1)
                batch_acc = np.sum(true_labels == prediction)
                train_acc += batch_acc
                total_num += true_labels.shape[0]
            train_acc = train_acc / total_num
            yield train_acc, train_loss

    def score(self, test_X):
        dset = PatientDataset(test_X)
        loader = DataLoader(dset, batch_size = test_X.shape[0], shuffle = False, num_workers = 16)
        with torch.no_grad():
            for x in loader:
                x = Variable(x, requires_grad = False).cuda()
                output = self.model(x)
                output = output.cpu().detach().numpy()
                return output

    def predict(self, test_X):
        output = self.score(test_X)
        prediction = output.cpu().numpy()
        prediction = prediction.argmax(axis = 1)
        return prediction