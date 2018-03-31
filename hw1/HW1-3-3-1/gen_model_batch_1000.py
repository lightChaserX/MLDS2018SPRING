
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse

import matplotlib.pyplot as plt
import numpy as np


# Hyper Parameters
num_epochs = 100
batch_size = 1024
learning_rate = 0.01

# MNIST Dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model

# Model 1 Deep


# number of para:  22,  52,  87, 129, 176, 231, 291, 358, 430, 510, 595, 687, 785, 889, 999,1116,1239,1368
# choice:           0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17
hidden_layer1 = [ 250, 500, 750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500]
hidden_layer2 = [ 125, 250, 375, 500, 625, 750, 875,1000,1125,1250,1375,1500,1625,1750,1875,2000,2125,2250]

choice = 7

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_layer1[choice])
        self.fc2 = nn.Linear(hidden_layer1[choice], hidden_layer2[choice])
        self.fc3 = nn.Linear(hidden_layer2[choice], 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().cuda()

    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss().cuda()  
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  


# Train the Model
for epoch in range(num_epochs):
    train_accu = 0.0
    train_loss = 0.0
    
    test_accu = 0.0
    test_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):          
        images = Variable(images.cuda())
        labels = Variable(labels.type(torch.LongTensor).cuda())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        _, pred = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # accuracy & loss
        train_accu += torch.sum(pred == labels.data)
        train_loss += loss.data[0]
        
        optimizer.step()
        
    tot = len(train_loader) * batch_size
    train_accu = train_accu/tot*100
    train_loss = train_loss/len(train_loader)
    
    # Testing Data
    for images, labels in test_loader:
        #images = Variable(images.view(-1, 28*28))
        images = Variable(images.cuda())
        outputs = net(images)
        labels = Variable(labels.type(torch.LongTensor).cuda())
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        # accuracy & loss
        test_accu += torch.sum(predicted == labels.data)
        test_loss += loss.data[0]
    
    tot = len(test_loader) * batch_size
    test_accu = test_accu/tot*100
    test_loss = test_loss/len(test_loader)
    
    print ('Epoch [%3d/%3d],  TrainAccu: %.4f  TestAccu: %.4f  TrainLoss: %.4f  TestLoss: %.4f' 
           % (epoch+1, num_epochs, train_accu, test_accu, train_loss, test_loss))

    if epoch != 0:
        plt.figure(1)
        l1, = plt.plot([epoch, epoch+1], [pre_train_accu, train_accu], 'k')
        l2, = plt.plot([epoch, epoch+1], [pre_test_accu, test_accu], 'b')
        plt.figure(2)
        l3, = plt.plot([epoch, epoch+1], [pre_train_loss, train_loss], 'k')
        l4, = plt.plot([epoch, epoch+1], [pre_test_loss, test_loss], 'b')

    pre_train_accu = train_accu
    pre_train_loss = train_loss
    pre_test_accu = test_accu
    pre_test_loss = test_loss
    
plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('accu')
plt.legend(handles=[l1,l2,], labels=['train_accu','test_accu',], loc='best')
plt.figure(2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(handles=[l3,l4,], labels=['train_loss','test_loss',], loc='best')
plt.show()

# Save the Model
torch.save(net.state_dict(), 'model_batch1000.pkl')

