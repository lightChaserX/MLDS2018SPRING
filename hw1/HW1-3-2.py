
# coding: utf-8

# In[ ]:


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
num_epochs = 30
batch_size = 100
learning_rate = 0.1

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
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model

# Model 1 Deep


# number of para:  22,  36,  52,  68,  87, 107, 129, 152, 176, 203, 231, 260, 291, 324, 358, 393, 430, 469, 510
# choice:           0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18
hidden_layer1 = [ 250, 375, 500, 625, 750, 875,1000,1125,1250,1375,1500,1625,1750,1875,2000,2125,2250,2375,2500]
hidden_layer2 = [ 125, 187, 250, 312, 375, 437, 500, 562, 625, 687, 750, 812, 875, 937,1000,1062,1125,1187,1250]

choice = 17

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

# Model 2: Shallow
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''

net = Net().cuda()


# number of enumerate(train_loader)
#enum_list = list(enumerate(train_loader))
#print(len(enum_list))


    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss().cuda()  
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  
#print(len(train_loader))



# random labels
new_labels = torch.LongTensor(600, 100)

for i, (images, labels) in enumerate(train_loader):
    #print(train_loader)
    new_labels[i] = torch.LongTensor(100).random_(0, 10)

# Train the Model
for epoch in range(num_epochs):
    train_accu = 0.0
    train_loss = 0.0
    
    test_accu = 0.0
    test_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):          
        #images = Variable(images.type(torch.FloatTensor).cuda())
        images = Variable(images.cuda())
        
        #print(labels.size())
        #print(type(labels))
        #print(labels)
        
        #labels = new_labels[i]
        labels = Variable(new_labels[i].type(torch.LongTensor).cuda())
        
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
    
    #total = 0
    # Testing Data
    for images, labels in test_loader:
        #images = Variable(images.view(-1, 28*28))
        images = Variable(images.cuda())
        outputs = net(images)
        labels = Variable(labels.type(torch.LongTensor).cuda())
        _, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        loss = criterion(outputs, labels)
        
        # accuracy & loss
        test_accu += torch.sum(predicted == labels.data)
        test_loss += loss.data[0]
    tot = len(test_loader) * batch_size
    test_accu = test_accu/tot*100
    test_loss = test_loss/len(test_loader)
    
    print ('Epoch [%5d/%5d], Accu:%.4f TrainLoss: %.4f TestLoss: %.4f' % (epoch+1, num_epochs, train_accu, train_loss, test_loss))

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
torch.save(net.state_dict(), 'model.pkl')

