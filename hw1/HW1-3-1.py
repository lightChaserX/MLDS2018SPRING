
# coding: utf-8

# In[3]:


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
num_epochs = 4000
batch_size = 100
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
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model

# Model 1 Deep

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    
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


# In[ ]:


# Train the Model
for epoch in range(num_epochs):
    train_accu = 0.0
    train_loss = 0.0
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
        
        #loss & accuracy
        train_accu += torch.sum(pred == labels.data)
        train_loss += loss.data[0]
        
        optimizer.step()
    
    train_accu = train_accu/len(train_loader)
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
        test_loss += loss.data[0]
    test_loss = test_loss/len(test_loader)
    
    print ('Epoch [%5d/%5d], Accu:%.4f TrainLoss: %.4f TestLoss: %.4f' % (epoch+1, num_epochs, train_accu, train_loss, test_loss))

    if epoch != 0:
        plt.figure(1)
        plt.plot([epoch, epoch+1], [pre_train_accu, train_accu], 'k')
        plt.figure(2)
        l1, = plt.plot([epoch, epoch+1], [pre_train_loss, train_loss], 'k')
        l2, = plt.plot([epoch, epoch+1], [pre_test_loss, test_loss], 'b')

    pre_train_accu = train_accu
    pre_train_loss = train_loss
    pre_test_loss = test_loss
    
plt.figure(1)
plt.ylabel('train accu')
plt.figure(2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(handles=[l1,l2,], labels=['aaa','bbb',], loc='best')
plt.show()

# Save the Model
torch.save(net.state_dict(), 'model.pkl')

