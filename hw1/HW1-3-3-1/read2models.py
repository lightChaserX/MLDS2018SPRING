
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import numpy as np

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


# In[3]:


net1 = Net().cuda()
net2 = Net().cuda()
net1.load_state_dict(torch.load('model_batch64.pkl'))
net2.load_state_dict(torch.load('model_batch1000.pkl'))

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# I haven't finished it...
# We need to deal with parameters by para = para1 * p + para2 * (1-p)
# Then we get a new model, record the train,test accuracy and loss.


# In[ ]:


test_accu = 0.0
test_loss = 0.0

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

tot = 60000
test_accu = test_accu/tot*100
test_loss = test_loss/len(test_loader)

print(test_accu, test_loss)

