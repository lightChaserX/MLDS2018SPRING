# utility.py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as UtiData
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

"""
TODO: Define DNN model
"""
# DNN model0
class Net0(nn.Module):

    def __init__(self):
        super(Net0, self).__init__()
        self.fc1 = nn.Linear(1,  5)
        self.fc2 = nn.Linear(5,  10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 5)
        self.fc8 = nn.Linear(5,  1)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = F.relu(self.fc3(out2))
        out4 = F.relu(self.fc4(out3))
        out5 = F.relu(self.fc5(out4))
        out6 = F.relu(self.fc6(out5))
        out7 = F.relu(self.fc7(out6))
        pred_y = self.fc8(out7)
        return pred_y

# DNN model1
class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(1,  10)
        self.fc2 = nn.Linear(10, 18)
        self.fc3 = nn.Linear(18, 15)
        self.fc4 = nn.Linear(15, 4)
        self.fc5 = nn.Linear(4,  1)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = F.relu(self.fc3(out2))
        out4 = F.relu(self.fc4(out3))
        pred_y = self.fc5(out4)
        return pred_y
    
# DNN model2
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(1,  190)
        self.fc2 = nn.Linear(190,  1)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        pred_y = self.fc2(out1)
        return pred_y

"""
TODO: CNN Model
"""
# CNN model0
class CNN_Net0(nn.Module):
    def __init__(self):
        super(CNN_Net0, self).__init__()
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

# CNN model1
class CNN_Net1(nn.Module):
    def __init__(self):
        super(CNN_Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(432, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = x.view(-1, 432)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# CNN model2
class CNN_Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(288, 30)
        self.fc2 = nn.Linear(30, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# CNN model3
class CNN_Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 5, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(125, 63)
        self.fc2 = nn.Linear(63, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 125)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# CNN model4
class CNN_Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 15, kernel_size=3)
        self.conv3 = nn.Conv2d(15, 10, kernel_size=2)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(40, 80)
        self.fc2 = nn.Linear(80, 21)
        self.fc3 = nn.Linear(21, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
"""
TODO: Generate the function data
"""
def f0(x):
    return 4*x ** (5) - x ** (2) + 2*x ** (1/2) - 1

def f1(x):
    x = x + 0.05
    return (np.cos(5*np.pi*x))/(5*np.pi*x)

def f2(x):
    return np.sign((np.cos(5*np.pi*x)))

def make_feature(num_data, f):
    x_data = np.linspace(0,1,num_data)
    y_data = f(x_data)
    return UtiData.TensorDataset(data_tensor=torch.from_numpy(x_data).unsqueeze(1), 
                                 target_tensor=torch.from_numpy(y_data).unsqueeze(1))
    
"""
TODO: Generate the MNIST data
"""
def fetch_mnist_data(dir):
    train_dataset = dsets.MNIST(root=dir, 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

    test_dataset = dsets.MNIST(root=dir, 
                           train=False, 
                           transform=transforms.ToTensor())
    return train_dataset, test_dataset

"""
TODO: Calculate model parameters number
"""
def model_params(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params