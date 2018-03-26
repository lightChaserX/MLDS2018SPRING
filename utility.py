# utility.py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as UtiData
import torch.optim as optim
import numpy as np

"""
TODO: Define model0
"""
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

"""
TODO: Define model1
"""
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
    
"""
TODO: Define model2
"""
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
TODO: Generate the data
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