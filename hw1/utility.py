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

"""
TODO: Plot arrow
"""
def arrowplot(axes, x, y, narrs=30, dspace=0.5, direc='pos', \
                          hl=0.3, hw=6, c='black'): 
    ''' narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head 

        hw     :  width of the arrow head        

        c      :  color of the edge and face of the arrow head  
    '''

    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1] 
        dy = y[i]-y[i-1] 
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1.*abs(dspace) 
    else:
        dspace = abs(dspace)

    arrowData = [] # will hold tuples of x,y,theta for each arrow
    arrowPos = aspace*(dspace) # current point on walk along data
                                 # could set arrowPos to 0 if you want
                                 # an arrow at the beginning of the curve

    ndrawn = 0
    rcount = 1 
    while arrowPos < r.sum() and ndrawn < narrs:
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        da = arrowPos-rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da+x1
        ay = np.cos(theta)*da+y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos+=aspace
        while arrowPos > rtot[rcount+1]: 
            rcount+=1
            if arrowPos > rtot[-1]:
                break

    # could be done in above block if you want
    for ax,ay,theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*hl/2. + ax
        dy0 = np.cos(theta)*hl/2. + ay
        dx1 = -1.*np.sin(theta)*hl/2. + ax
        dy1 = -1.*np.cos(theta)*hl/2. + ay

        if direc is 'neg' :
          ax0 = dx0 
          ay0 = dy0
          ax1 = dx1
          ay1 = dy1 
        else:
          ax0 = dx1 
          ay0 = dy1
          ax1 = dx0
          ay1 = dy0 

        axes.annotate('', xy=(ax0, ay0), xycoords='data',
                xytext=(ax1, ay1), textcoords='data',
                arrowprops=dict( headwidth=hw, frac=1., ec=c, fc=c))


    axes.plot(x,y, color = c)
    #axes.set_xlim(x.min()*1.1,x.max()*1.1)
    #axes.set_ylim(y.min()*1.1,y.max()*1.1)