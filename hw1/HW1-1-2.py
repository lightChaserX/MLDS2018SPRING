import os, csv, argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from utility import *

# Hyper Parameters 
parser = argparse.ArgumentParser(description='MLDS HW1-1-2')
parser.add_argument('--batch-size', type=int, default=100, metavar='B',
                help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='I',
                help='number of epochs to training (default: 5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                help='learning rate (default: 0.001)')
parser.add_argument('--num-model', type=int, default=0, metavar='NM',
                help='model no. for training (default: 0)')
parser.add_argument('--times', type=int, default=0, metavar='T',
                help='training times (default: 0)')
args = parser.parse_args()
    
num_epochs     =       args.epochs
batch_size     =       args.batch_size
learning_rate  =       args.lr
num_model      =       args.num_model    
input_size     =       784
hidden_size    =       500
num_classes    =       10

print('num_epochs=%d,batch_size=%d,learning_rate=%f,num_model=%d,times=%d'
    % (num_epochs,   batch_size,   learning_rate,   num_model,   args.times))

# MNIST Dataset 
train_dataset, test_dataset = fetch_mnist_data(dir='data')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

exec('net = CNN_Net%s().cuda().double() ' % num_model)
    
folder_name = 'Results1-1-2'
if not os.path.exists(folder_name):
    os.makedirs(folder_name) 
accu_save_name   = 'mnist' + '_model' + str(num_model) + '_accu'   + str(args.times) + '.dat'
loss_save_name   = 'mnist' + '_model' + str(num_model) + '_loss'   + str(args.times) + '.dat'
weight_save_name = 'mnist' + '_model' + str(num_model) + '_weight' + str(args.times) + '.dat'
bias_save_name   = 'mnist' + '_model' + str(num_model) + '_bias'   + str(args.times) + '.dat'
model_save_name  = 'mnist' + '_model' + str(num_model) + '_bias'   + str(args.times) + '_model.pkl'
accu_save_file_name   = os.path.join(folder_name, accu_save_name)
loss_save_file_name   = os.path.join(folder_name, loss_save_name)
weight_save_file_name = os.path.join(folder_name, weight_save_name)
bias_save_file_name   = os.path.join(folder_name, bias_save_name)
model_save_file_name  = os.path.join(folder_name, model_save_name)

print("num of model params %d" % model_params(net))
    
criterion = nn.CrossEntropyLoss().cuda() 
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

##################################################################################################
loss_total = []
accu_total = []
all_weight_params = np.array([])
all_bias_params   = np.array([])
for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    
    test_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        #images = Variable(images.view(-1, 28*28))
        images = Variable(images.type(torch.DoubleTensor)).cuda()
        labels = Variable(labels.type(torch.LongTensor)).cuda()
        
        optimizer.zero_grad()
        outputs = net(images)
        _, pred = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, labels)
        loss.backward()
    
        running_correct += torch.sum(pred == labels.data)
        running_loss += loss.data[0]
        
        optimizer.step()
        
    cur_accu = running_correct/len(train_loader)
    cur_loss = running_loss/len(train_loader)
    
    loss_total.append(cur_loss)
    accu_total.append(cur_accu)
    
    # Display loss and save weights
    print ('Epoch [%d/%d], Accu:%.4f Loss: %.4f' % (epoch+1, num_epochs, cur_accu, cur_loss))
    weight_params = np.array([])
    bias_params   = np.array([])
    for one_name, one_param in net.named_parameters():
        if 'weight' in one_name:
            weight_params = np.append(weight_params, one_param.cpu().data.numpy())
        if 'bias' in one_name:
            bias_params = np.append(bias_params, one_param.cpu().data.numpy())
    if not all_weight_params.size or not all_bias_params.size:
        all_weight_params = weight_params
        all_bias_params   = bias_params
    else:
        all_weight_params = np.vstack((all_weight_params, weight_params))
        all_bias_params   = np.vstack((all_bias_params,   bias_params))
    weight_df = pd.DataFrame(np.transpose(all_weight_params))
    weight_df.to_csv(weight_save_file_name, index=False)
    bias_df = pd.DataFrame(np.transpose(all_bias_params))
    bias_df.to_csv(bias_save_file_name, index=False)
    loss_df = pd.DataFrame({"loss" : loss_total})
    loss_df.to_csv(loss_save_file_name, index=False)
    loss_df = pd.DataFrame({"accu" : accu_total})
    loss_df.to_csv(accu_save_file_name, index=False)
    
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    #images = Variable(images.view(-1, 28*28))
    images = Variable(images.type(torch.DoubleTensor)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu().numpy() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), model_save_file_name) #net.load_state_dict(torch.load('model.pkl'))