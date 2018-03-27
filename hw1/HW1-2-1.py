import os, csv, argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from utility import *

"""
TODO: Execture
"""
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='MLDS HW1-2-1')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B',
                    help='input batch size for training (default: 100)')
    parser.add_argument('--iterations', type=int, default=2000, metavar='I',
                    help='number of iterations to training (default: 2000)')
    parser.add_argument('--num-data', type=int, default=2000, metavar='ND',
                    help='number of data to train (default: 2000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.005, metavar='M',
                    help='SGD momentum (default: 0.005)')
    parser.add_argument('--num-func', type=int, default=0, metavar='NF',
                    help='function no. for training (default: 0)')
    parser.add_argument('--num-model', type=int, default=0, metavar='NM',
                    help='model no. for training (default: 0)')
    parser.add_argument('--times', type=int, default=0, metavar='T',
                    help='training times (default: 0)')
    args = parser.parse_args()
    
    num_data       =       args.num_data 
    MAX_Iter       =       args.iterations
    BATCH_SIZE     =       args.batch_size
    train_lr       =       args.lr
    train_momentum =       args.momentum
    num_func       =       args.num_func
    num_model      =       args.num_model
    
    print('num_data=%d,MAX_Iter=%d,BATCH_SIZE=%d,train_lr=%f,train_momentum=%f,num_func=%d,num_model=%d,times=%d'
        % (num_data,   MAX_Iter,   BATCH_SIZE,   train_lr,   train_momentum,   num_func,   num_model,   args.times))
    
    exec('f = f%s' % num_func)
    exec('net = Net%s().cuda().double()' % num_model)
    
    folder_name = 'Results1-2-1'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 
    data_save_name   = 'func' + str(num_func) + '_model' + str(num_model) + '_data'   + str(args.times) + '.dat'
    loss_save_name   = 'func' + str(num_func) + '_model' + str(num_model) + '_loss'   + str(args.times) + '.dat'
    weight_save_name = 'func' + str(num_func) + '_model' + str(num_model) + '_weight' + str(args.times) + '.dat'
    bias_save_name   = 'func' + str(num_func) + '_model' + str(num_model) + '_bias'   + str(args.times) + '.dat'
    data_save_file_name   = os.path.join(folder_name, data_save_name)
    loss_save_file_name   = os.path.join(folder_name, loss_save_name)
    weight_save_file_name = os.path.join(folder_name, weight_save_name)
    bias_save_file_name   = os.path.join(folder_name, bias_save_name)
    
    loader = UtiData.DataLoader(dataset=make_feature(num_data, f), 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=1)
    
    criterion = nn.MSELoss().cuda()
    #optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=train_momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=train_lr, betas=(0.5, 0.999), weight_decay=0.000)
    
    ##################################################################################################
    loss_total = []
    all_weight_params = np.array([])
    all_bias_params   = np.array([])
    for iter in range(0,MAX_Iter):
        running_loss = 0.0
        optimizer.zero_grad()
        for step_i, (x_batch, y_batch) in enumerate(loader):
            if x_batch.size(0) != BATCH_SIZE or y_batch.size(0) != BATCH_SIZE:
                continue
            
            x_input = Variable(x_batch.type(torch.DoubleTensor).cuda())
            y_input = Variable(y_batch.type(torch.DoubleTensor).cuda())
            y_hat = net(x_input)
            
            loss = criterion(y_hat, y_input)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        
        # Display loss and save weights
        loss_total.append(running_loss/(num_data/BATCH_SIZE))
        print('Iteration %3d: loss %.10f' % (iter, running_loss/(num_data/BATCH_SIZE)))
        if not iter % (3*num_data//BATCH_SIZE):
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

    # Evaluate and plot results        
    x_eval = np.linspace(0,1,num_data) 
    y_eval = f(x_eval)
    yhat_test = net(Variable(torch.from_numpy(x_eval).unsqueeze(1).type(torch.DoubleTensor).cuda())).data.cpu().numpy().squeeze(1)
    yhat_eval = []
    for i in range(len(yhat_test)):
        yhat_eval.append(yhat_test[i])
    
    # Save evaluate results and loss
    data_df = pd.DataFrame({"x" : x_eval, "y" : y_eval, "yhat" : yhat_eval})
    loss_df = pd.DataFrame({"loss" : loss_total})
    data_df.to_csv(data_save_file_name, index=False)
    loss_df.to_csv(loss_save_file_name, index=False)
    
    print('Finish running the code!')
