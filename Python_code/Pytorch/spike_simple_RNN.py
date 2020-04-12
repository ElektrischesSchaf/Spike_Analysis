# https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plot
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt

import numpy as np
import imageio
import time
tStart=time.time()
import h5py
torch.manual_seed(1)    # reproducible
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm
from tqdm import trange
from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score

# My module
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file

my_parameters=my_parameters.my_parameters()
mat_file_processing=load_mat_file.mat_file_processing()

session_name='indy_20161007_02'

file_name_1='../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'
# file_name_2='../../Dataset/Sorted_Spike_Dataset/indy_20160411_01.mat'
# file_name_3='../../Dataset/Sorted_Spike_Dataset/indy_20160411_02.mat'
# file_name_4='../../Dataset/Sorted_Spike_Dataset/indy_20160418_01.mat'
# file_name_5='../../Dataset/Sorted_Spike_Dataset/indy_20160419_01.mat'
# file_name_6='../../Dataset/Sorted_Spike_Dataset/indy_20160420_01.mat'
# file_list=[file_name_1, file_name_2, file_name_3, file_name_4, file_name_5, file_name_6]
tStart=time.time()
time_stamp_64ms=[]
###################################### Auto-assigned parameters
#testing_data_index=5000
#testing_data_index=10222
testing_data_index=0 # Should be 10222 in indy_20160407_02
channel_number=0
units_have_value=0 # unit numbers that is not empty


###################################### Parameters should be assigned
the_sampling_rate=my_parameters.the_sampling_rate
file_numbers=my_parameters.file_numbers
time_lag=my_parameters.time_lag
order=my_parameters.order
with_sorted_spikes=my_parameters.with_sorted_spikes
include_hash_unit=my_parameters.include_hash_unit

# Must know these two numbers beforehand
channel_numbers_in_this_dataset=96
units_numbers_in_this_dataset=3

if with_sorted_spikes==True:
    feature_numbers=channel_numbers_in_this_dataset*units_numbers_in_this_dataset
else:
    feature_numbers=channel_numbers_in_this_dataset

X_for_training = np.empty([0, feature_numbers*(order+1)])
X_for_prediction = np.empty([0, feature_numbers*(order+1)])
X_for_prediction_with_time_lag = np.empty([0, feature_numbers*(order+1)])
X_for_prediction_with_time_lag_2 = np.empty([0, feature_numbers*(order+1)])

x_position_label_training= np.empty([0])
x_position_label_testing= np.empty([0])

y_position_label_training= np.empty([0])
y_position_label_testing= np.empty([0])

z_position_label_training= np.empty([0])
z_position_label_testing= np.empty([0])

x_velocity_label_training= np.empty([0])
x_velocity_label_testing= np.empty([0])

y_velocity_label_training= np.empty([0])
y_velocity_label_testing= np.empty([0])

z_velocity_label_training= np.empty([0])
z_velocity_label_testing= np.empty([0])

x_acceleration_label_training= np.empty([0])
x_acceleration_label_testing= np.empty([0])

y_acceleration_label_training= np.empty([0])
y_acceleration_label_testing= np.empty([0])

z_acceleration_label_training= np.empty([0])
z_acceleration_label_testing= np.empty([0])

# cross sessions control start
for session_index in range(file_numbers):
    print('In session '+ str(session_index+1) + ': ' + '\n' )

    [firing_rate_cell, channel_number, testing_data_index, time_stamp_64ms, x_position_label, y_position_label, z_position_label]=mat_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate, time_stamp_64ms, include_hash_unit)
    [time_stamp_64ms, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]=mat_file_processing.get_labels(file_name_1, the_sampling_rate, time_stamp_64ms)

    # Extract firing_rate_cell with rows have length bigger than zero
    firing_rate_final=[] # not[[]]
    for row_index in range( len( firing_rate_cell) ):   
        if len(firing_rate_cell[row_index]):
            firing_rate_final.append( firing_rate_cell[row_index] )
            units_have_value+=1

    '''
    for row_index in range( len( firing_rate_final) ):            
        print('length of firing_rate_final['+ str(row_index) +']: ',end='')
        print(len(firing_rate_final[row_index]))
    '''

    print('\n')

    firing_rate_matrix=np.array(firing_rate_final)
    print('firing_rate_matrix shape: ', firing_rate_matrix.shape) #  in indy_20160407_02 (226, 12777) eliminated null units, (288, 12777) with all 96X3 units
    print('\n')

    # Without spike sorting:
    if with_sorted_spikes==False:
        no_sorting_firing_rate=firing_rate_matrix.copy()
        firing_rate_matrix=np.zeros([ channel_number, firing_rate_matrix.shape[1] ])
        print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
        print('no_sorting_firing_rate shape: ', no_sorting_firing_rate.shape) # (288, 12777)
        print('\n')

        for i in range(no_sorting_firing_rate.shape[1]):
            index=0
            k=0
            while index < channel_number:
            #for k in range(channel_number-(units_numbers_in_this_dataset-1)): # Maximum 3 units in this session, indy_20160407_02.
                #print('index: ',index,end='')
                firing_rate_matrix[index][i]=no_sorting_firing_rate[k][i]+no_sorting_firing_rate[k+1][i]+no_sorting_firing_rate[k+2][i]

                # Test another way to exclude hash unit, but this only works in 96 features.
                #firing_rate_matrix[index][i]=no_sorting_firing_rate[k][i]+no_sorting_firing_rate[k+1][i]+no_sorting_firing_rate[k+2][i]

                #print('     firing_rate_matrix[index][i]: ',firing_rate_matrix[index][i] )
                index = index + 1
                k = k+ units_numbers_in_this_dataset
                #print('index: ', index, 'k: ', k)

        print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
        print('no_sorting_firing_rate shape: ', no_sorting_firing_rate.shape) # (288, 12777)
        print('\n')
    else:
        pass


    # Eliminate hash unit
    no_hash_unit_firing_rate=firing_rate_matrix.copy()

    # Making label data
    x_position_label=np.array(x_position_label)
    x_position_label=x_position_label.astype(np.float32)
    print('position x_position_label  list shape: ',end='') 
    print( x_position_label.shape ) # x is the label array should be feed into the model, (12777,)
    print('\n')

    y_position_label=np.array(y_position_label)
    y_position_label=y_position_label.astype(np.float32)
    print('position y_position_label list shape: ',end='')
    print( y_position_label.shape ) # y is the label array should be feed into the model, (12777,)
    print('\n')

    z_position_label=np.array(z_position_label)
    z_position_label=z_position_label.astype(np.float32)
    print('position z_position_label list shape: ',end='')
    print( z_position_label.shape ) # z is the label array should be feed into the model, (12777,)
    print('\n')

    firing_rate_matrix=np.transpose(firing_rate_matrix)
    print('transposed firing_rate_matrix shape: ', firing_rate_matrix.shape) # (12777, 288) in indy_20160407_02
    print('\n')
    feature_numbers= firing_rate_matrix.shape[1]

    X=firing_rate_matrix.astype(np.float32)
    print('features list shape: ',end='')
    print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
    print('\n')

    # Order Control Start
    order_index=order
    [X_for_training, X_for_prediction, X_for_prediction_with_time_lag, X_for_prediction_with_time_lag_2,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing] = mat_file_processing.order_and_timelag_processing(
    order_index, X, testing_data_index, time_lag, X_for_training, X_for_prediction, X_for_prediction_with_time_lag, X_for_prediction_with_time_lag_2,
    x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label, z_acceleration_label,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing)
    # Order Control End

# cross sessions control end


# Write featrue and label to csv files
CWD = os.getcwd()
model_name='RNN'
if model_name not in CWD:
    CWD = os.path.join(CWD, model_name)
    if not os.path.exists(CWD):
        os.mkdir(CWD)

if session_name not in CWD:
    CWD = os.path.join(CWD, session_name)
    if not os.path.exists(CWD):
        os.mkdir(CWD)

csv_path=os.path.join(CWD,'csv_files')
if not os.path.exists(csv_path):
    os.mkdir(str(csv_path))

plot_path = os.path.join(CWD, 'plots')
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


df = pd.DataFrame(X_for_training)
df.to_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), index=False)

df=pd.DataFrame(x_velocity_label_training)
df.to_csv(os.path.join(csv_path,'x_velocity_label_training.csv'), index=False)


df = pd.DataFrame(X_for_prediction)
df.to_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), index=False)

df=pd.DataFrame(x_velocity_label_testing)
df.to_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), index=False)

class AbstractDataset(Dataset):
    def __init__(self, feature_matrix, label_matrix):
        self.data=feature_matrix
        self.label=label_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('\nYee:', self.data[index], ', ' ,self.label[index])
        return self.data[index], self.label[index]
    
    def collate_fn(self, datas):
        # datas = [ batch_size X ( data + label ) ]
        print('\ncollate_fn！')
        print('\ndatas: ', datas)
        # return self.data, self.label


# read from csv file
training_x=pd.read_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), dtype=float)
training_y=pd.read_csv(os.path.join(csv_path,'x_velocity_label_training.csv'), dtype=float)

training_x = torch.from_numpy(training_x.values) # .values can turn pandas dataframe to numpy array
training_y = torch.from_numpy(training_y.values)

training_x=training_x.float()
training_y=training_y.float()

testing_x=pd.read_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), dtype=float)
testing_y=pd.read_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), dtype=float)

testing_x = torch.from_numpy(testing_x.values) # .values can turn pandas dataframe to numpy array
testing_y = torch.from_numpy(testing_y.values)

testing_x=testing_x.float()
testing_y=testing_y.float()

# Neural Network
batch_size = 32
learning_rate=1e-3

max_epoch=300

# LSTM
hidden_dim=100
layer_dim=2
output_dim=1

training_dataset=AbstractDataset(training_x,training_y)
testing_dataset=AbstractDataset(testing_x, testing_y)

# TODO collate_fn
# train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False, collate_fn=training_dataset.collate_fn)
# train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)
# test_loader=torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All models fit and predict, show R2 score

# this is one way to define a network
class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        x=x.unsqueeze(0)
        # print('\nshape of x: ', x.size(), ' ')

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0=h0.to(device)
        
        out, hn = self.rnn(x, h0)

        # print('\nshape of out1: ', out.size(), ' ')
        
        '''
        Index hidden state of last time step
        out.size() --> 100, 28, 100
        out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :])
        out.size() --> 100, 10
        '''
        
        out = self.fc(out)

        # print('\nshape of out2: ', out.size(), ' ')

        out=out.squeeze(0)

        # print('\nshape of out3: ', out.size(), ' ')


        return out

net = RNNModel(input_dim=training_x.shape[1], hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
net.to(device)
history = {'train':[],'valid':[]}


def _run_epoch(epoch, mode):
    net.train(True)
    if mode=='train':
        descrpition='Train'
        dataset=training_dataset
        schuffle=False
    else:
        descrpition='Valid'
        dataset=testing_dataset
        shuffle=False
    dataloader=torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False
                                            #collate_fn=dataset.collate_fn,
                                            )
    trange=tqdm(enumerate(dataloader), total=len(dataloader), desc=descrpition)
    loss=0

    my_prediction = []
    real_y_all=[]

    for i, (x, y) in trange:

        # LSTM batch
        # if(x.size()[0] is not batch_size):
        #     continue

        o_labels, batch_loss = _run_iter(x,y)

        if mode=='train':
            optimizer.zero_grad()   # clear gradients for next train
            batch_loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        loss+=batch_loss.item()

        real_y=y.cpu().data.numpy()
        for ele in o_labels.cpu().data.numpy():
            my_prediction.append(ele)

        for ele in real_y:
            real_y_all.append(ele)
        
        R_square=r2_score( real_y_all, my_prediction)

        trange.set_postfix(loss=loss/(i+1), R_square=R_square)

    if mode=='train':
        history['train'].append({'loss':loss/len(trange), 'R^2': R_square })
        # writer.add_scalar('Loss/train', loss/len(trange), epoch)
    else:
        history['valid'].append({'loss':loss/len(trange), 'R^2': R_square })
        # writer.add_scalar('Loss/valid', loss/len(trange), epoch)
    trange.close()

def _run_iter(x,y):
    feature = x.to(device)
    labels = y.to(device)
    #print('\n\n In _run_iter, ', 'shape of x', x.shape, ' ', 'shape of y', y.shape)
    o_labels = net(feature)
    #print('The output shape: ', o_labels.shape, ' The label shape: ', labels.shape, '\n')
    l_loss = loss_func(o_labels, labels)
    return o_labels, l_loss

def save(epoch):
    if not os.path.exists(os.path.join(CWD,'save')):
        os.makedirs(os.path.join(CWD,'save'))
    torch.save(net.state_dict(), os.path.join( CWD,'save/model.pkl.'+str(epoch) ))
    with open( os.path.join( CWD,'save/history.json'), 'w') as f:
        json.dump(history, f, indent=4)

for epoch in range(max_epoch):
    print('Epoch: {}'.format(epoch))
    _run_epoch(epoch, 'train')
    _run_epoch(epoch, 'valid')
    save(epoch)

# Plot the training results 
with open(os.path.join(CWD,'save/history.json'), 'r') as f:
    history = json.loads(f.read())
    
train_loss = [l['loss'] for l in history['train']]
valid_loss = [l['loss'] for l in history['valid']]

train_R_square = [l['R^2'] for l in history['train']]
valid_R_square = [l['R^2'] for l in history['valid']]


plt.figure(figsize=(7,5))
plt.title(model_name+' Loss')
plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='valid')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig(plot_path + '/' + model_name+"_Loss.png")

plt.figure(figsize=(7,5))
plt.title(model_name+' performance')
plt.plot(train_R_square, label='train')
plt.plot(valid_R_square, label='valid')
plt.xlabel('Epoch')
plt.ylabel('R square')
plt.legend()
plt.tight_layout()
plt.savefig(plot_path + '/' +model_name+"_R-square.png")

#global my_prediction
#global real_y_all


best_score, best_epoch=max([[l['R^2'], idx] for idx, l in enumerate(history['valid'])])
print('best_score= ', best_score, ', best_epoch= ', best_epoch, '\n')
print('Best R-square score ', max([[l['R^2'], idx] for idx, l in enumerate(history['valid'])]))


# Testing
best_model=best_epoch # TODO
net.load_state_dict(state_dict=torch.load(os.path.join(CWD,'save/model.pkl.{}'.format(best_model))))
net.train(False)
# start testing
dataloader = DataLoader(dataset=testing_dataset,
                            batch_size=batch_size,
                            shuffle=False
                            #collate_fn=testData.collate_fn,
                            #num_workers=8
                            )
trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
my_prediction = []
real_y_all=[]

for i, (x, testing_y) in trange:

    # LSTM batch
    # if(x.size()[0] is not batch_size):
    #     continue

    o_labels = net(x.to(device))
    real_y=testing_y.cpu().data.numpy()
    for ele in o_labels.cpu().data.numpy():
        my_prediction.append(ele)

    for ele in real_y:
        real_y_all.append(ele)

print('\n* model_x_velocity score in order ', order_index, ': ', r2_score( real_y_all, my_prediction))
testing_data_r_square=r2_score( real_y_all, my_prediction)


x_velocity_predict=my_prediction
plot.figure(figsize=(30,10))
plot.plot(time_stamp_64ms[testing_data_index:-1], x_velocity_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], x_velocity_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right', fontsize=20)
plot.title(model_name+' Model: velocity x prediction and ground truth (Spike Only), R sqaure= '+str( round( testing_data_r_square, 4) ),fontsize=30)
plot.xlabel('Time (second)', fontsize=25)
plot.ylabel('x velocity', fontsize=25)
plt.xticks(fontsize=20, color="black")
plt.yticks(fontsize=20, color="black")
axes = plot.gca()
axes.set_xlim([725, 745])
plot.tight_layout()
plot.savefig( plot_path + '/' +model_name+'_x-velocity_predict.png' )

plot.cla()
plot.clf()

tEnd=time.time()
print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )