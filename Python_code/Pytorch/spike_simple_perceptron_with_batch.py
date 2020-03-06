# https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt

import numpy as np
import imageio
import time
import h5py
torch.manual_seed(1)    # reproducible

from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score

# My module
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file

my_parameters=my_parameters.my_parameters()
mat_file_processing=load_mat_file.mat_file_processing()

file_name_1='../../Dataset/Sorted_Spike_Dataset/indy_20160407_02.mat'
file_name_2='../../Dataset/Sorted_Spike_Dataset/indy_20160411_01.mat'
file_name_3='../../Dataset/Sorted_Spike_Dataset/indy_20160411_02.mat'
file_name_4='../../Dataset/Sorted_Spike_Dataset/indy_20160418_01.mat'
file_name_5='../../Dataset/Sorted_Spike_Dataset/indy_20160419_01.mat'
file_name_6='../../Dataset/Sorted_Spike_Dataset/indy_20160420_01.mat'
file_list=[file_name_1, file_name_2, file_name_3, file_name_4, file_name_5, file_name_6]
tStart=time.time()

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

    [firing_rate_cell, channel_number, testing_data_index, x_position_label, y_position_label, z_position_label]=mat_file_processing.get_spike_bins_matrix(file_list[session_index], the_sampling_rate, include_hash_unit)
    [x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]=mat_file_processing.get_labels(file_list[session_index], the_sampling_rate)

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
    if order_index >=2:
        order_original_matrix=X[:-order_index, :]
        for order_loop_index in range(1, order_index):
            temp_order_matrix=X[order_loop_index: -(order_index-order_loop_index), :]
            order_original_matrix=np.concatenate((order_original_matrix, temp_order_matrix), axis=1)
        final_order_matrix=X[order_index:, :]
        order_original_matrix=np.concatenate((order_original_matrix, final_order_matrix), axis=1)
        print('\n')

        XX=order_original_matrix.copy()

        X_for_training = np.concatenate(( X_for_training, XX[:testing_data_index, :] ), axis=0 )
        X_for_prediction = np.concatenate(( X_for_prediction , XX[testing_data_index:] ), axis=0)
        X_for_prediction_with_time_lag = np.concatenate((X_for_prediction_with_time_lag , XX[testing_data_index:-time_lag]), axis=0)
        X_for_prediction_with_time_lag_2 = np.concatenate((X_for_prediction_with_time_lag_2, XX[testing_data_index:-1-time_lag]), axis=0)

        x_position_label_training = np.concatenate((x_position_label_training, x_position_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        x_position_label_testing = np.concatenate((x_position_label_testing, x_position_label[testing_data_index+order_index+time_lag:]), axis=0)

        y_position_label_training =  np.concatenate((y_position_label_training, y_position_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        y_position_label_testing = np.concatenate((y_position_label_testing,  y_position_label[testing_data_index+order_index+time_lag:]), axis=0)

        z_position_label_training = np.concatenate((z_position_label_training, z_position_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        z_position_label_testing = np.concatenate((z_position_label_testing, z_position_label[testing_data_index+order_index+time_lag:]), axis=0)
    
        x_velocity_label_training = np.concatenate((x_velocity_label_training, x_velocity_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        x_velocity_label_testing = np.concatenate((x_velocity_label_testing, x_velocity_label[testing_data_index+order_index+time_lag:]), axis=0)
        
        y_velocity_label_training = np.concatenate((y_velocity_label_training, y_velocity_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        y_velocity_label_testing = np.concatenate((y_velocity_label_testing, y_velocity_label[testing_data_index+order_index+time_lag:]), axis=0)

        z_velocity_label_training = np.concatenate((z_velocity_label_training, z_velocity_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        z_velocity_label_testing = np.concatenate((z_velocity_label_testing, z_velocity_label[testing_data_index+order_index+time_lag:]), axis=0)

        x_acceleration_label_training = np.concatenate((x_acceleration_label_training, x_acceleration_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        x_acceleration_label_testing = np.concatenate((x_acceleration_label_testing, x_acceleration_label[testing_data_index+order_index+time_lag:]), axis=0)

        y_acceleration_label_training = np.concatenate((y_acceleration_label_training, y_acceleration_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        y_acceleration_label_testing = np.concatenate((y_acceleration_label_testing, y_acceleration_label[testing_data_index+order_index+time_lag:]), axis=0)

        z_acceleration_label_training = np.concatenate((z_acceleration_label_training, z_acceleration_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        z_acceleration_label_testing = np.concatenate((z_acceleration_label_testing, z_acceleration_label[testing_data_index+order_index+time_lag:]), axis=0)

    if order_index==1:
        temp1=X[order_index:,:]
        temp2=X[:-order_index,:]
        #print('temp1 and temp2 shape: ', temp1.shape, temp2.shape)
        X_order_1=np.concatenate((temp1, temp2), axis=1)  # X_order_1 should be deprecated
        #print('order 1 fetures list shape: ', X_order_1.shape) #  X_order_1 is the feature matrix, (12776, 576) in indy_20160407_02
        print('\n')
        XX=np.concatenate((temp1, temp2), axis=1)

        X_for_training = np.concatenate(( X_for_training, XX[:testing_data_index, :] ), axis=0 )
        X_for_prediction = np.concatenate(( X_for_prediction , XX[testing_data_index:] ), axis=0)
        X_for_prediction_with_time_lag = np.concatenate((X_for_prediction_with_time_lag , XX[testing_data_index:-time_lag]), axis=0)
        X_for_prediction_with_time_lag_2 = np.concatenate((X_for_prediction_with_time_lag_2, XX[testing_data_index:-1-time_lag]), axis=0)

        x_position_label_training = np.concatenate((x_position_label_training, x_position_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        x_position_label_testing = np.concatenate((x_position_label_testing, x_position_label[testing_data_index+order_index+time_lag:]), axis=0)

        y_position_label_training =  np.concatenate((y_position_label_training, y_position_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        y_position_label_testing = np.concatenate(( y_position_label_testing, y_position_label[testing_data_index+order_index+time_lag:]), axis=0)

        z_position_label_training = np.concatenate((z_position_label_training, z_position_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        z_position_label_testing = np.concatenate((z_position_label_testing, z_position_label[testing_data_index+order_index+time_lag:]), axis=0)
    
        x_velocity_label_training = np.concatenate((x_velocity_label_training, x_velocity_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        x_velocity_label_testing = np.concatenate((x_velocity_label_testing, x_velocity_label[testing_data_index+order_index+time_lag:]), axis=0)
        
        y_velocity_label_training = np.concatenate((y_velocity_label_training, y_velocity_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        y_velocity_label_testing = np.concatenate((y_velocity_label_testing, y_velocity_label[testing_data_index+order_index+time_lag:]), axis=0)

        z_velocity_label_training = np.concatenate((z_velocity_label_training, z_velocity_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        z_velocity_label_testing = np.concatenate((z_velocity_label_testing, z_velocity_label[testing_data_index+order_index+time_lag:]), axis=0)

        x_acceleration_label_training = np.concatenate((x_acceleration_label_training, x_acceleration_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        x_acceleration_label_testing = np.concatenate((x_acceleration_label_testing, x_acceleration_label[testing_data_index+order_index+time_lag:]), axis=0)

        y_acceleration_label_training = np.concatenate((y_acceleration_label_training, y_acceleration_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        y_acceleration_label_testing = np.concatenate((y_acceleration_label_testing, y_acceleration_label[testing_data_index+order_index+time_lag:]), axis=0)

        z_acceleration_label_training = np.concatenate((z_acceleration_label_training, z_acceleration_label[order_index+time_lag:testing_data_index+order_index+time_lag]), axis=0)
        z_acceleration_label_testing = np.concatenate((z_acceleration_label_testing, z_acceleration_label[testing_data_index+order_index+time_lag:]), axis=0)

    if order_index==0:

        X_for_training = np.concatenate(( X_for_training, X[:testing_data_index, :] ), axis=0 )
        X_for_prediction = np.concatenate(( X_for_prediction , X[testing_data_index:] ), axis=0)
        X_for_prediction_with_time_lag = np.concatenate((X_for_prediction_with_time_lag , X[testing_data_index:-time_lag]), axis=0)
        X_for_prediction_with_time_lag_2 = np.concatenate((X_for_prediction_with_time_lag_2, X[testing_data_index:-1-time_lag]), axis=0)

        x_position_label_training = np.concatenate((x_position_label_training, x_position_label[time_lag:testing_data_index+time_lag]), axis=0)
        x_position_label_testing = np.concatenate((x_position_label_testing, x_position_label[testing_data_index+time_lag:]), axis=0)

        y_position_label_training =  np.concatenate((y_position_label_training, y_position_label[time_lag:testing_data_index+time_lag ]), axis=0)
        y_position_label_testing = np.concatenate((y_position_label_testing, y_position_label[testing_data_index+time_lag:]), axis=0)

        z_position_label_training = np.concatenate((z_position_label_training, z_position_label[time_lag:testing_data_index+time_lag ]), axis=0)
        z_position_label_testing = np.concatenate((z_position_label_testing, z_position_label[testing_data_index+time_lag:]), axis=0)
    
        x_velocity_label_training = np.concatenate((x_velocity_label_training, x_velocity_label[time_lag:testing_data_index+time_lag ]), axis=0)
        x_velocity_label_testing = np.concatenate((x_velocity_label_testing, x_velocity_label[testing_data_index+time_lag:]), axis=0)
        
        y_velocity_label_training = np.concatenate((y_velocity_label_training, y_velocity_label[time_lag:testing_data_index+time_lag ] ), axis=0)
        y_velocity_label_testing = np.concatenate((y_velocity_label_testing, y_velocity_label[testing_data_index+time_lag:]), axis=0)

        z_velocity_label_training = np.concatenate((z_velocity_label_training, z_velocity_label[time_lag:testing_data_index+time_lag ]), axis=0)
        z_velocity_label_testing = np.concatenate((z_velocity_label_testing, z_velocity_label[testing_data_index+time_lag:]), axis=0)

        x_acceleration_label_training = np.concatenate((x_acceleration_label_training, x_acceleration_label[time_lag:testing_data_index+time_lag ]), axis=0)
        x_acceleration_label_testing = np.concatenate((x_acceleration_label_testing, x_acceleration_label[testing_data_index+time_lag:]), axis=0)

        y_acceleration_label_training = np.concatenate((y_acceleration_label_training, y_acceleration_label[time_lag:testing_data_index+time_lag ]), axis=0)
        y_acceleration_label_testing = np.concatenate((y_acceleration_label_testing, y_acceleration_label[testing_data_index+time_lag:]), axis=0)

        z_acceleration_label_training = np.concatenate((z_acceleration_label_training, z_acceleration_label[time_lag:testing_data_index+time_lag ]), axis=0)
        z_acceleration_label_testing = np.concatenate((z_acceleration_label_testing, z_acceleration_label[testing_data_index+time_lag:]), axis=0)
    # Order Control End

# cross sessions control end


# Write featrue and label to csv files
CWD = os.getcwd()
if 'order_0' not in CWD:
    CWD = os.path.join(CWD, 'order_0')
    if not os.path.exists(CWD):
        os.mkdir(CWD)

csv_path=os.path.join(CWD,'csv_files')
if not os.path.exists(csv_path):
    os.mkdir(str(csv_path))

df = pd.DataFrame(X_for_training)
df.to_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), index=False)

df=pd.DataFrame(x_position_label_training)
df.to_csv(os.path.join(csv_path,'x_position_label_training.csv'), index=False)


df = pd.DataFrame(X_for_prediction)
df.to_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), index=False)

df=pd.DataFrame(x_position_label_testing)
df.to_csv(os.path.join(csv_path,'x_position_label_testing.csv'), index=False)

class AbstractDataset(Dataset):
    def __init__(self, feature_matrix, label_matrix):
        self.data=feature_matrix
        self.label=label_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def collate_fn(self, datas):
        # datas = [ batch_size X ( data + label ) ]
        print('\ncollate_fn！')
        print('\ndatas: ', datas)


# read from csv file
x=pd.read_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), dtype=float)
y=pd.read_csv(os.path.join(csv_path,'x_position_label_training.csv'), dtype=float)

x = torch.from_numpy(x.values) # .values can turn pandas dataframe to numpy array
y = torch.from_numpy(y.values)

x=x.float()
y=y.float()

testing_x=pd.read_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), dtype=float)
testing_y=pd.read_csv(os.path.join(csv_path,'x_position_label_testing.csv'), dtype=float)

testing_x = torch.from_numpy(testing_x.values) # .values can turn pandas dataframe to numpy array
testing_y = torch.from_numpy(testing_y.values)

testing_x=testing_x.float()
testing_y=testing_y.float()

batch_size = 16
n_iters = 3000
num_epochs = n_iters / ( (x.shape[0]) / batch_size )
num_epochs = int(num_epochs)

training_dataset=AbstractDataset(x,y)
testing_dataset=AbstractDataset(testing_x, testing_y)

# TODO collate_fn
# train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False, collate_fn=training_dataset.collate_fn)
train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)
test_loader=torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All models fit and predict, show R2 score

# this is one way to define a network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=x.shape[1], n_hidden=50, n_output=1)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

net.to(device)



# train the network
iter = 0
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        prediction = net( x.to(device) ) # do not flatten     # input x and predict based on x
        #print('size of prediction= ', prediction.shape, ' size of y= ',y.shape,'\n')

        loss = loss_func(prediction, y.to(device))     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        iter += 1

        if iter % 100 == 0:

            my_prediction=[]

            for i, (testing_x, testing_y) in enumerate(test_loader):
                prediction=net( testing_x.to(device) ).flatten()
                # print('prediction=', prediction.cpu().data.numpy(),'\n')
                for ele in prediction.cpu().data.numpy():
                    my_prediction.append( ele )
                # print('len of my_prediction=', len(my_prediction), '\n')


            # print('len of my_prediction2=', len(my_prediction), '\n')
    # my_prediction=torch.cat(my_prediction).detach().numpy().astype(float)


# print('shape of x_position_label_training = ', x_position_label_training.shape, '\n shape of prediction = ', prediction.shape, '\n')
# print('\n* model_x_position score in order ', order_index, ': ', r2_score( x_position_label_training.flatten(), prediction.cpu().data.numpy()))


            # predict from testing feature matrix

            print('shape of x_position_label_testing = ', x_position_label_testing.shape, '\n len of my_prediction = ', len(my_prediction), '\n')
            print('\n* model_x_position score in order ', order_index, ': ', r2_score( x_position_label_testing.flatten(), my_prediction))