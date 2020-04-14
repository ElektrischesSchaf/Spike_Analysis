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
import seaborn as sns
import gcmi
# My module
import sys
sys.path.append("../..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file


my_parameters=my_parameters.my_parameters()
mat_file_processing=load_mat_file.mat_file_processing()
session_name='indy_20160915_01'
file_name_1='../../../Dataset/Sorted_Spike_Dataset/'+session_name+'.mat'
file_list=[file_name_1]

band_start=0.5
band_cutoff=4

if band_start==0.5:
    bandwidth_token='0_5' +'-'+str(band_cutoff) +'Hz'
else:
    bandwidth_token=str(band_start)+'-'+str(band_cutoff)+'Hz'

file_itpc_abs_1='../../Signal_Processing/Inter-Channel_Clustering_Preprocess/Inter-Channel_Clustering_Output_Table/'+session_name+'/'+bandwidth_token+'/250Hz/ITPC_abs_250Hz.csv'
file_itpc_angle_1='../../Signal_Processing/Inter-Channel_Clustering_Preprocess/Inter-Channel_Clustering_Output_Table/'+session_name+'/'+bandwidth_token+'/250Hz/ITPC_angle_250Hz.csv'
file_itpc_time_stamp_1='../../Signal_Processing/Inter-Channel_Clustering_Preprocess/Inter-Channel_Clustering_Output_Table/'+session_name+'/'+bandwidth_token+'/250Hz/nwb_timestamp_to_mat_timestamp.csv'

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


# ITPC read file
iptc_abs=pd.read_csv(file_itpc_abs_1, dtype=float)
iptc_angle=pd.read_csv(file_itpc_angle_1, dtype=float)
itpc_time_stamp=pd.read_csv(file_itpc_time_stamp_1, dtype=float)

iptc_abs=np.array(iptc_abs)
iptc_angle=np.array(iptc_angle)
itpc_time_stamp=np.array(itpc_time_stamp)

itpc_64ms_rate=my_parameters.the_sampling_rate
iptc_abs=iptc_abs[::itpc_64ms_rate]
iptc_angle=iptc_angle[::itpc_64ms_rate]
itpc_time_stamp=itpc_time_stamp[::itpc_64ms_rate]

print(iptc_abs.shape, ' ', iptc_abs.shape, ' ', itpc_time_stamp.shape)

itpc_testing_data_index=int(int(len(itpc_time_stamp))*0.8) # split 80% into training
print('itpc_testing_data_index= ',itpc_testing_data_index) # 6142


iptc_abs_traing=iptc_abs[:itpc_testing_data_index,:] # TODO
iptc_abs_testing=iptc_abs[itpc_testing_data_index:,:]

iptc_angle_traing=iptc_angle[:itpc_testing_data_index,:]
iptc_angle_testing=iptc_angle[itpc_testing_data_index:,:]

# cross sessions control start
for session_index in range(file_numbers):
    print('In session '+ str(session_index+1) + ': ' + '\n' )

    [firing_rate_cell, channel_number, testing_data_index, time_stamp_64ms, x_position_label, y_position_label, z_position_label]=mat_file_processing.get_spike_bins_matrix(file_list[session_index], the_sampling_rate, time_stamp_64ms, include_hash_unit)
    [time_stamp_64ms, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]=mat_file_processing.get_labels(file_list[session_index], the_sampling_rate, time_stamp_64ms)

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
if 'LSTM' not in CWD:
    CWD = os.path.join(CWD, 'LSTM')
    if not os.path.exists(CWD):
        os.mkdir(CWD)

csv_path=os.path.join(CWD,'csv_files')
if not os.path.exists(csv_path):
    os.mkdir(str(csv_path))

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

training_x_spike=training_x.float()
training_y=training_y.float()

testing_x=pd.read_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), dtype=float)
testing_y=pd.read_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), dtype=float)

testing_x = torch.from_numpy(testing_x.values) # .values can turn pandas dataframe to numpy array
testing_y = torch.from_numpy(testing_y.values)

testing_x_spike=testing_x.float()
testing_y=testing_y.float()

# IPTC to torch
training_itpc_abs=torch.from_numpy(iptc_abs_traing)
training_itpc_angle=torch.from_numpy(iptc_angle_traing)


training_itpc_abs=training_itpc_abs.float()
training_itpc_angle=training_itpc_angle.float()

training_itpc=np.concatenate((training_itpc_angle, training_itpc_abs ), axis=1)

training_itpc=torch.from_numpy(training_itpc)

testing_itpc_abs=torch.from_numpy(iptc_abs_testing)
testing_itpc_angle=torch.from_numpy(iptc_angle_testing)

testing_itpc_abs=testing_itpc_abs.float()
testing_itpc_angle=testing_itpc_angle.float()



print(training_itpc_abs.size(), ' ',training_x.size() )
print(testing_itpc_abs.size(), ' ',testing_x.size() )

length_difference=abs(testing_x.size()[0]-testing_itpc_abs.size()[0])
print(length_difference)

if length_difference !=0:
    print(testing_itpc_abs.size(), ' ',testing_x[:,:].size() )

    testing_itpc_abs=testing_itpc_abs[:-length_difference,:]
    testing_itpc_angle=testing_itpc_angle[:-length_difference,:]

testing_itpc=np.concatenate((testing_itpc_angle, testing_itpc_abs), axis=1) 
testing_itpc=torch.from_numpy(testing_itpc)

print(testing_itpc_abs.size(), ' ', testing_y.size())


new_training_x=torch.cat(( training_x_spike,training_itpc ) , 1)
print('new_training_x= ', new_training_x.size())

new_testing_x=torch.cat(( testing_x_spike, testing_itpc), 1)




for k in range(new_training_x.size(0)):

    for i in range(96):
        pass
        # new_training_x[k,-2]=abs(new_training_x[k,-2])

        # Phase-of-Firing

        # absolute
        # new_training_x[k,i]=new_training_x[k,i]*abs(new_training_x[k,-2])*new_training_x[k,-1]
        # new_training_x[k,i]=new_training_x[k,i]*abs(new_training_x[k,-2])

        # new_training_x[k,i]=new_training_x[k,i]*new_training_x[k,-2]*new_training_x[k,-1]        

        # Concatenate
        # new_training_x[k,-2]=abs(new_training_x[k,-2])*new_training_x[k,-1]


for k in range(new_testing_x.size(0)):
    for i in range(96):
        pass
        # new_testing_x[k,-2]=abs(new_testing_x[k,-2])

        # Phase-of-Firing

        # absolute
        # new_testing_x[k,i]=new_testing_x[k,i]*abs(new_testing_x[k,-2])*new_testing_x[k,-1]
        # new_testing_x[k,i]=new_testing_x[k,i]*abs(new_testing_x[k,-2])

        # new_testing_x[k,i]=new_testing_x[k,i]*new_testing_x[k,-2]*new_testing_x[k,-1]        

        # Concatenate
        # new_testing_x[k,-2]=abs(new_testing_x[k,-2])*new_testing_x[k,-1]

new_training_x=new_training_x[:,:96]
new_testing_x=new_testing_x[:,:96]


training_itpc_abs=training_itpc_abs.numpy()
training_y=training_y.numpy()
training_itpc_angle=training_itpc_angle.numpy()
new_training_x=new_training_x.numpy()


print('shape of training_itpc_abs= ', training_itpc_abs.shape )

training_itpc_abs=np.transpose(training_itpc_abs)
training_y=np.transpose(training_y)
training_itpc_angle=np.transpose(training_itpc_angle)
new_training_x=np.transpose(new_training_x)
print('shape of new_training_x= ',   new_training_x.shape )

yee=gcmi.gcmi_cc( training_itpc_abs , training_y )
print('training_itpc_abs vs training_y = ', yee, '\n')

yee=gcmi.gcmi_cc(  training_itpc_angle , training_y )
print('training_itpc_angle vs training_y = ', yee, '\n')

yee=gcmi.gcmi_cc(  abs(training_itpc_angle) , training_y )
print('abs(training_itpc_angle) vs training_y = ', yee, '\n')

mi_of_all_channels=[]
yee=0
for i in range(96):
    # yee=yee+gcmi.gcmi_cc(  new_training_x[i,:] , training_y )
    yee=gcmi.gcmi_cc(  new_training_x[i,:] , training_y )
    mi_of_all_channels.append(yee)
# print('average new_training_x vs training_y = ', yee/96, '\n')
print(mi_of_all_channels)

yee=gcmi.gcmi_cc(  np.multiply( abs(training_itpc_angle) ,  training_itpc_abs), training_y )
print('abs(training_itpc_angle)*training_itpc_abs vs training_y = ', yee, '\n')