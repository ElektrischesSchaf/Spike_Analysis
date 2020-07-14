# https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
# Figures
import imageio
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
width_two=0.2
# Data Processing
import pandas as pd
import json
import math
import numpy as np
from numpy import linalg as LA
import h5py
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from sklearn import datasets, svm, metrics
# Regression Problem Evaluation Methods
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
# Read/Write file
import os
CWD_origin=os.getcwd()
import shutil

import time
tStart=time.time()

import seaborn as sns

# My module
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file
my_parameters=my_parameters.my_parameters()
mat_file_processing=load_mat_file.mat_file_processing()


session_name='indy_20160927_04'
file_name_1='../../Dataset/Sorted_Spike_Dataset/'+session_name+'.mat'
file_list=[file_name_1]

time_stamp_64ms=[]

# Auto-assigned parameters
testing_data_index=0
channel_number=0
units_have_value=0

# Parameters should be assigned
the_sampling_rate=my_parameters.the_sampling_rate
file_numbers=my_parameters.file_numbers
time_lag=my_parameters.time_lag

with_sorted_spikes=True
include_hash_unit=my_parameters.include_hash_unit

print('In session '+ session_name + ': ' + '\n' )

# Load Spike Firing Rate
[firing_rate_cell, channel_number, testing_data_index, time_stamp_64ms, unit_number]=mat_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate, time_stamp_64ms, include_hash_unit)

# Get channel and unit numbers
channel_numbers_in_this_dataset = channel_number
units_numbers_in_this_dataset = unit_number

if with_sorted_spikes==True:
    feature_numbers=channel_numbers_in_this_dataset*units_numbers_in_this_dataset
else:
    feature_numbers=channel_numbers_in_this_dataset

# Create empty arrrays from data
[X_for_training, X_for_prediction, 
x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing]=mat_file_processing.create_empty_traing_and_testing_label(feature_numbers)

[time_stamp_64ms, x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]=mat_file_processing.get_labels(file_name_1, the_sampling_rate, time_stamp_64ms)

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


# New Without spike sorting:
if with_sorted_spikes==False:
    with_sorting_firing_rate=firing_rate_matrix.copy()
    firing_rate_matrix=np.zeros([ channel_number, firing_rate_matrix.shape[1] ])
    print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
    print('with_sorting_firing_rate shape: ', with_sorting_firing_rate.shape) # (288, 12777)
    print('\n')

    for i in range(with_sorting_firing_rate.shape[1]):
        index=0
        k=0
        while index < channel_number:
            
            all_units_firing_rate_sum=0
            for unit_index in range( int(with_sorting_firing_rate.shape[0] / channel_number) ):
                all_units_firing_rate_sum+=with_sorting_firing_rate[k+unit_index][i]
            firing_rate_matrix[index][i]=all_units_firing_rate_sum

            # firing_rate_matrix[index][i]=with_sorting_firing_rate[k][i]+with_sorting_firing_rate[k+1][i]+with_sorting_firing_rate[k+2][i]

            index = index + 1
            k = k+ unit_number
    print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
    print('with_sorting_firing_rate shape: ', with_sorting_firing_rate.shape) # (288, 12777)
    print('\n')

firing_rate_matrix=np.transpose(firing_rate_matrix)        
feature_numbers_of_firing_rate = firing_rate_matrix.shape[1]

X=firing_rate_matrix.astype(np.float32)
print('features list shape: ',end='')
print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
print('\n')

# Cross Session Data Concatenation
[X_for_training, X_for_prediction,
x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing] = mat_file_processing.cross_session_data_concatenation(
feature_numbers_of_firing_rate, X, testing_data_index, X_for_training, X_for_prediction,
x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label, z_acceleration_label,
x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing)

print('shape of X_for_training before', X_for_training.shape)

# Processing max orders
# order_num=max_timestep-1
# [X_for_training, X_for_prediction,
# x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
# x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
# x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing] = mat_file_processing.max_order_preparation(
# order_num, feature_numbers, X_for_training, X_for_prediction,
# x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
# x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
# x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing)

print('shape of X_for_training after', X_for_training.shape)
print('shape of x_velocity_label_training after', x_velocity_label_training.shape)

print('\nshape of X_for_prediction after', X_for_prediction.shape)
print('shape of x_velocity_label_testing after', x_velocity_label_testing.shape)

# Write featrue and label to csv files
CWD = os.getcwd()

CWD = os.path.join(CWD, 'Firing_Rate_Visualization')
if not os.path.exists(CWD):
    os.mkdir(CWD)

csv_path=os.path.join(CWD,'csv_files')
if not os.path.exists(csv_path):
    os.mkdir(str(csv_path))

df = pd.DataFrame(X_for_training)
df.to_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), index=False)

df = pd.DataFrame(X_for_prediction)
df.to_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), index=False)


df=pd.DataFrame(x_position_label_training)
df.to_csv(os.path.join(csv_path,'x_position_label_training.csv'), index=False)

df=pd.DataFrame(x_position_label_testing)
df.to_csv(os.path.join(csv_path,'x_position_label_testing.csv'), index=False)

df=pd.DataFrame(y_position_label_training)
df.to_csv(os.path.join(csv_path,'y_position_label_training.csv'), index=False)

df=pd.DataFrame(y_position_label_testing)
df.to_csv(os.path.join(csv_path,'y_position_label_testing.csv'), index=False)


# read from csv file
training_x=pd.read_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), dtype=float)
training_x = torch.from_numpy(training_x.values) # .values can turn pandas dataframe to numpy array
training_x=training_x.float()

testing_x=pd.read_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), dtype=float)
testing_x = torch.from_numpy(testing_x.values) # .values can turn pandas dataframe to numpy array
testing_x=testing_x.float()


# x_pos
training_y_1 = pd.read_csv(os.path.join(csv_path,'x_position_label_training.csv'), dtype=float)    
training_y_1  = torch.from_numpy(training_y_1.values)    
training_y_1  = training_y_1.float()

testing_y_1 = pd.read_csv(os.path.join(csv_path,'x_position_label_testing.csv'), dtype=float)    
testing_y_1 = torch.from_numpy(testing_y_1.values)    
testing_y_1 = testing_y_1.float()

# y_pos
training_y_2 = pd.read_csv(os.path.join(csv_path,'y_position_label_training.csv'), dtype=float)    
training_y_2 = torch.from_numpy(training_y_2.values)    
training_y_2 = training_y_2.float()

testing_y_2 = pd.read_csv(os.path.join(csv_path,'y_position_label_testing.csv'), dtype=float)    
testing_y_2 = torch.from_numpy(testing_y_2.values)    
testing_y_2 = testing_y_2.float()

training_y = torch.cat( (training_y_1, training_y_2), 1)
testing_y = torch.cat( (testing_y_1, testing_y_2), 1)


# Start plotting
reduce_time_bin = 50
my_fontsize = 30
my_plot_width = 30
my_plot_height = 20

CWD = os.getcwd()

if 'Data_Visualization' not in CWD:
    CWD = os.path.join(CWD, 'Data_Visualization')
    if not os.path.exists(CWD):
        os.mkdir(CWD)

plot_path = os.path.join(CWD, session_name )
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


# Figure firing rate only
plt.figure(figsize=(my_plot_width, my_plot_height))

# plt.subplot(211)
plt.title(  session_name + ' Firing Rate', fontsize=30, color="black")
# plt.title('test')
sns.set(font_scale=3)
data = torch.transpose( testing_x[reduce_time_bin:reduce_time_bin*2,:], 0, 1)
# https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.colorbar.html
cbar_kws={"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.07, 'ticks' : [ torch.min(data), torch.max(data) ]}

ax = sns.heatmap( data=data, xticklabels=True, yticklabels=True, cbar_kws=cbar_kws, cmap='YlGnBu_r')
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = my_fontsize)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = my_fontsize)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

plt.xlabel('Time Bins', fontsize=my_fontsize, color="black")
plt.ylabel('Units', fontsize=my_fontsize, color="black")

plt.tight_layout()

plt.savefig(plot_path+'/' +'Firing Rate'+'.png')

plt.clf()
plt.cla()
plt.close()

# Figure firing rate and all kinematic variables




# plt.figure(figsize=(my_plot_width,  my_plot_height/4 ))
# plt.title(  session_name + ' x velocity', fontsize=30, color="black")
# plt.scatter( time_stamp_64ms[:reduce_time_bin] , torch.transpose(training_y[:reduce_time_bin,:],0,1), s=50, c='blue')
# plt.xlabel('Time (S)', fontsize=my_fontsize, color="black")
# plt.xlim([time_stamp_64ms[0] , time_stamp_64ms[reduce_time_bin] ])
# plt.ylabel('Velocity', fontsize=my_fontsize, color="black")
# plt.tight_layout()
# plt.savefig(plot_path+'/' +'Label_x-velocity.png')

# plt.clf()
# plt.cla()
# plt.close()