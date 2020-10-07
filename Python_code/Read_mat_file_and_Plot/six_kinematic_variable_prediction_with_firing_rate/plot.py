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
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

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
sys.path.append("../..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file
my_parameters=my_parameters.my_parameters()
mat_file_processing=load_mat_file.mat_file_processing()


session_name='indy_20160420_01'
file_name_1='../../../Dataset/Sorted_Spike_Dataset/'+session_name+'.mat'
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
[firing_rate_cell, channel_number, testing_data_index, time_stamp_64ms, unit_number] = mat_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate, time_stamp_64ms, include_hash_unit)

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
x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,
x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing ]=mat_file_processing.create_empty_traing_and_testing_label(feature_numbers)

[time_stamp_64ms, 
x_position_label, y_position_label, z_position_label, 
x_velocity_label, y_velocity_label, z_velocity_label, 
x_acceleration_label, y_acceleration_label, z_acceleration_label,
x_position_target, y_position_target] = mat_file_processing.get_labels(file_name_1, the_sampling_rate, time_stamp_64ms)

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
x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, 
x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, 
x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing,
x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing] = mat_file_processing.cross_session_data_concatenation(
session_name, feature_numbers_of_firing_rate, X, testing_data_index, X_for_training, X_for_prediction,
x_position_label, y_position_label,  x_velocity_label, y_velocity_label,  x_acceleration_label, y_acceleration_label,  x_position_target, y_position_target,
x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, 
x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, 
x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing,
x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing)
print('shape of X_for_training before', X_for_training.shape)


print('shape of X_for_training after', X_for_training.shape)
print('shape of x_velocity_label_training after', x_velocity_label_training.shape)

print('\nshape of X_for_prediction after', X_for_prediction.shape)
print('shape of x_velocity_label_testing after', x_velocity_label_testing.shape)

# Write featrue and label to csv files
CWD = os.getcwd()

Firing_Rate_Visualization = os.path.join(CWD, 'Firing_Rate_Visualization')
if not os.path.exists(Firing_Rate_Visualization):
    os.mkdir(Firing_Rate_Visualization)

csv_path=os.path.join(Firing_Rate_Visualization,'csv_files')
if not os.path.exists(csv_path):
    os.mkdir(str(csv_path))

df = pd.DataFrame(X_for_training)
df.to_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), index=False)

df = pd.DataFrame(X_for_prediction)
df.to_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), index=False)

# positino
df=pd.DataFrame(x_position_label_training)
df.to_csv(os.path.join(csv_path,'x_position_label_training.csv'), index=False)

df=pd.DataFrame(x_position_label_testing)
df.to_csv(os.path.join(csv_path,'x_position_label_testing.csv'), index=False)

df=pd.DataFrame(y_position_label_training)
df.to_csv(os.path.join(csv_path,'y_position_label_training.csv'), index=False)

df=pd.DataFrame(y_position_label_testing)
df.to_csv(os.path.join(csv_path,'y_position_label_testing.csv'), index=False)

# velocity
df=pd.DataFrame(x_velocity_label_training)
df.to_csv(os.path.join(csv_path,'x_velocity_label_training.csv'), index=False)

df=pd.DataFrame(x_velocity_label_testing)
df.to_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), index=False)

df=pd.DataFrame(y_velocity_label_training)
df.to_csv(os.path.join(csv_path,'y_velocity_label_training.csv'), index=False)

df=pd.DataFrame(y_velocity_label_testing)
df.to_csv(os.path.join(csv_path,'y_velocity_label_testing.csv'), index=False)

# acceleration
df=pd.DataFrame(x_acceleration_label_training)
df.to_csv(os.path.join(csv_path,'x_acceleration_label_training.csv'), index=False)

df=pd.DataFrame(x_acceleration_label_testing)
df.to_csv(os.path.join(csv_path,'x_acceleration_label_testing.csv'), index=False)

df=pd.DataFrame(y_acceleration_label_training)
df.to_csv(os.path.join(csv_path,'y_acceleration_label_training.csv'), index=False)

df=pd.DataFrame(y_acceleration_label_testing)
df.to_csv(os.path.join(csv_path,'y_acceleration_label_testing.csv'), index=False)

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


# x_vel
training_y_3 = pd.read_csv(os.path.join(csv_path,'x_velocity_label_training.csv'), dtype=float)    
training_y_3  = torch.from_numpy(training_y_3.values)    
training_y_3  = training_y_3.float()

testing_y_3 = pd.read_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), dtype=float)    
testing_y_3 = torch.from_numpy(testing_y_3.values)    
testing_y_3 = testing_y_3.float()

# y_vel
training_y_4 = pd.read_csv(os.path.join(csv_path,'y_velocity_label_training.csv'), dtype=float)    
training_y_4 = torch.from_numpy(training_y_4.values)    
training_y_4 = training_y_4.float()

testing_y_4 = pd.read_csv(os.path.join(csv_path,'y_velocity_label_testing.csv'), dtype=float)    
testing_y_4 = torch.from_numpy(testing_y_4.values)    
testing_y_4 = testing_y_4.float()


# x_acc
training_y_5 = pd.read_csv(os.path.join(csv_path,'x_acceleration_label_training.csv'), dtype=float)    
training_y_5  = torch.from_numpy(training_y_5.values)    
training_y_5  = training_y_5.float()

testing_y_5 = pd.read_csv(os.path.join(csv_path,'x_acceleration_label_testing.csv'), dtype=float)    
testing_y_5 = torch.from_numpy(testing_y_5.values)    
testing_y_5 = testing_y_5.float()

# y_acc
training_y_6 = pd.read_csv(os.path.join(csv_path,'y_acceleration_label_training.csv'), dtype=float)    
training_y_6 = torch.from_numpy(training_y_6.values)    
training_y_6 = training_y_6.float()

testing_y_6 = pd.read_csv(os.path.join(csv_path,'y_acceleration_label_testing.csv'), dtype=float)    
testing_y_6 = torch.from_numpy(testing_y_6.values)    
testing_y_6 = testing_y_6.float()


shutil.rmtree(Firing_Rate_Visualization)

# Start plotting
start_time_bin = 200
end_time_bin = 400
my_fontsize = 30
my_plot_width = 30
my_plot_height = 20

CWD = os.getcwd()

if 'my_plot' not in CWD:
    CWD = os.path.join(CWD, 'my_plot')
    if not os.path.exists(CWD):
        os.mkdir(CWD)

plot_path = os.path.join(CWD, session_name )
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


# Figure firing rate only with seconds
'''
plt.figure(figsize=(my_plot_width, my_plot_height))

plt.title( 'Session ' + session_name + ' Firing Rate', fontsize=30, color="black")
sns.set(font_scale=3)
sns.set_style("white")
sns.color_palette(palette=None)

data = torch.transpose( testing_x[start_time_bin:end_time_bin,:], 0, 1)

# Eliminate empty units
valid_rows=[]
for row_idx in range(data.size(0)):
    if not torch.all( data[row_idx,:] ==0 ):
        valid_rows.append(row_idx)
data = data[valid_rows,:]

my_second_labels=[]
for second_label in range( len( time_stamp_64ms[start_time_bin:end_time_bin] ) ):
    my_second_labels.append( time_stamp_64ms[second_label] )

# https://stackoverflow.com/questions/47784215/seaborn-heatmap-custom-tick-values
num_ticks = 5
# the index of the position of yticks
xticks = np.linspace(0, len(my_second_labels) - 1, num_ticks, dtype=np.int)
# the content of labels of these yticks
xticklabels = [ my_second_labels[idx] for idx in xticks ]

# https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.colorbar.html
cbar_kws={"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.07, 'ticks' : [ torch.min(data), torch.max(data) ]}

ax = sns.heatmap( data=data, vmin=0, vmax=4, xticklabels=xticklabels, yticklabels=True, cbar_kws=cbar_kws, cmap='YlGnBu_r', cbar=False)
ax.set_xticks(xticks)

ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = my_fontsize, rotation=0)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = my_fontsize)




ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

plt.xlabel('Time (Seconds)', fontsize=my_fontsize, color="black")
plt.ylabel('Units', fontsize=my_fontsize, color="black")

plt.tight_layout()

plt.savefig(plot_path+'/' +'firing_rate_in_seconds'+'.png')

plt.clf()
plt.cla()
plt.close()
'''

# Official one

position_path = './position_data/'+session_name+'/csv_files'
velocity_path = './velocity_data/'+session_name+'/csv_files'
acceleration_path = './acceleration_data/'+session_name+'/csv_files'

position_x_predction = pd.read_csv(os.path.join(position_path,'my_prediction_x_pos.csv'), dtype=float, header=None)
position_x_predction = position_x_predction.to_numpy()
print('shape of position_x_predction= ', position_x_predction.shape , '\n')
position_y_predction = pd.read_csv(os.path.join(position_path,'my_predictiony_y_pos.csv'), dtype=float, header=None) # TODO fix the typo my_predictiony_y_pos
position_y_predction = position_y_predction.to_numpy()
print('shape of position_y_predction= ', position_y_predction.shape , '\n')
position_x_actual = pd.read_csv(os.path.join(position_path,'Ground_Truth_x_pos.csv'), dtype=float, header=None)
position_x_actual = position_x_actual.to_numpy()
print('shape of position_x_actual= ', position_x_actual.shape , '\n')
position_y_actual = pd.read_csv(os.path.join(position_path,'Ground_Truth_y_pos.csv'), dtype=float, header=None)
position_y_actual = position_y_actual.to_numpy()
print('shape of position_y_actual= ', position_y_actual.shape , '\n')
position_timestamp = pd.read_csv(os.path.join(position_path,'plotting_time_elapsed.csv'), dtype=float, header=None)
position_timestamp = position_timestamp.to_numpy()
print('shape of position_timestamp= ', position_timestamp.shape , '\n')


velocity_x_predction = pd.read_csv(os.path.join(velocity_path,'my_prediction_x_vel.csv'), dtype=float, header=None)
velocity_x_predction = velocity_x_predction.to_numpy()
print('shape of velocity_x_predction= ', velocity_x_predction.shape , '\n')
velocity_y_predction = pd.read_csv(os.path.join(velocity_path,'my_predictiony_y_vel.csv'), dtype=float, header=None) # TODO fix the typo my_predictiony_y_pos
velocity_y_predction = velocity_y_predction.to_numpy()
print('shape of velocity_y_predction= ', velocity_y_predction.shape , '\n')
velocity_x_actual = pd.read_csv(os.path.join(velocity_path,'Ground_Truth_x_vel.csv'), dtype=float, header=None)
velocity_x_actual = velocity_x_actual.to_numpy()
print('shape of velocity_x_actual= ', velocity_x_actual.shape , '\n')
velocity_y_actual = pd.read_csv(os.path.join(velocity_path,'Ground_Truth_y_vel.csv'), dtype=float, header=None)
velocity_y_actual = velocity_y_actual.to_numpy()
print('shape of velocity_y_actual= ', velocity_y_actual.shape , '\n')
velocity_timestamp = pd.read_csv(os.path.join(velocity_path,'plotting_time_elapsed.csv'), dtype=float, header=None)
velocity_timestamp = velocity_timestamp.to_numpy()
print('shape of velocity_timestamp= ', velocity_timestamp.shape , '\n')

acceleration_x_predction = pd.read_csv(os.path.join(acceleration_path,'my_prediction_x_acc.csv'), dtype=float, header=None)
acceleration_x_predction = acceleration_x_predction.to_numpy()
print('shape of acceleration_x_predction= ', acceleration_x_predction.shape , '\n')
acceleration_y_predction = pd.read_csv(os.path.join(acceleration_path,'my_predictiony_y_acc.csv'), dtype=float, header=None) # TODO fix the typo my_predictiony_y_pos
acceleration_y_predction = acceleration_y_predction.to_numpy()
print('shape of acceleration_y_predction= ', acceleration_y_predction.shape , '\n')
acceleration_x_actual = pd.read_csv(os.path.join(acceleration_path,'Ground_Truth_x_acc.csv'), dtype=float, header=None)
acceleration_x_actual = acceleration_x_actual.to_numpy()
print('shape of acceleration_x_actual= ', acceleration_x_actual.shape , '\n')
acceleration_y_actual = pd.read_csv(os.path.join(acceleration_path,'Ground_Truth_y_acc.csv'), dtype=float, header=None)
acceleration_y_actual = acceleration_y_actual.to_numpy()
print('shape of acceleration_y_actual= ', acceleration_y_actual.shape , '\n')
acceleration_timestamp = pd.read_csv(os.path.join(acceleration_path,'plotting_time_elapsed.csv'), dtype=float, header=None)
acceleration_timestamp = acceleration_timestamp.to_numpy()
print('shape of acceleration_timestamp= ', acceleration_timestamp.shape , '\n')

if ( position_timestamp.shape[0] < velocity_timestamp.shape[0]  ) and ( position_timestamp.shape[0]<acceleration_timestamp.shape[0] ) and (position_timestamp.shape[0]<time_stamp_64ms.shape[0]):

    a = np.where( velocity_timestamp==position_timestamp[0][0] )
    # position_timestamp[0][0] is eqal to velocity_timestamp[a]
    velocity_x_predction = velocity_x_predction[int(a[0]):]
    velocity_y_predction = velocity_y_predction[int(a[0]):]
    velocity_x_actual = velocity_x_actual[int(a[0]):]
    velocity_y_actual = velocity_y_actual[int(a[0]):]
    velocity_timestamp = velocity_timestamp[int(a[0]):]

    a = np.where( acceleration_timestamp==position_timestamp[0][0] )
    # position_timestamp[0][0] is eqal to acceleration_timestamp[a]
    acceleration_x_predction = acceleration_x_predction[int(a[0]):]
    acceleration_y_predction = acceleration_y_predction[int(a[0]):]
    acceleration_x_actual = acceleration_x_actual[int(a[0]):]
    acceleration_y_actual = acceleration_y_actual[int(a[0]):]
    acceleration_timestamp = acceleration_timestamp[int(a[0]):]

    a = np.where( time_stamp_64ms==position_timestamp[0][0] )
    time_stamp_64ms = time_stamp_64ms[int(a[0]):]
    testing_x = testing_x[int(a[0]):]

else:
    exit()


# Figure firing rate and all kinematic variables
plt.title( 'Session ' + session_name + ' Firing Rate', fontsize=30, color="black")

sns.set(font_scale=3)
sns.set_style("white")
sns.color_palette(palette=None)

f ,ax = plt.subplots(4,1, gridspec_kw={'height_ratios': [3, 1, 1, 1],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05}, constrained_layout=True , figsize=(my_plot_width, my_plot_height*1.2))


data = torch.transpose( testing_x[start_time_bin:end_time_bin,:], 0, 1)
# Eliminate empty units
valid_rows=[]
for row_idx in range(data.size(0)):
    if not torch.all( data[row_idx,:] ==0 ):
        valid_rows.append(row_idx)
data = data[valid_rows,:]

cbar_kws={"orientation": "horizontal", "shrink": 0.2, "aspect":20,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ torch.min(data), 4 ]}
sns.heatmap( data=data, vmax=4 ,xticklabels=False, yticklabels=True, cbar_kws=cbar_kws, cmap='YlGnBu_r', ax=ax[0]) # important, not ax[0] = sns.heatmap(...)
# ax[0].set_xticklabels(ax[0].get_xmajorticklabels(), fontsize = my_fontsize, rotation=0)
ax[0].set_title('Firing rate from session '+ session_name, fontsize=my_fontsize)
ax[0].set_yticklabels(ax[0].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)

# ax[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
# ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())

ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")



# ax[1].set_title( 'Position', fontsize=my_fontsize)
ax[1].plot( position_timestamp[start_time_bin:end_time_bin], position_x_predction[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-axis prediction', alpha=0.9 )
ax[1].plot(position_timestamp[start_time_bin:end_time_bin], position_y_predction[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-axis prediction', alpha=0.9 )
ax[1].plot( position_timestamp[start_time_bin:end_time_bin], position_x_actual[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-axis actual', alpha=0.8 )
ax[1].plot(position_timestamp[start_time_bin:end_time_bin], position_y_actual[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-axis actual', alpha=0.8 )
ax[1].set_ylabel( 'pos ($mm$)', fontsize=my_fontsize, rotation=90)
ax[1].get_xaxis().set_ticks([])
# ax[1].legend(loc='upper right', fontsize=my_fontsize)
ax[1].set_xlim([ position_timestamp[start_time_bin] ,  position_timestamp[end_time_bin]  ])
ax[1].set_ylim([ -150, 150  ])


# ax[2].set_title( 'Velocity', fontsize=my_fontsize)
ax[2].plot(velocity_timestamp[start_time_bin:end_time_bin], velocity_x_predction[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-axis prediction', alpha=0.9 )
ax[2].plot(velocity_timestamp[start_time_bin:end_time_bin], velocity_y_predction[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-axis prediction', alpha=0.9 )
ax[2].plot(velocity_timestamp[start_time_bin:end_time_bin], velocity_x_actual[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-axis actual', alpha=0.8 )
ax[2].plot(velocity_timestamp[start_time_bin:end_time_bin], velocity_y_actual[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-axis actual', alpha=0.8 )
ax[2].set_ylabel( 'vel ($mm/s$)', fontsize=my_fontsize, rotation=90)
ax[2].get_xaxis().set_ticks([])
# ax[2].legend(loc='upper right', fontsize=my_fontsize)
ax[2].set_xlim([ velocity_timestamp[start_time_bin] ,  velocity_timestamp[end_time_bin]  ])
ax[2].set_ylim([ -350, 350  ])


# ax[3].set_title( 'Acceleration', fontsize=my_fontsize)
ax[3].plot(acceleration_timestamp[start_time_bin:end_time_bin], acceleration_x_predction[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-axis prediction', alpha=0.9 )
ax[3].plot(acceleration_timestamp[start_time_bin:end_time_bin], acceleration_y_predction[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-axis prediction', alpha=0.9 )
ax[3].plot(acceleration_timestamp[start_time_bin:end_time_bin], acceleration_x_actual[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-axis actual', alpha=0.8 )
ax[3].plot(acceleration_timestamp[start_time_bin:end_time_bin], acceleration_y_actual[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-axis actual', alpha=0.8 )
ax[3].set_ylabel( 'acc ($mm/s^2$)', fontsize=my_fontsize, rotation=90)
ax[3].set_xlabel('Time (Seconds)', fontsize=my_fontsize)
# ax[3].legend(loc='upper right', fontsize=my_fontsize)
ax[3].set_xlim([ acceleration_timestamp[start_time_bin] ,  acceleration_timestamp[end_time_bin]  ])
ax[3].set_ylim([ -2500, 2500  ])


# plt.tight_layout()
plt.savefig( plot_path+'/'+ 'Firing_rate_heatmap_and_all_kinematic_variables_pred_and_actual_wit_color_bar' +'.png' )


plt.cla()
plt.clf()
plt.close()

