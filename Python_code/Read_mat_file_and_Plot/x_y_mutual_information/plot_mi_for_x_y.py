# Figures
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
width_two=0.2
# Data Processing
import pandas as pd
import json
import math
import numpy as np
from numpy import linalg as LA
import h5py
from sklearn import datasets, svm, metrics
# Regression Problem Evaluation Methods
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
# Read/Write file
import os
CWD_origin=os.getcwd()
import shutil
import gcmi
import time
tStart=time.time()

# My module
import sys
sys.path.append("../..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file
import data_processing.load_chewie_mat_file as load_chewie_mat_file
my_parameters = my_parameters.my_parameters()
mat_file_processing = load_mat_file.mat_file_processing()
chewie_file_processing = load_chewie_mat_file.mat_file_processing()

import data_processing.some_modules as some_modules
regular_modules = some_modules.regular_modules()


# Make file list
kinematic_variable_type='x_and_y_pos' # x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc
FILE_PATH = '../../../Dataset/Sorted_Spike_Dataset/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

# Model Performance Lists
R_square_across_all_sessions=[]
SNR_across_all_sessions=[]
RMSE_across_all_sessions=[]
best_epoch_arcoss_all_sessions=[]
person_correlation_coefficient_across_all_sessions=[]
testing_data_length_all_sessions = []
my_best_epoch_dict={}

CWD_origin=os.getcwd()

position_figures_path = os.path.join(CWD_origin, 'position_figures')
if not os.path.exists(position_figures_path):
    os.mkdir(position_figures_path)

velocity_figures_path = os.path.join(CWD_origin, 'velocity_figures')
if not os.path.exists(velocity_figures_path):
    os.mkdir(velocity_figures_path)

acceleration_figures_path = os.path.join(CWD_origin, 'acceleration_figures')
if not os.path.exists(acceleration_figures_path):
    os.mkdir(acceleration_figures_path)

# session control start
for session_k in range(len(session_file_list)):

    session_name = str(session_file_list[session_k])[:-4]

    if session_name.startswith('indy') or session_name.startswith('loco'):

        file_name_1='../../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'    
        time_stamp_64ms=[]

        # Auto-assigned parameters
        testing_data_index=0
        channel_number=0
        units_have_value=0

        # Parameters should be assigned
        the_sampling_rate = my_parameters.the_sampling_rate
        file_numbers = my_parameters.file_numbers
        time_lag = my_parameters.time_lag

        with_sorted_spikes = False
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
        firing_rate_matrix = regular_modules.with_or_without_sorting(with_sorted_spikes, firing_rate_matrix, channel_number, unit_number)

        firing_rate_matrix=np.transpose(firing_rate_matrix)        
        feature_numbers_of_firing_rate = firing_rate_matrix.shape[1]

        X=firing_rate_matrix.astype(np.float32)
        print('features list shape: ',end='')
        print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
        print('\n')

    if session_name.startswith('Chewie'):

        file_name_1='../../../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'    
        time_stamp_64ms=[]

        # Auto-assigned parameters
        testing_data_index=0
        channel_number=0
        units_have_value=0

        # Parameters should be assigned
        the_sampling_rate = 64
        file_numbers = my_parameters.file_numbers
        time_lag = my_parameters.time_lag

        print('In session '+ session_name + ': ' + '\n' )

        # Load Spike Firing Rate
        [firing_rate_cell, testing_data_index, time_stamp_64ms, unit_number] = chewie_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate)

        firing_rate_final=[] # not[[]]
        for row_index in range( len( firing_rate_cell) ):   
            if len(firing_rate_cell[row_index]):
                firing_rate_final.append( firing_rate_cell[row_index] )
                units_have_value+=1

        feature_numbers = unit_number

        firing_rate_matrix=np.array(firing_rate_final)
        firing_rate_matrix=np.transpose(firing_rate_matrix)        
        feature_numbers_of_firing_rate = firing_rate_matrix.shape[1]
        X=firing_rate_matrix.astype(np.float32)

        # Create empty arrrays from data
        [X_for_training, X_for_prediction, 
        x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
        x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
        x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,
        x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing ]=chewie_file_processing.create_empty_traing_and_testing_label(feature_numbers)

        [time_stamp_64ms, 
        x_position_label, y_position_label,  
        x_velocity_label, y_velocity_label,  
        x_acceleration_label, y_acceleration_label, 
        x_position_target, y_position_target] = chewie_file_processing.get_labels(file_name_1, the_sampling_rate)


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


    mi_of_all_channels_x_pos=[]
    mi_of_all_channels_y_pos=[]
    for i in range(96):
        yee=gcmi.gcmi_cc(  X_for_training[:,i] , x_position_label_training ) 
        mi_of_all_channels_x_pos.append(yee)
        yee=gcmi.gcmi_cc(  X_for_training[:,i] , y_position_label_training ) 
        mi_of_all_channels_y_pos.append(yee)

    plt.figure(figsize=(16,3))
    ind = np.arange(1 , len(mi_of_all_channels_x_pos)+1)

    plt.bar(ind-width_two,  mi_of_all_channels_x_pos, width=width_two, color='b')
    plt.bar(ind,            mi_of_all_channels_y_pos, width=width_two, color='g')

    plt.legend(['x-pos', 'y-pos'], loc='upper right')
    plt.ylabel('GCMI')
    plt.xlabel('Channels')
    plt.xlim([0,96+1])
    plt.ylim([0, 0.02])
    plt.xticks(ind, rotation=-90)


    plt.grid(False)
    plt.title('Position Mutual Information '+ session_name)
    plt.tight_layout()
    plt.savefig( position_figures_path+'/'+ session_name + '_pos_MI'+'.png')
    # plt.show()

    plt.cla()
    plt.clf()
    plt.close()


    mi_of_all_channels_x_vel=[]
    mi_of_all_channels_y_vel=[]
    for i in range(96):
        yee=gcmi.gcmi_cc(  X_for_training[:,i] , x_velocity_label_training ) 
        mi_of_all_channels_x_vel.append(yee)
        yee=gcmi.gcmi_cc(  X_for_training[:,i] , y_velocity_label_training ) 
        mi_of_all_channels_y_vel.append(yee)

    plt.figure(figsize=(16,3))
    ind = np.arange(1 , len(mi_of_all_channels_x_vel)+1)

    plt.bar(ind-width_two,  mi_of_all_channels_x_vel, width=width_two, color='b')
    plt.bar(ind,            mi_of_all_channels_y_vel, width=width_two, color='g')

    plt.legend(['x-vel', 'y-vel'], loc='upper right')
    plt.ylabel('GCMI')
    plt.xlabel('Channels')
    plt.xlim([0,96+1])
    plt.ylim([0, 0.02])
    plt.xticks(ind, rotation=-90)


    plt.grid(False)
    plt.title('Velocity Mutual Information '+ session_name)
    plt.tight_layout()
    plt.savefig( velocity_figures_path+'/'+ session_name +'_vel_MI'+ '.png')
    # plt.show()

    plt.cla()
    plt.clf()
    plt.close()


    mi_of_all_channels_x_acc=[]
    mi_of_all_channels_y_acc=[]
    for i in range(96):
        yee=gcmi.gcmi_cc(  X_for_training[:,i] , x_velocity_label_training ) 
        mi_of_all_channels_x_acc.append(yee)
        yee=gcmi.gcmi_cc(  X_for_training[:,i] , y_velocity_label_training ) 
        mi_of_all_channels_y_acc.append(yee)

    plt.figure(figsize=(16,3))
    ind = np.arange(1 , len(mi_of_all_channels_y_acc)+1)

    plt.bar(ind-width_two,  mi_of_all_channels_x_acc, width=width_two, color='b')
    plt.bar(ind,            mi_of_all_channels_y_acc, width=width_two, color='g')

    plt.legend(['x-acc', 'y-acc'], loc='upper right')
    plt.ylabel('GCMI')
    plt.xlabel('Channels')
    plt.xlim([0,96+1])
    plt.ylim([0, 0.02])
    plt.xticks(ind, rotation=-90)


    plt.grid(False)
    plt.title('Acceleration Mutual Information '+ session_name)
    plt.tight_layout()
    plt.savefig( acceleration_figures_path+'/'+ session_name +'_acc_MI'+ '.png')
    # plt.show()

    plt.cla()
    plt.clf()
    plt.close()