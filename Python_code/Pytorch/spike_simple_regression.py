# https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data


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
the_sampling_rate=16
file_numbers=1
time_lag=0
order=0
with_sorted_spikes=True
include_hash_unit=True

# Must know these two numbers beforehand
channel_numbers_in_this_dataset=96
units_numbers_in_this_dataset=3

if with_sorted_spikes==True:
    feature_numbers=channel_numbers_in_this_dataset*units_numbers_in_this_dataset
else:
    feature_numbers=channel_numbers_in_this_dataset


def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

def get_spike_bins_matrix(the_file_name, the_sampling_rate):
    with h5py.File(the_file_name, 'r') as mat_file:        
        time_stamp=mat_file['t']  
        # or
        # time_stamp=mat_file.get('t')
        # time_stamp=np.array(time_stamp)
        # time_stamp.shape = (1, 204446)
        spikes = mat_file['spikes']
        firing_rate_cell=[[]]
        

        print('spikes shape: ', spikes.shape) #  (3, 192) in indy_20160407_02
        channel_number=int(spikes.shape[1] / 2) # 96 in indy_20160407_02
        numpy_finger_pos_1=np.empty([])
        numpy_finger_pos_GET_1=mat_file.get('finger_pos')
        numpy_finger_pos_1=np.array(numpy_finger_pos_GET_1)
        print('numpy_finger_pos_1.shape: ', numpy_finger_pos_1.shape) # (3, 204446) in indy_20160407_02
        
        finger_z_coor=numpy_finger_pos_1[0][:]
        finger_x_coor=numpy_finger_pos_1[1][:]
        finger_y_coor=numpy_finger_pos_1[2][:]

        x_position_label=[]
        y_position_label=[]
        z_position_label=[]

        x_velocity_label=[]
        y_velocity_label=[]
        z_velocity_label=[]

        x_acceleration_label=[]
        y_acceleration_label=[]
        z_acceleration_label=[]

        time_stamp_64ms=[]

        sampling_rate=the_sampling_rate # because 64ms

        #duration=1000
        duration=time_stamp.shape[1]
        sampling_index=0

        # Too slow app. 70 seconds
        ''' 
        while sampling_index < duration:
            #print('sampling_index = ', sampling_index)
            print( 'Progress of making sampling array: '+ str(   round( (sampling_index / duration)*100, 3)   )+' %' )
            time_stamp_64ms.append(time_stamp[0][sampling_index])
            sampling_index+=sampling_rate
        '''
        time_stamp_64ms=time_stamp[0][::sampling_rate]  # way faster, app. 4 seconds
        print('lenght of time_stamp_64ms: ', len(time_stamp_64ms)) # 12778 in indy_20160407_02
        
        testing_data_index=int(int(len(time_stamp_64ms))*0.8) # split 80% into training
        print('testing_data_index= ',testing_data_index) # 10222 in indy_20160407_02.mat

        # make x, y, z position label matrix with the sampling_rate
        '''
        index_label=0
        while index_label < duration:
            x_position_label.append(finger_x_coor[index_label] )
            y_position_label.append(finger_y_coor[index_label] )
            z_position_label.append(finger_z_coor[index_label] )
            index_label+=sampling_rate
        '''
        x_position_label=finger_x_coor[::sampling_rate] 
        x_position_label=x_position_label[1:] # [:-1] original, match Chieh's result
        y_position_label=finger_y_coor[::sampling_rate]
        y_position_label=y_position_label[1:]
        z_position_label=finger_z_coor[::sampling_rate]
        z_position_label=z_position_label[1:]
        print('Position label arrays finished')

        # Making spike counts matrix
        for channel_index in range(channel_number):
            #print('Channel progress: ' + str( round( (channel_index/channel_number)*100, 3) )+' %' ) # 96 channels in this dataset
            
            temp_spike_cell_1=[]
            temp_spike_cell_2=[]
            temp_spike_cell_3=[]

            temp_spike_cell_1=mat_file[ ( spikes[0][channel_index] ) ][()]
            temp_spike_cell_2=mat_file[ ( spikes[1][channel_index] ) ][()]
            temp_spike_cell_3=mat_file[ ( spikes[2][channel_index] ) ][()]

            temp_spike_cell_1=np.asarray(temp_spike_cell_1)
            temp_spike_cell_2=np.asarray(temp_spike_cell_2)
            temp_spike_cell_3=np.asarray(temp_spike_cell_3)

            time_stamp_64ms=np.asarray(time_stamp_64ms)

        
            temp_spike_cell_1=temp_spike_cell_1.flatten()
            temp_spike_cell_2=temp_spike_cell_2.flatten()
            temp_spike_cell_3=temp_spike_cell_3.flatten()
            time_stamp_64ms=time_stamp_64ms.flatten()
            
            '''
            print('shape of temp_spike_cell_1: ',temp_spike_cell_1.shape) # (5595,) in channel 1, indy_20160407_02
            print('shape of temp_spike_cell_2: ',temp_spike_cell_2.shape) # (2,) in channel 1, indy_20160407_02
            print('shape of temp_spike_cell_3: ',temp_spike_cell_3.shape) # (2,) in channel 1, indy_20160407_02
            print('shape of time_stamp_64ms: ',time_stamp_64ms.shape)
            '''

            if temp_spike_cell_1.shape[0] != 2 and include_hash_unit==True:

                yee=histc(temp_spike_cell_1, time_stamp_64ms)
                #print('shape of yee:  ',yee.shape)
                firing_rate_cell.append(yee[:-1])
                #print('yee: ',yee)

            else:
                r = np.zeros( len(time_stamp_64ms)-1 )
                firing_rate_cell.append(r)
            '''
            print('length of firing_rate in cell 1: ',end='')
            print(len(firing_rate_cell[:-1]))
            '''

            firing_rate_cell.append([])  

            if temp_spike_cell_2.shape[0] != 2:

                # firing rate
                yee=histc(temp_spike_cell_2, time_stamp_64ms)
                #print('shape of yee:  ',yee.shape)
                firing_rate_cell.append(yee[:-1])

            else:
                r = np.zeros( len(time_stamp_64ms)-1 )
                firing_rate_cell.append(r)

            '''
            print('length of firing_rate in cell 2: ',end='')
            print(len(firing_rate_cell[-1]))
            '''
            firing_rate_cell.append([])

            if temp_spike_cell_3.shape[0] != 2:
                
                # firing rate
                yee=histc(temp_spike_cell_3, time_stamp_64ms)
                #print('shape of yee:  ',yee.shape)
                firing_rate_cell.append(yee[:-1])

            else:
                r = np.zeros( len(time_stamp_64ms)-1 )
                firing_rate_cell.append(r)
            '''
            print('length of firing_rate in cell 3: ',end='')
            print(len(firing_rate_cell[-1]))
            print('\n\n')
            '''
            firing_rate_cell.append([])

            '''
            print('row numbers of firing_rate_cell: ',end='')
            print( len( firing_rate_cell) )
            print('\n')
            '''
            '''
            for row_index in range( len( firing_rate_cell) ):            
                print('length of firing_rate_cell['+ str(row_index) +']: ',end='')
                print(len(firing_rate_cell[row_index]))
            print('\n')
            print('End of one channel '+ str(channel_index+1) +'\n') 
            '''
    return [firing_rate_cell, channel_number, testing_data_index, x_position_label, y_position_label, z_position_label]

def get_labels(the_file_name, the_sampling_rate):
    with h5py.File(the_file_name, 'r') as mat_file:
        finger_pos = mat_file['finger_pos']
        time_stamp=mat_file['t']

        numpy_finger_pos_GET_2=mat_file.get('finger_pos')
        numpy_finger_pos_2=np.array(numpy_finger_pos_GET_2)

        numpy_time_stamp=mat_file.get('t')
        numpy_time_stamp=np.array(numpy_time_stamp)

        print('numpy_finger_pos_2 shape: ', numpy_finger_pos_2.shape) #  (3, 204446) in indy_20160407_02
        print('\n') 

        print('numpy_time_stamp: ',end='')
        print(numpy_time_stamp.shape)  #  (1, 204446)in indy_20160407_02

        sampling_rate=the_sampling_rate
        time_stamp_64ms=time_stamp[0][::sampling_rate]

        finger_z_pos_64ms=numpy_finger_pos_2[0][::sampling_rate]
        finger_x_pos_64ms=numpy_finger_pos_2[1][::sampling_rate]
        finger_y_pos_64ms=numpy_finger_pos_2[2][::sampling_rate]
        print('shape of finger_x_pos_64ms', finger_x_pos_64ms.shape) # (12778,) in indy_20160407_02

        finger_x_velocity=[]
        finger_y_velocity=[]
        finger_z_velocity=[]
        velocity_time_coor=[]

        finger_x_acceleration=[]
        finger_y_acceleration=[]
        finger_z_acceleration=[]
        acceleration_time_coor=[]

        duration=time_stamp_64ms.shape[0]

        for i in range(duration):
            #print('Velocity computing progress: ' + str( round( (i/duration)*100, 3) )+' %' )
            
            if ( i<duration-1 ):
                #velocity=( finger_z_pos_64ms[i+1] - finger_z_pos_64ms[i] ) / ( time_stamp_64ms[i+1]-time_stamp_64ms[i] )
                velocity=( finger_z_pos_64ms[i+1] - finger_z_pos_64ms[i] ) / ( 1 )
                finger_z_velocity.append(velocity)

                #velocity=( finger_x_pos_64ms[i+1] - finger_x_pos_64ms[i] ) / ( time_stamp_64ms[i+1]-time_stamp_64ms[i] )
                velocity=( finger_x_pos_64ms[i+1] - finger_x_pos_64ms[i] ) / ( 1 )
                finger_x_velocity.append(velocity)

                #velocity=( finger_y_pos_64ms[i+1] - finger_y_pos_64ms[i] ) / ( time_stamp_64ms[i+1]-time_stamp_64ms[i] )
                velocity=( finger_y_pos_64ms[i+1] - finger_y_pos_64ms[i] ) / ( 1 )
                finger_y_velocity.append(velocity)

                velocity_time_coor.append( numpy_time_stamp[0][i] )

            else:
                '''
                finger_x_velocity.append(0)
                finger_y_velocity.append(0)
                finger_z_velocity.append(0)
                velocity_time_coor.append(0)
                '''
                pass
        
        finger_x_velocity=np.array(finger_x_velocity)
        finger_x_velocity=finger_x_velocity.astype(np.float32)
        
        finger_y_velocity=np.array(finger_y_velocity)
        finger_y_velocity=finger_y_velocity.astype(np.float32)

        finger_z_velocity=np.array(finger_z_velocity)
        finger_z_velocity=finger_z_velocity.astype(np.float32)

        velocity_time_coor=np.array(velocity_time_coor)

        duration=velocity_time_coor.shape[0]
        for i in range(duration):
            #print('Aceeleration computing progress '+ str( round( (i/duration)*100, 3) )+' %')

            if(i<duration-1):
                #acceleration=(finger_x_velocity[i+1]-finger_x_velocity[i])/ (velocity_time_coor[i+1]-velocity_time_coor[i] )
                acceleration=(finger_x_velocity[i+1]-finger_x_velocity[i])/ ( 1 )
                finger_x_acceleration.append(acceleration)

                #acceleration=(finger_y_velocity[i+1]-finger_y_velocity[i])/(velocity_time_coor[i+1]-velocity_time_coor[i])
                acceleration=(finger_y_velocity[i+1]-finger_y_velocity[i])/( 1 )
                finger_y_acceleration.append(acceleration)

                #acceleration=(finger_z_velocity[i+1]-finger_z_velocity[i])/(velocity_time_coor[i+1]-velocity_time_coor[i])
                acceleration=(finger_z_velocity[i+1]-finger_z_velocity[i])/( 1 )
                finger_z_acceleration.append(acceleration)

                acceleration_time_coor.append(velocity_time_coor[i])
            else:
                '''
                finger_x_acceleration.append(0)
                finger_y_acceleration.append(0)
                finger_z_acceleration.append(0)
                acceleration_time_coor.append(0)
                '''
                pass
        finger_x_acceleration=np.array(finger_x_acceleration)
        finger_x_acceleration=finger_x_acceleration.astype(np.float32)
        
        finger_y_acceleration=np.array(finger_y_acceleration)
        finger_y_acceleration=finger_y_acceleration.astype(np.float32)
            
        finger_z_acceleration=np.array(finger_z_acceleration)
        finger_z_acceleration=finger_z_acceleration.astype(np.float32)

        acceleration_time_coor=np.array(acceleration_time_coor)

        x_velocity_label=finger_x_velocity
        y_velocity_label=finger_y_velocity
        z_velocity_label=finger_z_velocity

        x_acceleration_label=finger_x_acceleration
        y_acceleration_label=finger_y_acceleration
        z_acceleration_label=finger_z_acceleration
    return [x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]

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

    [firing_rate_cell, channel_number, testing_data_index, x_position_label, y_position_label, z_position_label]=get_spike_bins_matrix(file_list[session_index], the_sampling_rate)
    [x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]=get_labels(file_list[session_index], the_sampling_rate)

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
    print('fetures list shape: ',end='')
    print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
    print('\n')


    # Organizing feature Matrix and labels

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
    

# cross sessions control end

# All models fit and predict, show R2 score

x = torch.from_numpy(X_for_training)
y = torch.from_numpy(x_position_label_training)

x=x.float()
y=y.float()
#x = X_for_training
#y = x_position_label_training

# torch can only train on Variable, so convert them to Variable
# x, y = Variable(x), Variable(y)


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

net = Net(n_feature=288, n_hidden=50, n_output=1)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


# train the network
for t in range(2000):
  
    prediction = net(x).flatten()     # input x and predict based on x
    #print('size of prediction= ', prediction.shape, ' size of y= ',y.shape,'\n')

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients


    
    # plot and show learning process
    '''
    plt.cla()
    ax.set_title('Regression Analysis', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-1.05, 1.5)
    ax.set_ylim(-0.25, 1.25)
    ax.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
    ax.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)
    ax.text(1.0, 0.1, 'Step = %d' % t, fontdict={'size': 24, 'color':  'red'})
    ax.text(1.0, 0, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 24, 'color':  'red'})
    

    # Used to return the plot as an image array 
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    my_images.append(image)
    '''



print('shape of x_position_label_training = ', x_position_label_training.shape, '\n shape of prediction = ', prediction.shape, '\n')
print('\n* model_x_position score in order ', order_index, ': ', r2_score( x_position_label_training.flatten(), prediction.data.numpy()))
