# -*- coding: utf-8 -*-
import numpy as np
import h5py
import time
import matplotlib.pyplot as plot 
import copy

from sklearn.linear_model import LinearRegression
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from  sklearn.svm import SVC
from sklearn.svm import SVR

file_name_1='indy_20160407_02.mat'
tStart=time.time()

###################################### Auto-assigned parameters
#testing_data_index=5000
#testing_data_index=10222
testing_data_index=0 # Should be 10222 in indy_20160407_02
channel_number=0
units_have_value=0 # unit numbers that is not empty
feature_numbers=0

###################################### Parameters should be assigned
time_lag=0
order=2
with_sorted_spikes=True
include_hash_unit=True


def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r


# Cauculating spike bins below
with h5py.File(file_name_1, 'r') as mat_file:

    time_stamp=mat_file['t']  
    # or
    # time_stamp=mat_file.get('t')
    # time_stamp=np.array(time_stamp)
    # time_stamp.shape = (1, 204446)
    spikes = mat_file['spikes']
    firing_rate_cell=[[]]
    firing_rate_final=[] # not[[]]

    print('spikes shape: ', spikes.shape) #  (3, 192) in indy_20160407_02
    channel_number=int(spikes.shape[1] / 2) # 96 in indy_20160407_02

    numpy_finger_pos=mat_file.get('finger_pos')
    numpy_finger_pos=np.array(numpy_finger_pos)
    print('numpy_finger_pos.shape: ', numpy_finger_pos.shape) # (3, 204446) in indy_20160407_02

    finger_z_coor=numpy_finger_pos[0][:]
    finger_x_coor=numpy_finger_pos[1][:]
    finger_y_coor=numpy_finger_pos[2][:]

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

    sampling_rate=16 # because 64ms

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
# Cauculating spike bins above

# Calculating velocity and acceleration below
with h5py.File(file_name_1, 'r') as mat_file:

    finger_pos = mat_file['finger_pos']
    time_stamp=mat_file['t']

    numpy_finger_pos=mat_file.get('finger_pos')
    numpy_finger_pos=np.array(numpy_finger_pos)
    numpy_time_stamp=mat_file.get('t')
    numpy_time_stamp=np.array(numpy_time_stamp)

    print('numpy_finger_pos shape: ',end='')
    print(numpy_finger_pos.shape) #  (3, 204446) in indy_20160407_02
    print('numpy_time_stamp: ',end='')
    print(numpy_time_stamp.shape)  #  (1, 204446)in indy_20160407_02


    time_stamp_64ms=time_stamp[0][::sampling_rate]

    finger_z_pos_64ms=numpy_finger_pos[0][::sampling_rate]
    finger_x_pos_64ms=numpy_finger_pos[1][::sampling_rate]
    finger_y_pos_64ms=numpy_finger_pos[2][::sampling_rate]
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
    finger_x_velocity=finger_x_velocity.astype(np.float64)
    
    finger_y_velocity=np.array(finger_y_velocity)
    finger_y_velocity=finger_y_velocity.astype(np.float64)

    finger_z_velocity=np.array(finger_z_velocity)
    finger_z_velocity=finger_z_velocity.astype(np.float64)

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
    finger_x_acceleration=finger_x_acceleration.astype(np.float64)
    
    finger_y_acceleration=np.array(finger_y_acceleration)
    finger_y_acceleration=finger_y_acceleration.astype(np.float64)

    finger_z_acceleration=np.array(finger_z_acceleration)
    finger_z_acceleration=finger_z_acceleration.astype(np.float64)

    acceleration_time_coor=np.array(acceleration_time_coor)

    x_velocity_label=finger_x_velocity
    y_velocity_label=finger_y_velocity
    z_velocity_label=finger_z_velocity

    x_acceleration_label=finger_x_acceleration
    y_acceleration_label=finger_y_acceleration
    z_acceleration_label=finger_z_acceleration
# Caluclating velocity and acceleration above

# Extract firing_rate_cell with rows have length bigger than zero
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
        for k in range(channel_number-2): # Maximum 3 units in this session, indy_20160407_02.
            #print('index: ',index,end='')
            firing_rate_matrix[index][i]=no_sorting_firing_rate[k][i]+no_sorting_firing_rate[k+1][i]+no_sorting_firing_rate[k+2][i]

            # Test another way to exclude hash unit, but this only works in 96 features.
            #firing_rate_matrix[index][i]=no_sorting_firing_rate[k][i]+no_sorting_firing_rate[k+1][i]+no_sorting_firing_rate[k+2][i]

            #print('     firing_rate_matrix[index][i]: ',firing_rate_matrix[index][i] )
            index = index+1

    print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
    print('no_sorting_firing_rate shape: ', no_sorting_firing_rate.shape) # (288, 12777)
    print('\n')
else:
    pass


# Eliminate hash unit
no_hash_unit_firing_rate=firing_rate_matrix.copy()

# Making label data
x_position_label=np.array(x_position_label)
x_position_label=x_position_label.astype(np.float64)
print('position x_position_label  list shape: ',end='') 
print( x_position_label.shape ) # x is the label array should be feed into the model, (12777,)
print('\n')

y_position_label=np.array(y_position_label)
y_position_label=y_position_label.astype(np.float64)
print('position y_position_label list shape: ',end='')
print( y_position_label.shape ) # y is the label array should be feed into the model, (12777,)
print('\n')

z_position_label=np.array(z_position_label)
z_position_label=z_position_label.astype(np.float64)
print('position z_position_label list shape: ',end='')
print( z_position_label.shape ) # z is the label array should be feed into the model, (12777,)
print('\n')

firing_rate_matrix=np.transpose(firing_rate_matrix)
print('transposed firing_rate_matrix shape: ', firing_rate_matrix.shape) # (12777, 288) in indy_20160407_02
print('\n')
feature_numbers= firing_rate_matrix.shape[1]

X=firing_rate_matrix.astype(np.float64)
print('fetures list shape: ',end='')
print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
print('\n')

'''
# Order 1 feature matrix making
order_1_offset=1
temp1=X[order_1_offset:,:]
temp2=X[:-order_1_offset,:]
print('temp1 and temp2 shape: ', temp1.shape, temp2.shape)
X_order_1=np.concatenate((temp1, temp2),axis=1)
print('order 1 fetures list shape: ', X_order_1.shape) #  X_order_1 is the feature matrix, (12776, 576) in indy_20160407_02
print('\n')
'''
'''
# Order 2 feature matrix making
order_2_offset=2
temp1=X[order_2_offset:,:]
temp2=X[order_2_offset-1:-1,:]
temp3=X[:-order_2_offset,:]
print('temp1, temp2, temp3 shape: ', temp1.shape, temp2.shape, temp3.shape)
X_order_2=np.concatenate((temp1, temp2, temp3), axis=1)
print('order 2 fetures list shape: ', X_order_2.shape) #  X_order_2 is the feature matrix, (12775, 864) in indy_20160407_02
print('\n')
'''

# Universal order feature matrix making
print('In time lag: ', time_lag, '\n')
for order_index in range(order+1):
    if order_index ==2:
        
        order_2_offset=order_index # order_2_offset should be deprecated
        order_original_matrix=X[:-order_index, :]
        for order_loop_index in range(1, order_index):
            temp_order_matrix=X[order_loop_index: -(order_index-order_loop_index), :]
            order_original_matrix=np.concatenate((order_original_matrix, temp_order_matrix), axis=1)
        final_order_matrix=X[order_index:, :]
        order_original_matrix=np.concatenate((order_original_matrix, final_order_matrix), axis=1)
        X_order_2=order_original_matrix.copy() # X_order_2 should be deprecated        
        #print('order 2 fetures list shape: ', X_order_2.shape) #  X_order_2 is the feature matrix, (12775, 864) in indy_20160407_02
        print('\n')

        # New
        XX=order_original_matrix.copy()

        model_x_position = LinearRegression(fit_intercept=True)
        model_x_position.fit( XX[:testing_data_index, :], x_position_label[order_index+time_lag:testing_data_index+order_index+time_lag] )
        if time_lag==0:
            x_position_predict=model_x_position.predict( XX[testing_data_index:] )
        else:
            x_position_predict=model_x_position.predict( XX[testing_data_index:-time_lag] )
        print('* model_x_position score in order ', order_index, ': ', r2_score(  x_position_label[testing_data_index+order_index+time_lag:], x_position_predict ))

        model_y_position = LinearRegression(fit_intercept=True)
        model_y_position.fit( XX[:testing_data_index, :], y_position_label[order_index+time_lag:testing_data_index+order_index+time_lag] )
        if time_lag==0:
            y_position_predict=model_y_position.predict( XX[testing_data_index:] )
        else:
            y_position_predict=model_y_position.predict( XX[testing_data_index:-time_lag] )
        print('* model_y_position score in order ', order_index, ': ', r2_score(  y_position_label[testing_data_index+order_index+time_lag:], y_position_predict ))

        model_z_position = LinearRegression(fit_intercept=True)
        model_z_position.fit( XX[:testing_data_index, :], z_position_label[order_index+time_lag:testing_data_index+order_index+time_lag] )
        if time_lag==0:
            z_position_predict=model_z_position.predict( XX[testing_data_index:] )
        else:
            z_position_predict=model_z_position.predict( XX[testing_data_index:-time_lag] )
        print('* model_z_position score in order ', order_index, ': ', r2_score(  z_position_label[testing_data_index+order_index+time_lag:], z_position_predict ))

    if order_index==1:

        order_1_offset=order_index # order_1_offset should be deprecated
        temp1=X[order_index:,:]
        temp2=X[:-order_index,:]
        #print('temp1 and temp2 shape: ', temp1.shape, temp2.shape)
        X_order_1=np.concatenate((temp1, temp2), axis=1)  # X_order_1 should be deprecated
        #print('order 1 fetures list shape: ', X_order_1.shape) #  X_order_1 is the feature matrix, (12776, 576) in indy_20160407_02
        print('\n')

        # New
        XX=np.concatenate((temp1, temp2), axis=1)

        model_x_position = LinearRegression(fit_intercept=True)
        model_x_position.fit( XX[:testing_data_index, :], x_position_label[order_index+time_lag:testing_data_index+order_index+time_lag] )
        if time_lag==0:
            x_position_predict=model_x_position.predict( XX[testing_data_index:] )
        else:
            x_position_predict=model_x_position.predict( XX[testing_data_index:-time_lag] )
        print('* model_x_position score in order ', order_index, ': ', r2_score(  x_position_label[testing_data_index+order_index+time_lag:], x_position_predict ))

        model_y_position = LinearRegression(fit_intercept=True)
        model_y_position.fit( XX[:testing_data_index, :], y_position_label[order_index+time_lag:testing_data_index+order_index+time_lag] )
        if time_lag==0:
            y_position_predict=model_y_position.predict( XX[testing_data_index:] )
        else:
            y_position_predict=model_y_position.predict( XX[testing_data_index:-time_lag] )
        print('* model_y_position score in order ', order_index, ': ', r2_score(  y_position_label[testing_data_index+order_index+time_lag:], y_position_predict ))

        model_z_position = LinearRegression(fit_intercept=True)
        model_z_position.fit( XX[:testing_data_index, :], z_position_label[order_index+time_lag:testing_data_index+order_index+time_lag] )
        if time_lag==0:
            z_position_predict=model_z_position.predict( XX[testing_data_index:] )
        else:
            z_position_predict=model_z_position.predict( XX[testing_data_index:-time_lag] )
        print('* model_z_position score in order ', order_index, ': ', r2_score(  z_position_label[testing_data_index+order_index+time_lag:], z_position_predict ))

    if order_index==0:
        #pass

        # New
        model_x_position = LinearRegression(fit_intercept=True)
        model_x_position.fit( X[:testing_data_index, :], x_position_label[time_lag:testing_data_index+time_lag ] )
        if time_lag==0:
            x_position_predict=model_x_position.predict( X[testing_data_index:] )
        else:
            x_position_predict=model_x_position.predict( X[testing_data_index:-time_lag] )
        print('* model_x_position score in order ', order_index, ': ', r2_score( x_position_label[testing_data_index+time_lag:], x_position_predict))

        model_y_position = LinearRegression(fit_intercept=True)
        model_y_position.fit( X[:testing_data_index, :], y_position_label[time_lag:testing_data_index+time_lag ] )
        if time_lag==0:
            y_position_predict=model_y_position.predict( X[testing_data_index:] )
        else:
            y_position_predict=model_y_position.predict( X[testing_data_index:-time_lag] )
        print('* model_y_position score in order ', order_index, ': ', r2_score( y_position_label[testing_data_index+time_lag:], y_position_predict))

        model_z_position = LinearRegression(fit_intercept=True)
        model_z_position.fit( X[:testing_data_index, :], z_position_label[time_lag:testing_data_index+time_lag ] )
        if time_lag==0:
            z_position_predict=model_z_position.predict( X[testing_data_index:] )
        else:
            z_position_predict=model_z_position.predict( X[testing_data_index:-time_lag] )
        print('* model_z_position score in order ', order_index, ': ', r2_score( z_position_label[testing_data_index+time_lag:], z_position_predict))

print('\n>>>', 'Session(s): ', file_name_1,  '. Time Lag: ', time_lag, '. Feature numbers: ', feature_numbers, '. Include hash unit: ', include_hash_unit,'.  >>>')
print('\n')

# X position model fit and score
model_x_position = LinearRegression(fit_intercept=True)
model_x_position.fit( X[:testing_data_index, :], x_position_label[time_lag:testing_data_index+time_lag ] )
if time_lag==0:
    x_position_predict=model_x_position.predict( X[testing_data_index:] )
else:
    x_position_predict=model_x_position.predict( X[testing_data_index:-time_lag] )
print('* model_x_position score: ', r2_score( x_position_label[testing_data_index+time_lag:], x_position_predict))

model_x_position_order_1 = LinearRegression(fit_intercept=True)
model_x_position_order_1.fit( X_order_1[:testing_data_index, :], x_position_label[order_1_offset+time_lag:testing_data_index+order_1_offset+time_lag] )
if time_lag==0:
    x_position_predict_order_1=model_x_position_order_1.predict( X_order_1[testing_data_index:] )
else:
    x_position_predict_order_1=model_x_position_order_1.predict( X_order_1[testing_data_index:-time_lag] )
print('* model_x_position_order_1 score: ', r2_score( x_position_label[testing_data_index+order_1_offset+time_lag:], x_position_predict_order_1 ))

model_x_position_order_2 = LinearRegression(fit_intercept=True)
model_x_position_order_2.fit( X_order_2[:testing_data_index,:], x_position_label[order_2_offset+time_lag:testing_data_index+order_2_offset+time_lag])
if time_lag==0:
    x_position_predict_order_2=model_x_position_order_2.predict( X_order_2[testing_data_index:])
else:
    x_position_predict_order_2=model_x_position_order_2.predict( X_order_2[testing_data_index:-time_lag])
print('* model_x_position_order_2 score: ', r2_score( x_position_label[testing_data_index+order_2_offset+time_lag:], x_position_predict_order_2 ))

print('\n')

# Y position model fit and score
model_y_position = LinearRegression(fit_intercept=True)
model_y_position.fit( X[:testing_data_index, :], y_position_label[time_lag:testing_data_index+time_lag ])
if time_lag==0:
    y_position_predict=model_y_position.predict( X[testing_data_index:] )
else:
    y_position_predict=model_y_position.predict( X[testing_data_index:-time_lag] )
print('* model_y_position score: ', r2_score( y_position_label[testing_data_index+time_lag:], y_position_predict))

model_y_position_order_1=LinearRegression(fit_intercept=True)
model_y_position_order_1.fit(X_order_1[:testing_data_index, :], y_position_label[order_1_offset+time_lag:testing_data_index+order_1_offset+time_lag])
if time_lag==0:
    y_position_predict_order_1=model_y_position_order_1.predict( X_order_1[testing_data_index:] )
else:
    y_position_predict_order_1=model_y_position_order_1.predict( X_order_1[testing_data_index:-time_lag] )
print('* model_y_position_order_1 score: ', r2_score( y_position_label[testing_data_index+order_1_offset+time_lag:], y_position_predict_order_1 ))

model_y_position_order_2 = LinearRegression(fit_intercept=True)
model_y_position_order_2.fit( X_order_2[:testing_data_index,:], y_position_label[order_2_offset+time_lag:testing_data_index+order_2_offset+time_lag])
if time_lag==0:
    y_position_predict_order_2=model_y_position_order_2.predict(X_order_2[testing_data_index:])
else:
    y_position_predict_order_2=model_y_position_order_2.predict(X_order_2[testing_data_index:-time_lag])
print('* model_y_position_order_2 score: ', r2_score( y_position_label[testing_data_index+order_2_offset+time_lag:], y_position_predict_order_2 ))

print('\n')

# Z position model fit and score
model_z_position = LinearRegression(fit_intercept=True)
model_z_position.fit( X[:testing_data_index, :], z_position_label[time_lag:testing_data_index+time_lag ])
if time_lag==0:
    z_position_predict=model_z_position.predict( X[testing_data_index:] )
else:
    z_position_predict=model_z_position.predict( X[testing_data_index:-time_lag] )
print('* model_z_position score: ', r2_score( z_position_label[testing_data_index+time_lag:], z_position_predict))

print('\n')

# X velocity model fit and score
model_x_velocity = LinearRegression(fit_intercept=True)
model_x_velocity.fit( X[:testing_data_index, :], x_velocity_label[time_lag:testing_data_index+time_lag ] )
if time_lag==0:
    x_velocity_predict=model_x_velocity.predict( X[testing_data_index:] )
else:
    x_velocity_predict=model_x_velocity.predict( X[testing_data_index:-time_lag] )
print('* model_x_velocity score: ', r2_score(  x_velocity_label[testing_data_index+time_lag:], x_velocity_predict))

model_x_velocity_order_1=LinearRegression(fit_intercept=True)
model_x_velocity_order_1.fit(X_order_1[:testing_data_index,:], x_velocity_label[order_1_offset+time_lag:testing_data_index+order_1_offset+time_lag])
if time_lag==0:
    x_velocity_predict_order_1=model_x_velocity_order_1.predict(X_order_1[testing_data_index:])
else:
    x_velocity_predict_order_1=model_x_velocity_order_1.predict(X_order_1[testing_data_index:-time_lag])
print('* model_x_velocity_order_1 score: ', r2_score(  x_velocity_label[testing_data_index+order_1_offset+time_lag:], x_velocity_predict_order_1))

model_x_velocity_order_2=LinearRegression(fit_intercept=True)
model_x_velocity_order_2.fit( X_order_2[:testing_data_index,:], x_velocity_label[order_2_offset+time_lag:testing_data_index+order_2_offset+time_lag] )
if time_lag==0:
    x_velocity_predict_order_2=model_x_velocity_order_2.predict(X_order_2[testing_data_index:])
else:
    x_velocity_predict_order_2=model_x_velocity_order_2.predict(X_order_2[testing_data_index:-time_lag])
print('* model_x_velocity_order_2 score: ', r2_score( x_velocity_label[testing_data_index+order_2_offset+time_lag:], x_velocity_predict_order_2 ) )

print('\n')

# Y velocity model fit and score
model_y_velocity = LinearRegression(fit_intercept=True)
model_y_velocity.fit( X[:testing_data_index, :], y_velocity_label[time_lag:testing_data_index+time_lag ] )
if time_lag==0:
    y_velocity_predict=model_y_velocity.predict( X[testing_data_index:] )
else:
    y_velocity_predict=model_y_velocity.predict( X[testing_data_index:-time_lag] )
print('* model_y_velocity score: ', r2_score( y_velocity_label[testing_data_index+time_lag:], y_velocity_predict))

model_y_velocity_order_1=LinearRegression(fit_intercept=True)
model_y_velocity_order_1.fit(X_order_1[:testing_data_index,:], y_velocity_label[order_1_offset+time_lag:testing_data_index+order_1_offset+time_lag])
if time_lag==0:
    y_velocity_predict_order_1=model_y_velocity_order_1.predict(X_order_1[testing_data_index:])
else:
    y_velocity_predict_order_1=model_y_velocity_order_1.predict(X_order_1[testing_data_index:-time_lag])
print('* model_y_velocity_order_1 score: ', r2_score(  y_velocity_label[testing_data_index+order_1_offset+time_lag:], y_velocity_predict_order_1))

model_y_velocity_order_2=LinearRegression(fit_intercept=True)
model_y_velocity_order_2.fit( X_order_2[:testing_data_index,:], y_velocity_label[order_2_offset+time_lag:testing_data_index+order_2_offset+time_lag] )
if time_lag==0:
    y_velocity_predict_order_2=model_y_velocity_order_2.predict(X_order_2[testing_data_index:])
else:
    y_velocity_predict_order_2=model_y_velocity_order_2.predict(X_order_2[testing_data_index:-time_lag])
print('* model_y_velocity_order_2 score: ', r2_score( y_velocity_label[testing_data_index+order_2_offset+time_lag:], y_velocity_predict_order_2 ) )

print('\n')

# Z velocity model fit and score
model_z_velocity = LinearRegression(fit_intercept=True)
model_z_velocity.fit( X[:testing_data_index, :], z_velocity_label[time_lag:testing_data_index+time_lag ] )
if time_lag==0:
    z_velocity_predict=model_z_velocity.predict( X[testing_data_index:] )
else:
    z_velocity_predict=model_z_velocity.predict( X[testing_data_index:-time_lag] )
print('* model_z_velocity score: ',end='')
print( r2_score( z_velocity_label[testing_data_index+time_lag:],z_velocity_predict) )

print('\n')

# X acceleration model fit and score
model_x_acceleration = LinearRegression(fit_intercept=True)
model_x_acceleration.fit( X[:testing_data_index, :], x_acceleration_label[time_lag:testing_data_index+time_lag ] )
x_acceleration_predict=model_x_acceleration.predict( X[testing_data_index:-1-time_lag] )
print('* model_x_acceleration score: ', r2_score(x_acceleration_label[testing_data_index+time_lag:], x_acceleration_predict ))
model_x_acceleration_order_1 = LinearRegression(fit_intercept=True)
model_x_acceleration_order_1.fit( X_order_1[:testing_data_index, :], x_acceleration_label[order_1_offset+time_lag:testing_data_index+order_1_offset+time_lag] )
x_acceleration_predict_order_1=model_x_acceleration_order_1.predict( X_order_1[testing_data_index:-1-time_lag] )
print('* model_x_acceleration_order_1 score: ', r2_score(x_acceleration_label[testing_data_index+order_1_offset+time_lag:], x_acceleration_predict_order_1 ))
model_x_acceleration_order_2 = LinearRegression(fit_intercept=True)
model_x_acceleration_order_2.fit( X_order_2[:testing_data_index, :], x_acceleration_label[order_2_offset+time_lag:testing_data_index+order_2_offset+time_lag] )
x_acceleration_predict_order_2=model_x_acceleration_order_2.predict( X_order_2[testing_data_index:-1-time_lag] )
print('* model_x_acceleration_order_2 score: ', r2_score(x_acceleration_label[testing_data_index+order_2_offset+time_lag:], x_acceleration_predict_order_2 ))

print('\n')

# Y acceleration model fit and score
model_y_acceleration = LinearRegression(fit_intercept=True)
model_y_acceleration.fit( X[:testing_data_index, :], y_acceleration_label[time_lag:testing_data_index+time_lag ] )
y_acceleration_predict=model_y_acceleration.predict( X[testing_data_index:-1-time_lag] )
print('* model_y_acceleration score: ', r2_score( y_acceleration_label[testing_data_index+time_lag:], y_acceleration_predict))
model_y_acceleration_order_1 = LinearRegression(fit_intercept=True)
model_y_acceleration_order_1.fit( X_order_1[:testing_data_index, :], y_acceleration_label[order_1_offset+time_lag:testing_data_index+order_1_offset+time_lag] )
y_acceleration_predict_order_1=model_y_acceleration_order_1.predict( X_order_1[testing_data_index:-1-time_lag] )
print('* model_y_acceleration_order_1 score: ', r2_score(y_acceleration_label[testing_data_index+order_1_offset+time_lag:], y_acceleration_predict_order_1 ))
model_y_acceleration_order_2 = LinearRegression(fit_intercept=True)
model_y_acceleration_order_2.fit( X_order_2[:testing_data_index, :], y_acceleration_label[order_2_offset+time_lag:testing_data_index+order_2_offset+time_lag] )
y_acceleration_predict_order_2=model_y_acceleration_order_2.predict( X_order_2[testing_data_index:-1-time_lag] )
print('* model_y_acceleration_order_2 score: ', r2_score(y_acceleration_label[testing_data_index+order_2_offset+time_lag:], y_acceleration_predict_order_2 ))

print('\n')

# Z acceleration model fit and score
model_z_acceleration = LinearRegression(fit_intercept=True)
model_z_acceleration.fit( X[:testing_data_index, :], z_acceleration_label[time_lag:testing_data_index+time_lag ] )
z_acceleration_predict=model_z_acceleration.predict( X[testing_data_index:-1-time_lag] )
print('* model_z_acceleration score: ', r2_score( z_acceleration_label[testing_data_index+time_lag:], z_acceleration_predict))
print('\n')

print('There are '+str(units_have_value)+' units have value in this session')
print('\n')

print('<<<', 'Session(s): ', file_name_1,  '. Time Lag: ', time_lag, '. Feature numbers: ', feature_numbers, '. Include hash unit: ', include_hash_unit,'.  <<<')

print('\n')

print('How many weights in model_y_position: ', model_y_position.coef_.shape[0])
'''
for i in range(model_y_position.coef_.shape[0] ):
    print('W_'+str( f'{i+1:03}' )+ ' = ',end='')
    print( str(model_y_position.coef_[i]) )
'''
print('model_y_position intercept = ', model_y_position.intercept_)
print('\n')

print('shape of finger_x_velocity', finger_x_velocity.shape)
print('lenght of x_acceleration_label', len(x_acceleration_label))
print('\n')

tEnd=time.time()
print('Overall processing time: '+ str ( round(tEnd-tStart, 3) )+'seconds' )

'''
plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, x_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], x_position_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], x_position_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('position x prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('x coordinate')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('X_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], y_position_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], y_position_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('position y prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('y coordinate')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Y_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], z_position_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], z_position_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('position z prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('z coordinate')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Z_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], x_velocity_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], x_velocity_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('velocity x prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('x velocity')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Z_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], y_velocity_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], y_velocity_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('velocity y prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('y velocity')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Z_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], z_velocity_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], z_velocity_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('velocity z prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('z velocity')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Z_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], x_acceleration_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-3], x_acceleration_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('acceleration x prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('x acceleration')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Z_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], y_acceleration_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-3], y_acceleration_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('acceleration y prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('y acceleration')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Z_position_prediction.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], z_acceleration_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-3], z_acceleration_label[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('acceleration z prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('z acceleration')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('Z_position_prediction.png' )
'''
