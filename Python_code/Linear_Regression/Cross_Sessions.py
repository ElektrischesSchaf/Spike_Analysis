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

# My module
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file_with_lag_and_order as load_mat_file

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
time_stamp_64ms=[]

###################################### Auto-assigned parameters
#testing_data_index=5000
#testing_data_index=10222
testing_data_index=0 # Should be 10222 in indy_20160407_02
channel_number=0
units_have_value=0 # unit numbers that is not empty


###################################### Parameters should be assigned
the_sampling_rate=my_parameters.the_sampling_rate
file_numbers = 1
time_lag=my_parameters.time_lag
order = 5
with_sorted_spikes= True
include_hash_unit=my_parameters.include_hash_unit

# Must know these two numbers beforehand
channel_numbers_in_this_dataset=96
units_numbers_in_this_dataset=3

if with_sorted_spikes==True:
    feature_numbers=channel_numbers_in_this_dataset*units_numbers_in_this_dataset
else:
    feature_numbers=channel_numbers_in_this_dataset

# Create empty X and y matrices before actural read the data in the loop
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

# All models fit and predict, show R2 score
print('In time lag: ', time_lag, '\n')

if order_index >=2:

    model_x_position = LinearRegression(fit_intercept=True)
    model_x_position.fit( X_for_training, x_position_label_training )
    if time_lag==0:
        x_position_predict=model_x_position.predict( X_for_prediction )
    else:
        x_position_predict=model_x_position.predict( X_for_prediction_with_time_lag )
    print('* model_x_position score in order ', order_index, ': ', r2_score( x_position_label_testing, x_position_predict ))

    model_y_position = LinearRegression(fit_intercept=True)
    model_y_position.fit( X_for_training, y_position_label_training )
    if time_lag==0:
        y_position_predict=model_y_position.predict( X_for_prediction )
    else:
        y_position_predict=model_y_position.predict( X_for_prediction_with_time_lag )
    print('* model_y_position score in order ', order_index, ': ', r2_score( y_position_label_testing, y_position_predict ))

    model_z_position = LinearRegression(fit_intercept=True)
    model_z_position.fit( X_for_training, z_position_label_training )
    if time_lag==0:
        z_position_predict=model_z_position.predict( X_for_prediction )
    else:
        z_position_predict=model_z_position.predict( X_for_prediction_with_time_lag )
    print('* model_z_position score in order ', order_index, ': ', r2_score( z_position_label_testing, z_position_predict ))

    print('\n')

    model_x_velocity=LinearRegression(fit_intercept=True)
    model_x_velocity.fit( X_for_training, x_velocity_label_training )
    if time_lag==0:
        x_velocity_predict=model_x_velocity.predict(X_for_prediction)
    else:
        x_velocity_predict=model_x_velocity.predict(X_for_prediction_with_time_lag)
    print('* model_x_velocity score in order ', order_index, ': ', r2_score( x_velocity_label_testing, x_velocity_predict ) )

    model_y_velocity=LinearRegression(fit_intercept=True)
    model_y_velocity.fit( X_for_training, y_velocity_label_training )
    if time_lag==0:
        y_velocity_predict=model_y_velocity.predict(X_for_prediction)
    else:
        y_velocity_predict=model_y_velocity.predict(X_for_prediction_with_time_lag)
    print('* model_y_velocity score in order ', order_index, ': ', r2_score( y_velocity_label_testing, y_velocity_predict ) )

    model_z_velocity=LinearRegression(fit_intercept=True)
    model_z_velocity.fit( X_for_training, z_velocity_label_training )
    if time_lag==0:
        z_velocity_predict=model_z_velocity.predict(X_for_prediction)
    else:
        z_velocity_predict=model_z_velocity.predict( X_for_prediction_with_time_lag )
    print('* model_z_velocity score in order ', order_index, ': ', r2_score( z_velocity_label_testing, z_velocity_predict ) )

    print('\n')

    model_x_acceleration = LinearRegression(fit_intercept=True)
    model_x_acceleration.fit( X_for_training, x_acceleration_label_training   )
    x_acceleration_predict=model_x_acceleration.predict( X_for_prediction_with_time_lag_2  )
    print('* model_x_acceleration score in order ', order_index, ': ', r2_score(x_acceleration_label_testing, x_acceleration_predict ))

    model_y_acceleration = LinearRegression(fit_intercept=True)
    model_y_acceleration.fit( X_for_training, y_acceleration_label_training )
    y_acceleration_predict=model_y_acceleration.predict( X_for_prediction_with_time_lag_2 )
    print('* model_y_acceleration score in order ', order_index, ': ', r2_score(y_acceleration_label_testing, y_acceleration_predict ))

    model_z_acceleration = LinearRegression(fit_intercept=True)
    model_z_acceleration.fit( X_for_training, z_acceleration_label_training )
    z_acceleration_predict=model_z_acceleration.predict(  X_for_prediction_with_time_lag_2)
    print('* model_z_acceleration score in order ', order_index, ': ', r2_score(z_acceleration_label_testing, z_acceleration_predict ))


if order_index==1:

    model_x_position = LinearRegression(fit_intercept=True)
    model_x_position.fit( X_for_training, x_position_label_training )
    if time_lag==0:
        x_position_predict=model_x_position.predict( X_for_prediction )
    else:
        x_position_predict=model_x_position.predict( X_for_prediction_with_time_lag )
    print('* model_x_position score in order ', order_index, ': ', r2_score( x_position_label_testing, x_position_predict))

    model_y_position = LinearRegression(fit_intercept=True)
    model_y_position.fit( X_for_training, y_position_label_training )
    if time_lag==0:
        y_position_predict=model_y_position.predict( X_for_prediction )
    else:
        y_position_predict=model_y_position.predict( X_for_prediction_with_time_lag )
    print('* model_y_position score in order ', order_index, ': ', r2_score(  y_position_label_testing, y_position_predict ))

    model_z_position = LinearRegression(fit_intercept=True)
    model_z_position.fit( X_for_training, z_position_label_training )
    if time_lag==0:
        z_position_predict=model_z_position.predict( X_for_prediction )
    else:
        z_position_predict=model_z_position.predict( X_for_prediction_with_time_lag )
    print('* model_z_position score in order ', order_index, ': ', r2_score(  z_position_label_testing, z_position_predict ))
    
    print('\n')

    model_x_velocity=LinearRegression(fit_intercept=True)
    model_x_velocity.fit( X_for_training, x_velocity_label_training )
    if time_lag==0:
        x_velocity_predict=model_x_velocity.predict(X_for_prediction)
    else:
        x_velocity_predict=model_x_velocity.predict(X_for_prediction_with_time_lag)
    print('* model_x_velocity score in order ', order_index, ': ', r2_score( x_velocity_label_testing , x_velocity_predict))

    model_y_velocity=LinearRegression(fit_intercept=True)
    model_y_velocity.fit( X_for_training, y_velocity_label_training )
    if time_lag==0:
        y_velocity_predict=model_y_velocity.predict(X_for_prediction)
    else:
        y_velocity_predict=model_y_velocity.predict(X_for_prediction_with_time_lag)
    print('* model_y_velocity score in order ', order_index, ': ', r2_score( y_velocity_label_testing, y_velocity_predict))

    model_z_velocity=LinearRegression(fit_intercept=True)
    model_z_velocity.fit( X_for_training, z_velocity_label_training)
    if time_lag==0:
        z_velocity_predict=model_z_velocity.predict(X_for_prediction)
    else:
        z_velocity_predict=model_z_velocity.predict(X_for_prediction_with_time_lag)
    print('* model_z_velocity score in order ', order_index, ': ', r2_score(  z_velocity_label_testing, z_velocity_predict))

    print('\n')

    model_x_acceleration = LinearRegression(fit_intercept=True)
    model_x_acceleration.fit( X_for_training, x_acceleration_label_training )
    x_acceleration_predict=model_x_acceleration.predict(X_for_prediction_with_time_lag_2  )
    print('* model_x_acceleration score in order ', order_index, ': ', r2_score( x_acceleration_label_testing, x_acceleration_predict ))

    model_y_acceleration = LinearRegression(fit_intercept=True)
    model_y_acceleration.fit( X_for_training, y_acceleration_label_training  )
    y_acceleration_predict=model_y_acceleration.predict( X_for_prediction_with_time_lag_2 )
    print('* model_y_acceleration score in order ', order_index, ': ', r2_score( y_acceleration_label_testing, y_acceleration_predict ))

    model_z_acceleration = LinearRegression(fit_intercept=True)
    model_z_acceleration.fit( X_for_training, z_acceleration_label_training  )
    z_acceleration_predict=model_z_acceleration.predict( X_for_prediction_with_time_lag_2 )
    print('* model_z_acceleration score in order ', order_index, ': ', r2_score( z_acceleration_label_testing, z_acceleration_predict ))


if order_index==0:

    model_x_position = LinearRegression(fit_intercept=True)
    model_x_position.fit( X_for_training, x_position_label_training )
    if time_lag==0:
        x_position_predict=model_x_position.predict( X_for_prediction )
    else:
        x_position_predict=model_x_position.predict( X_for_prediction_with_time_lag )
    print('* model_x_position score in order ', order_index, ': ', r2_score( x_position_label_testing, x_position_predict))

    model_y_position = LinearRegression(fit_intercept=True)
    model_y_position.fit( X_for_training, y_position_label_training )
    if time_lag==0:
        y_position_predict=model_y_position.predict( X_for_prediction )
    else:
        y_position_predict=model_y_position.predict( X_for_prediction_with_time_lag )
    print('* model_y_position score in order ', order_index, ': ', r2_score( y_position_label_testing, y_position_predict))

    model_z_position = LinearRegression(fit_intercept=True)
    model_z_position.fit( X_for_training, z_position_label_training)
    if time_lag==0:
        z_position_predict=model_z_position.predict( X_for_prediction )
    else:
        z_position_predict=model_z_position.predict( X_for_prediction_with_time_lag )
    print('* model_z_position score in order ', order_index, ': ', r2_score( z_position_label_testing, z_position_predict))

    print('\n')

    model_x_velocity = LinearRegression(fit_intercept=True)
    model_x_velocity.fit( X_for_training, x_velocity_label_training  )
    if time_lag==0:
        x_velocity_predict=model_x_velocity.predict( X_for_prediction )
    else:
        x_velocity_predict=model_x_velocity.predict( X_for_prediction_with_time_lag )
    print('* model_x_velocity score in order ', order_index, ': ', r2_score(  x_velocity_label_testing, x_velocity_predict))

    model_y_velocity = LinearRegression(fit_intercept=True)
    model_y_velocity.fit( X_for_training, y_velocity_label_training)
    if time_lag==0:
        y_velocity_predict=model_y_velocity.predict( X_for_prediction )
    else:
        y_velocity_predict=model_y_velocity.predict( X_for_prediction_with_time_lag )
    print('* model_y_velocity score in order ', order_index, ': ', r2_score( y_velocity_label_testing, y_velocity_predict))

    model_z_velocity = LinearRegression(fit_intercept=True)
    model_z_velocity.fit( X_for_training, z_velocity_label_training )
    if time_lag==0:
        z_velocity_predict=model_z_velocity.predict( X_for_prediction )
    else:
        z_velocity_predict=model_z_velocity.predict( X_for_prediction_with_time_lag )
    print('* model_z_velocity score in order ', order_index, ': ', r2_score( z_velocity_label_testing , z_velocity_predict))

    print('\n')

    model_x_acceleration = LinearRegression(fit_intercept=True)
    model_x_acceleration.fit( X_for_training,  x_acceleration_label_training)
    x_acceleration_predict=model_x_acceleration.predict( X_for_prediction_with_time_lag_2 )
    print('* model_x_acceleration score in order ', order_index, ': ', r2_score( x_acceleration_label_testing, x_acceleration_predict ))

    model_y_acceleration = LinearRegression(fit_intercept=True)
    model_y_acceleration.fit( X_for_training, y_acceleration_label_training )
    y_acceleration_predict=model_y_acceleration.predict( X_for_prediction_with_time_lag_2 )
    print('* model_y_acceleration score in order ', order_index, ': ', r2_score( y_acceleration_label_testing, y_acceleration_predict ))

    model_z_acceleration = LinearRegression(fit_intercept=True)
    model_z_acceleration.fit( X_for_training, z_acceleration_label_training )
    z_acceleration_predict=model_z_acceleration.predict( X_for_prediction_with_time_lag_2 )
    print('* model_z_acceleration score in order ', order_index, ': ', r2_score( z_acceleration_label_testing, z_acceleration_predict ))


print('There are '+str(units_have_value)+' units have value in this session')
print('\n')
print('z_acceleration_label_training.shape= ', z_acceleration_label_training.shape)
print('X_for_training shape= ', X_for_training.shape)
print('X_for_prediction= ', X_for_prediction.shape)
print('How many weights in model_y_position: ', model_y_position.coef_.shape[0])

'''
for i in range(model_y_position.coef_.shape[0] ):
    print('W_'+str( f'{i+1:03}' )+ ' = ',end='')
    print( str(model_y_position.coef_[i]) )
'''
print('model_y_position intercept = ', model_y_position.intercept_)
print('\n')

tEnd=time.time()
print('Overall processing time: '+ str ( round(tEnd-tStart, 3) )+'seconds' )


plot.figure(figsize=(32,9))
#plot.scatter(time_stamp_64ms, x_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1-order], x_position_predict, 'b',linewidth=5 ,label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], x_position_label[testing_data_index:-1],'r', linewidth=5, label='Actual', alpha=0.7)
plot.legend(loc='upper right', fontsize=30)
plot.title('')
plot.xlabel('time (second)', fontsize=25)
plot.ylabel('x-position (mm)', fontsize=25)
plot.xticks(fontsize=25, color="black")
plot.yticks(fontsize=25, color="black")
axes = plot.gca()
axes.set_xlim([740, 760])
#plot.show()
plot.tight_layout()
plot.savefig('X_position_prediction.png' )

plot.cla()
plot.clf()


plot.figure(figsize=(32,9))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1-order], y_position_predict, 'b', linewidth=5,label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], y_position_label[testing_data_index:-1],'r', linewidth=5, label='Actual', alpha=0.7)
plot.legend(loc='upper right', fontsize=30)
plot.title('')
plot.xlabel('time (second)', fontsize=25)
plot.ylabel('y position (mm)', fontsize=25)
plot.xticks(fontsize=25, color="black")
plot.yticks(fontsize=25, color="black")
axes = plot.gca()
axes.set_xlim([740, 760])
#plot.show()
plot.tight_layout()
plot.savefig('Y_position_prediction.png' )

plot.cla()
plot.clf()

'''

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], z_position_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], z_position_label[testing_data_index:-1], 'r--', label='Actual')
plot.legend(loc='upper right')
plot.title('Linear Regression position z prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('z coordinate')
axes = plot.gca()
axes.set_xlim([740, 750])
#plot.show()
plot.savefig('Z_position_prediction.png' )

plot.cla()
plot.clf()


plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], x_velocity_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], x_velocity_label[testing_data_index:-1], 'r--', label='Actual')
plot.legend(loc='upper right')
plot.title('Linear Regression velocity x prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('x velocity')
axes = plot.gca()
axes.set_xlim([740, 750])
#plot.show()
plot.savefig('x_velocity_predict.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], y_velocity_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], y_velocity_label[testing_data_index:-1], 'r--', label='Actual')
plot.legend(loc='upper right')
plot.title('Linear Regression velocity y prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('y velocity')
axes = plot.gca()
axes.set_xlim([740, 750])
#plot.show()
plot.savefig('y_velocity_predict.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], z_velocity_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-2], z_velocity_label[testing_data_index:-1], 'r--', label='Actual')
plot.legend(loc='upper right')
plot.title('Linear Regression velocity z prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('z velocity')
axes = plot.gca()
axes.set_xlim([740, 750])
#plot.show()
plot.savefig('z_velocity_predict.png' )

plot.cla()
plot.clf()



plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], x_acceleration_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-3], x_acceleration_label[testing_data_index:-1], 'r--', label='Actual')
plot.legend(loc='upper right')
plot.title('Linear Regression acceleration x prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('x acceleration')
axes = plot.gca()
axes.set_xlim([740, 750])
#plot.show()
plot.savefig('x_acceleration_predict.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], y_acceleration_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-3], y_acceleration_label[testing_data_index:-1], 'r--', label='Actual')
plot.legend(loc='upper right')
plot.title('Linear Regression acceleration y prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('y acceleration')
axes = plot.gca()
axes.set_xlim([740, 750])
#plot.show()
plot.savefig('y_acceleration_predict.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_position_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-2], z_acceleration_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-3], z_acceleration_label[testing_data_index:-1], 'r--', label='Actual')
plot.legend(loc='upper right')
plot.title('Linear Regression acceleration z prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('z acceleration')
axes = plot.gca()
axes.set_xlim([740, 750])
#plot.show()
plot.savefig('z_acceleration_predict.png' )

'''