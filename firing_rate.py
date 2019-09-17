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
file_name='indy_20160407_02.mat'
tStart=time.time()
#testing_data_index=5000
testing_data_index=10222
def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r
not_empty=0

with h5py.File(file_name, 'r') as mat_file:

    time_stamp=mat_file['t']  
    # or
    # time_stamp=mat_file.get('t')
    # time_stamp=np.array(time_stamp)
    # time_stamp.shape = (1, 204446)
    spikes = mat_file['spikes']
    firing_rate_cell=[[]]
    firing_rate_final=[] # not[[]]

    numpy_finger_pos=mat_file.get('finger_pos')
    numpy_finger_pos=np.array(numpy_finger_pos)

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
    ''' # Too slow app. 70 seconds
    while sampling_index < duration:
        #print('sampling_index = ', sampling_index)
        print( 'Progress of making sampling array: '+ str(   round( (sampling_index / duration)*100, 3)   )+' %' )
        time_stamp_64ms.append(time_stamp[0][sampling_index])
        sampling_index+=sampling_rate
    '''
    time_stamp_64ms=time_stamp[0][::sampling_rate]  # way faster, app. 4 seconds
    print('lenght of time_stamp_64ms: ', len(time_stamp_64ms))

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
    x_position_label=x_position_label[:-1]
    y_position_label=finger_y_coor[::sampling_rate]
    y_position_label=y_position_label[:-1]
    z_position_label=finger_z_coor[::sampling_rate]
    z_position_label=z_position_label[:-1]
    print('Position label arrays finished')

    # Making spike counts matrix
    for channel_index in range(96):
        print('Channel progress: ' + str( round( (channel_index/96)*100, 3) )+' %' ) # 96 channels in this dataset
        
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
        print('shape of temp_spike_cell_1: ',temp_spike_cell_1.shape)
        print('shape of temp_spike_cell_2: ',temp_spike_cell_2.shape)
        print('shape of temp_spike_cell_3: ',temp_spike_cell_3.shape)
        print('shape of time_stamp_64ms: ',time_stamp_64ms.shape)
        '''
       
        if temp_spike_cell_1.shape[0] != 2:

            # firing rate
            yee=histc(temp_spike_cell_1, time_stamp_64ms)
            #print('shape of yee:  ',yee.shape)
            firing_rate_cell.append(yee[:-1])
            #print('yee: ',yee)
            #end firing rate

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
            #end firing rate

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
            #end firing rate
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

# Extract firing_rate_cell with rows have length bigger than zero
for row_index in range( len( firing_rate_cell) ):   
    if len(firing_rate_cell[row_index]):
        firing_rate_final.append( firing_rate_cell[row_index] )
        not_empty+=1

'''
for row_index in range( len( firing_rate_final) ):            
    print('length of firing_rate_final['+ str(row_index) +']: ',end='')
    print(len(firing_rate_final[row_index]))
'''

print('\n')

firing_rate_matrix=np.array(firing_rate_final)
print('firing_rate_matrix shape: ',end='')
print(firing_rate_matrix.shape)
print('\n')


x_position_label=np.array(x_position_label)
x_position_label=x_position_label.astype(np.float64)
print('position x_position_label  list shape: ',end='')
print( x_position_label.shape ) # x is the label array should be feed into the model
print('\n')

y_position_label=np.array(y_position_label)
y_position_label=y_position_label.astype(np.float64)
print('position y_position_label list shape: ',end='')
print( y_position_label.shape ) # y is the label array should be feed into the model
print('\n')

z_position_label=np.array(z_position_label)
z_position_label=z_position_label.astype(np.float64)
print('position z_position_label list shape: ',end='')
print( z_position_label.shape ) # y is the label array should be feed into the model
print('\n')

firing_rate_matrix=np.transpose(firing_rate_matrix)
print('transposed firing_rate_matrix shape: ',end='')
print(firing_rate_matrix.shape)
print('\n')

X=firing_rate_matrix.astype(np.float64)
print('fetures list shape: ',end='')
print( X.shape ) # X is the feature matrix
print('\n')

model_x_position = LinearRegression(fit_intercept=True)
model_x_position.fit( X[:testing_data_index, :], x_position_label[:testing_data_index ] )

model_y_position = LinearRegression(fit_intercept=True)
model_y_position.fit( X[:testing_data_index, :], y_position_label[:testing_data_index ])

model_z_position = LinearRegression(fit_intercept=True)
model_z_position.fit( X[:testing_data_index, :], z_position_label[:testing_data_index ])
print('passed model fit')


print('how many weights in model_y_position: ', model_y_position.coef_.shape)
'''
for i in range(model_y_position.coef_.shape[0] ):
    print('W_'+str( f'{i+1:03}' )+ ' = ',end='')
    print( str(model_y_position.coef_[i]) )
'''

print('model_y_position intercept = ', model_y_position.intercept_)

x_position_predict=model_x_position.predict( X[testing_data_index:] )
print('shape of x_position_predict: ', x_position_predict.shape)

y_position_predict=model_y_position.predict( X[testing_data_index:] )
print('shape of y_position_predict: ', y_position_predict.shape)

z_position_predict=model_y_position.predict( X[testing_data_index:] )
print('shape of z_position_predict: ', z_position_predict.shape)

# Calculating velocity and acceleration below
with h5py.File(file_name, 'r') as mat_file:

    finger_pos = mat_file['finger_pos']
    time_stamp=mat_file['t']

    numpy_finger_pos=mat_file.get('finger_pos')
    numpy_finger_pos=np.array(numpy_finger_pos)
    numpy_time_stamp=mat_file.get('t')
    numpy_time_stamp=np.array(numpy_time_stamp)

    print('numpy_finger_pos shape: ',end='')
    print(numpy_finger_pos.shape) #  (3, 204446)
    print('numpy_time_stamp: ',end='')
    print(numpy_time_stamp.shape)  #  (1, 204446)

    time_stamp_64ms=time_stamp[0][::sampling_rate]

    finger_z_pos_64ms=numpy_finger_pos[0][::sampling_rate]
    finger_x_pos_64ms=numpy_finger_pos[1][::sampling_rate]
    finger_y_pos_64ms=numpy_finger_pos[2][::sampling_rate]
    print('shape of finger_x_pos_64ms', finger_x_pos_64ms.shape) # (12778,)

    finger_x_velocity=[]
    finger_y_velocity=[]
    finger_z_velocity=[]
    velocity_time_coor=[]

    finger_x_acceleration=[]
    finger_y_acceleration=[]
    finger_z_acceleration=[]
    acceleration_time_coor=[]

    #duration=204445
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

print('shape of finger_x_velocity', finger_x_velocity.shape)
print('lenght of x_acceleration_label', len(x_acceleration_label))

print('model_x_position score: ',end='')
print( r2_score( x_position_label[testing_data_index:], x_position_predict) )
print('model_y_position score: ',end='')
print( r2_score( y_position_label[testing_data_index:], y_position_predict) )
print('model_z_position score: ',end='')
print( r2_score( z_position_label[testing_data_index:], z_position_predict) )
print('\n')

model_x_velocity = LinearRegression(fit_intercept=True)
model_x_velocity.fit( X[:testing_data_index, :], x_velocity_label[:testing_data_index ] )
x_velocity_predict=model_x_velocity.predict( X[testing_data_index:-1] )
print('model_x_velocity score: ',end='')
print( r2_score(  x_velocity_label[testing_data_index:-1], x_velocity_predict) )


model_y_velocity = LinearRegression(fit_intercept=True)
model_y_velocity.fit( X[:testing_data_index, :], y_velocity_label[:testing_data_index ] )
y_velocity_predict=model_y_velocity.predict( X[testing_data_index:-1] )
print('model_y_velocity score: ',end='')
print( r2_score( y_velocity_label[testing_data_index:-1], y_velocity_predict) )

model_z_velocity = LinearRegression(fit_intercept=True)
model_z_velocity.fit( X[:testing_data_index, :], z_velocity_label[:testing_data_index ] )
z_velocity_predict=model_z_velocity.predict( X[testing_data_index:-1] )
print('model_z_velocity score: ',end='')
print( r2_score( z_velocity_label[testing_data_index:-1],z_velocity_predict) )
print('\n')

model_x_acceleration = LinearRegression(fit_intercept=True)
model_x_acceleration.fit( X[:testing_data_index, :], x_acceleration_label[:testing_data_index ] )
x_acceleration_predict=model_x_acceleration.predict( X[testing_data_index:-1] )
print('model_x_acceleration score: ',end='')
print( r2_score(x_acceleration_label[testing_data_index:], x_acceleration_predict ) )

model_y_acceleration = LinearRegression(fit_intercept=True)
model_y_acceleration.fit( X[:testing_data_index, :], y_acceleration_label[:testing_data_index ] )
y_acceleration_predict=model_y_acceleration.predict( X[testing_data_index:-1] )
print('model_y_acceleration score: ',end='')
print( r2_score( y_acceleration_label[testing_data_index:], y_acceleration_predict) )

model_z_acceleration = LinearRegression(fit_intercept=True)
model_z_acceleration.fit( X[:testing_data_index, :], z_acceleration_label[:testing_data_index ] )
z_acceleration_predict=model_z_acceleration.predict( X[testing_data_index:-1] )
print('model_z_acceleration score: ',end='')
print( r2_score( z_acceleration_label[testing_data_index:], z_acceleration_predict) )
print('\n')

print('There are '+str(not_empty)+' units used in this model')
print('\n')

print('how many weights in model_z_position: ', model_z_position.coef_.shape)
#for i in range(model_z_position.coef_.shape[0] ):
for i in range(10 ):
    print('W_'+str( f'{i+1:03}' )+ ' = ',end='')
    print( str(model_z_position.coef_[i]) )

print('some value from x_acceleration_label ')
for i in range( 10 ):
    print('ACC_'+str( f'{i+1:03}' )+ ' = ',end='')
    print( str(x_acceleration_label[i]) )



tEnd=time.time()

print('Overall processing time: '+ str ( round(tEnd-tStart, 3) )+'seconds' )

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