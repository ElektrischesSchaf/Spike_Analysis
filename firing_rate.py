# -*- coding: utf-8 -*-
import numpy as np
import h5py
import time
import matplotlib.pyplot as plot 
import copy

from sklearn.linear_model import LinearRegression
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.feature_selection import RFE
from  sklearn.svm import SVC
from sklearn.svm import SVR

tStart=time.time()
testing_data_index=5000
classifier = svm.SVC(gamma=0.001)

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

with h5py.File('indy_20160407_02.mat', 'r') as mat_file:

    time_stamp=mat_file['t']
    spikes = mat_file['spikes']
    firing_rate_cell=[[]]
    firing_rate_final=[] # not[[]]

    numpy_finger_pos=mat_file.get('finger_pos')
    numpy_finger_pos=np.array(numpy_finger_pos)

    finger_x_coor=numpy_finger_pos[0][:]
    finger_y_coor=numpy_finger_pos[1][:]
    finger_z_coor=numpy_finger_pos[2][:]

    x_label=[]
    y_label=[]
    z_label=[]
    time_stamp_64ms=[]

    sampling_rate=16 # because 64ms

    #duration=1000
    duration=time_stamp.shape[1]
    sampling_index=0
    while sampling_index < duration:
        #print('sampling_index = ', sampling_index)
        print( 'Progress of making sampling array: '+ str(   round( (sampling_index / duration)*100, 3)   )+' %' )
        time_stamp_64ms.append(time_stamp[0][sampling_index])
        sampling_index+=16

    # make y label matrix first
    index_label=0
    while index_label < duration:
        x_label.append(finger_x_coor[index_label] )
        y_label.append(finger_y_coor[index_label] )
        z_label.append(finger_z_coor[index_label] )
        index_label+=sampling_rate
    print('Label appending finished')

    # plot each channel start
    for channel_index in range(96):
        print('Channel progress: ' + str( round( (channel_index/96)*100, 3) )+' %' ) # 96 channels in this dataset

        #channel_index=0

        temp_spike_cell_1=[]
        temp_spike_cell_2=[]
        temp_spike_cell_3=[]
        temp_spike_cell_4=[]
        temp_spike_cell_5=[]
        temp_spike_cell_6=[]        

        #plot_row = [[]]

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
        
        print('shape of temp_spike_cell_1: ',temp_spike_cell_1.shape)
        print('shape of temp_spike_cell_2: ',temp_spike_cell_2.shape)
        print('shape of temp_spike_cell_3: ',temp_spike_cell_3.shape)
        print('shape of time_stamp_64ms: ',time_stamp_64ms.shape)

        ''' # Disable cell 4, 5, 6
        temp_spike_cell_4=mat_file[ ( spikes[0][channel_index+96] ) ][()]
        temp_spike_cell_5=mat_file[ ( spikes[1][channel_index+96] ) ][()]
        temp_spike_cell_6=mat_file[ ( spikes[2][channel_index+96] ) ][()]
        '''
        
        if temp_spike_cell_1.shape[0] != 2:
            
            '''
            for a in range (temp_spike_cell_1.shape[1]):
                plot_row[-1].append( temp_spike_cell_1[0][a] )
            '''

            # firing rate
            yee=histc(temp_spike_cell_1, time_stamp_64ms)
            print('shape of yee:  ',yee.shape)
            firing_rate_cell.append(yee)
            print('yee: ',yee)
            #end firing rate

        print('length of firing_rate in cell 1: ',end='')
        print(len(firing_rate_cell[-1]))

        firing_rate_cell.append([])  

        if temp_spike_cell_2.shape[0] != 2:

            '''
            for i in range (temp_spike_cell_2.shape[1]):
                plot_row[-1].append( temp_spike_cell_2[0][i] )
            '''

            # firing rate
            yee=histc(temp_spike_cell_2, time_stamp_64ms)
            print('shape of yee:  ',yee.shape)
            firing_rate_cell.append(yee)            
            #end firing rate


        print('length of firing_rate in cell 2: ',end='')
        print(len(firing_rate_cell[-1]))
        firing_rate_cell.append([])

        if temp_spike_cell_3.shape[0] != 2:
            '''
            for i in range (temp_spike_cell_3.shape[1]):
                plot_row[-1].append( temp_spike_cell_3[0][i] )
            '''

            # firing rate
            yee=histc(temp_spike_cell_3, time_stamp_64ms)
            print('shape of yee:  ',yee.shape)
            firing_rate_cell.append(yee)
            #end firing rate

        print('length of firing_rate in cell 3: ',end='')
        print(len(firing_rate_cell[-1]))
        print('\n\n')
        firing_rate_cell.append([])


        ''' # Disable cell 4, 5, 6
        if temp_spike_cell_4.shape[0] != 2:
            for i in range (temp_spike_cell_4.shape[1]):
                plot_row[-1].append( temp_spike_cell_4[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell[-1]: ',end='')
        print(len(firing_rate_cell[-1]))        
        firing_rate_cell.append([])

        plot_row.append([])

        if temp_spike_cell_5.shape[0] != 2:
            for i in range (temp_spike_cell_5.shape[1]):
                plot_row[-1].append( temp_spike_cell_5[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell[-1]: ',end='')
        print(len(firing_rate_cell[-1]))        
        firing_rate_cell.append([])

        plot_row.append([])


        if temp_spike_cell_6.shape[0] != 2:
            for i in range (temp_spike_cell_6.shape[1]):
                plot_row[-1].append( temp_spike_cell_6[0][i] )
        else:
            plot_row[-1].append(0)
        '''


        print('row numbers of firing_rate_cell: ',end='')
        print( len( firing_rate_cell) )
        print('\n')

        for row_index in range( len( firing_rate_cell) ):            
            print('length of firing_rate_cell['+ str(row_index) +']: ',end='')
            print(len(firing_rate_cell[row_index]))
        print('\n')

        print('End of one channel '+ str(channel_index+1) +'\n') 


# Extract firing_rate_cell with rows have length bigger than zero
for row_index in range( len( firing_rate_cell) ):   
    if len(firing_rate_cell[row_index]):
        firing_rate_final.append( firing_rate_cell[row_index] )

for row_index in range( len( firing_rate_final) ):            
    print('length of firing_rate_final['+ str(row_index) +']: ',end='')
    print(len(firing_rate_final[row_index]))

print('\n')

firing_rate_matrix=np.array(firing_rate_final)
print('firing_rate_matrix shape: ',end='')
print(firing_rate_matrix.shape)
print('\n')


y_label=np.array(y_label)
y=y_label.astype(np.float64)
print('Label list shape: ',end='')
print( y.shape ) # y is the label matrix
print('\n')


firing_rate_matrix=np.transpose(firing_rate_matrix)
print('transposed firing_rate_matrix shape: ',end='')
print(firing_rate_matrix.shape)
print('\n')

X=firing_rate_matrix.astype(np.float64)
print('fetures list shape: ',end='')
print( X.shape ) # X is the feature matrix
print('\n')

model = LinearRegression(fit_intercept=True)
model.fit( X[:testing_data_index, :], y[:testing_data_index ])

print('how many weights: ', model.coef_.shape)
for i in range(model.coef_.shape[0] ):
    print('W_'+str( f'{i+1:03}' )+ ' = ',end='')
    print( str(model.coef_[i]) )

print('Model intercept = ', model.intercept_)

y_predict=model.predict( X[testing_data_index:-1] )
print('shape of y_predict: ', y_predict.shape)

print('score: ',end='')
print( model.score(X[testing_data_index:-1],y[testing_data_index:-1]) )

tEnd=time.time()
print('Overall processing time: '+ str ( round(tEnd-tStart, 3) )+'seconds' )

plot.figure(figsize=(15,5))
#plot.scatter(time_stamp_64ms, y_predict, s=1)
plot.plot(time_stamp_64ms[testing_data_index:-1], y_predict, 'b--',label='Prediction' )
plot.plot(time_stamp_64ms[testing_data_index:-1], y[testing_data_index:-1], 'r--', label='True value')
plot.legend(loc='upper right')
plot.title('position y prediction and ground truth')
plot.xlabel('time (second)')
plot.ylabel('y coordinate')
axes = plot.gca()
#axes.set_xlim([60, 890])
plot.show()
#plot.savefig('X_axis_velocity.png' )




