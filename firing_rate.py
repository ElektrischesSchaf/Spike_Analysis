# -*- coding: utf-8 -*-
import numpy as np
import h5py

import matplotlib.pyplot as plot 
import copy

from sklearn.linear_model import LinearRegression
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.feature_selection import RFE
from  sklearn.svm import SVC
from sklearn.svm import SVR

classifier = svm.SVC(gamma=0.001)

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

    sampling_rate=16 # because 64ms

    duration=1000
    #duration=time_stamp.shape[1]

    # make y label matrix first
    index_label=0
    while index_label < duration:
        x_label.append(finger_x_coor[index_label] )
        y_label.append(finger_y_coor[index_label] )
        z_label.append(finger_z_coor[index_label] )
        index_label+=sampling_rate
    print('Label appending finished')

    # plot each channel start
    for channel_index in range(5):
        print('channel progress: ' + str( (channel_index/96)*100 )+'%' ) # 96 channels in this dataset

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
            i=0    #i is the index for time_stemp
            index=0
            k=0    #k is the index for spikes
            while i<duration :
                '''
                print('i= ',end='')      
                print(i)
                print('\n')

                print('index_1=',end='')
                print(index)
                print('\n')

                print('k=',end='')
                print(k)
                print('\n')

                print('time target: ',end='')
                print(time_stamp[0][i])
                print('\n')

                print('length of firing_rate_cell[-1]: ',end='')
                print(len(firing_rate_cell[-1]))
                print('\n')
                '''
                if time_stamp[0][i] < temp_spike_cell_1[0][k] and time_stamp[0][i] > temp_spike_cell_1[0][k-1] :
                    firing_rate_cell[-1].append(k-index)
                    index=k
                    k=k-1

                    i+=sampling_rate
                    
                else:
                    
                    k=k+1
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
            i=0    #i is the index for time_stemp
            index=0
            k=0    #k is the index for spikes
            while i<duration :
                '''
                print('i= ',end='')      
                print(i)
                print('\n')

                print('index_1=',end='')
                print(index)
                print('\n')

                print('k=',end='')
                print(k)
                print('\n')

                print('time target: ',end='')
                print(time_stamp[0][i])
                print('\n')

                print('length of firing_rate_cell[-1]: ',end='')
                print(len(firing_rate_cell[-1]))
                print('\n')
                '''

                if time_stamp[0][i] < temp_spike_cell_2[0][k] and time_stamp[0][i] > temp_spike_cell_2[0][k-1] :
                    firing_rate_cell[-1].append(k-index)
                    index=k
                    k=k-1

                    i+=sampling_rate
                    
                else:
                    
                    k=k+1
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
            i=0    #i is the index for time_stemp
            index=0
            k=0    #k is the index for spikes
            while i<duration :
                '''
                print('i= ',end='')      
                print(i)
                print('\n')

                print('index_1=',end='')
                print(index)
                print('\n')

                print('k=',end='')
                print(k)
                print('\n')

                print('time target: ',end='')
                print(time_stamp[0][i])
                print('\n')

                print('length of firing_rate_cell[-1]: ',end='')
                print(len(firing_rate_cell[-1]))
                print('\n')
                '''

                if time_stamp[0][i] < temp_spike_cell_3[0][k] and time_stamp[0][i] > temp_spike_cell_3[0][k-1] :
                    firing_rate_cell[-1].append(k-index)
                    index=k
                    k=k-1
                    i+=sampling_rate
                    
                else:
                    
                    k=k+1
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
model.fit( X, y)

print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

'''
estimator=SVR(kernel="linear")
selector = RFE(estimator, 10, step=200)
selector = selector.fit(X[:37],y[:37])

print('\n---------------------------------\n')
print('selector.support: ',end='')
print(selector.support_)
print('\n---------------------------------\n')
print('length of selector.support: ',end='')
print(len(selector.support_))
print('\n---------------------------------\n')
new_X=X[:,selector.support_]
print('new_X.shape:  ',end='')
print(new_X.shape)
print('\n---------------------------------\n')
classifier = svm.SVC(gamma=0.001,kernel='rbf')
classifier.fit(new_X[:37],y[:37])
print('new_X[:37] shape:  ',end='')
print(new_X[:37].shape)
print('\n---------------------------------\n')
print('y[:37] shape:  ',end='')
print(y[:37].shape)
print('\n---------------------------------\n')

expected_training_data=y[:37]
predicted_training_data=classifier.predict(new_X[:37])

expected=y[37:]
predicted=classifier.predict(new_X[37:])
print('='*50)
print('type of metrics.classification_report(expected, predicted):',end='')
print(type(metrics.classification_report(expected, predicted) )  )
print('\n')
'''