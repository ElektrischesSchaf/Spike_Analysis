import numpy as np
import h5py
import time
import numpy
import matplotlib.pyplot as plot 
import copy
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

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
x_position_label=x_position_label.astype(np.float32)
print('position x_position_label  list shape: ',end='')
print( x_position_label.shape ) # x is the label array should be feed into the model
print('\n')

y_position_label=np.array(y_position_label)
y_position_label=y_position_label.astype(np.float32)
print('position y_position_label list shape: ',end='')
print( y_position_label.shape ) # y is the label array should be feed into the model
print('\n')

z_position_label=np.array(z_position_label)
z_position_label=z_position_label.astype(np.float32)
print('position z_position_label list shape: ',end='')
print( z_position_label.shape ) # y is the label array should be feed into the model
print('\n')

firing_rate_matrix=np.transpose(firing_rate_matrix)
print('transposed firing_rate_matrix shape: ',end='')
print(firing_rate_matrix.shape)
print('\n')

X=firing_rate_matrix.astype(np.float32)
print('fetures list shape: ',end='')
print( X.shape ) # X is the feature matrix
print('\n')


# Start Tensorflow Linear Regression
''' Example
x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))
'''
def R_squared(y, y_pred):
    '''
    R_squared computes the coefficient of determination.
    It is a measure of how well the observed outcomes are replicated by the model.
    '''
    #total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    #residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))    
    #r2 = tf.subtract(1.0, tf.divide(residual, total))
    
    total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    R_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    
    return R_squared
    
#x_data=tf.convert_to_tensor(X)
x_data=X

''' Example
# y = mx + b + noise_levels
b = 5


y_true =  (0.5 * x_data ) + 5 + noise
'''
#y_true=tf.convert_to_tensor(y)
y_true=y_position_label

'''
# Random 10 points to grab
batch_size = 8
'''

'''
m = tf.Variable(0.5)
b = tf.Variable(1.0)
'''
print('x_data.shape[0]: ')
print(x_data.shape[0])

m=tf.Variable(tf.zeros([ x_data.shape[1] , 1], tf.float32) ) 

#tf.Variable(tf.convert_to_tensor(np.eye(784), dtype=tf.float32)) 
#b=tf.Variable(tf.zeros([ y_true.shape[0] , 1], tf.float32) )
b=tf.Variable(tf.zeros([ testing_data_index , 1], tf.float32) )

#xph = tf.placeholder(tf.float32,[ x_data.shape[0],x_data.shape[1] ]) # not [ x_data.shape[0], 1 ]
#yph = tf.placeholder(tf.float32,[ y_true.shape[0] ]) # not [ y_true.shape[0], 1 ]
xph = tf.placeholder(tf.float32,[ testing_data_index, x_data.shape[1] ]) # not [ x_data.shape[0], 1 ]
yph = tf.placeholder(tf.float32,[ testing_data_index ]) # not [ y_true.shape[0], 1 ]
c = tf.matmul(xph, m)
y_model = c + b

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()



with tf.Session() as sess:
    
    sess.run(init)
    
    epoch = 1000

    for i in range(epoch):       
        feed = {
            xph: x_data[:testing_data_index, :],
            yph: y_true[:testing_data_index]
        }        
        sess.run(train, feed_dict=feed)
        
    model_m,model_b = sess.run([m,b])

    print('Success!!!!!!!')
    print('size of model_m ',end='')
    print(model_m.shape)
    print('\n')
    print('size of model_b ',end='')
    print(model_b.shape)

    c=tf.matmul( x_data[testing_data_index:,:], model_m )
    yee=x_data[testing_data_index:,:].shape[0]
    y_predict = c + model_b[ :yee ]
    print('shape of y_predict: ', y_predict.shape)
    print('\n')

    y_true_true=y_true[testing_data_index:]
    sess.run(tf.reshape(y_true_true, [-1]))
    sess.run(tf.reshape(y_predict, [-1]))
    print('shape of y_true_true ', y_true_true.shape)
    print('shape of y_predict ', y_predict.shape)
    print('model_y_position score: ',end='')
    r2_test = R_squared( y_true_true, y_predict)
    print( sess.run(r2_test) )

    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true_true, tf.reduce_mean(y_true_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true_true, y_predict)))
    R_squared_2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    print( sess.run(total_error) )
    print( sess.run(unexplained_error) )
    print( sess.run(R_squared_2) )

tEnd=time.time()

print('Overall processing time: '+ str ( round(tEnd-tStart, 3) )+' seconds' )