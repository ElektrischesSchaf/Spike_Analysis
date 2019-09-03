# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 

with h5py.File('indy_20160407_02.mat', 'r') as mat_file:

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

    finger_x_coor=numpy_finger_pos[0][:]
    finger_y_coor=numpy_finger_pos[1][:]
    finger_z_coor=numpy_finger_pos[2][:]

    finger_x_velocity=[]
    finger_y_velocity=[]
    finger_z_velocity=[]
    time_coor=[]

    #duration=204445
    duration=numpy_time_stamp.shape[1] - 1 

    for i in range(duration):
        print('progress: ' + str( (i/duration)*100 )+'%' )
        
        if ( i<duration ):
            velocity=( numpy_finger_pos[0][i+1] - numpy_finger_pos[0][i] ) / ( numpy_time_stamp[0][i+1]-numpy_time_stamp[0][i] )
            finger_x_velocity.append(velocity)

            velocity=( numpy_finger_pos[1][i+1] - numpy_finger_pos[1][i] ) / ( numpy_time_stamp[0][i+1]-numpy_time_stamp[0][i] )
            finger_y_velocity.append(velocity)

            velocity=( numpy_finger_pos[2][i+1] - numpy_finger_pos[2][i] ) / ( numpy_time_stamp[0][i+1]-numpy_time_stamp[0][i] )
            finger_z_velocity.append(velocity)

            time_coor.append( numpy_time_stamp[0][i] )

plot.figure(figsize=(15,5))
plot.scatter(time_coor, finger_x_velocity, s=1)
plot.title('X axis velocity with respect to time')
plot.xlabel('Time')
plot.ylabel('X velocity')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig('X_axis_velocity.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
plot.scatter(time_coor, finger_y_velocity, s=1)
plot.title('Y axis velocity with respect to time')
plot.xlabel('Time')
plot.ylabel('Y velocity')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig('Y_axis_velocity.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
plot.scatter(time_coor, finger_z_velocity, s=1)
plot.title('Z axis velocity with respect to time')
plot.xlabel('Time')
plot.ylabel('Z velocity')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig('Z_axis_velocity.png' )
        

        
