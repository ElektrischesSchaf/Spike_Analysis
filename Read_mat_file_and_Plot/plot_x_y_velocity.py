# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 
path=r'''../Kinematic_Variables_Plots/'''
with h5py.File('../Sorted_Spike_Dataset/indy_20160407_02.mat', 'r') as mat_file:

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

    finger_z_coor=numpy_finger_pos[0][:]
    finger_x_coor=numpy_finger_pos[1][:]
    finger_y_coor=numpy_finger_pos[2][:]

    finger_x_velocity=[]
    finger_y_velocity=[]
    finger_z_velocity=[]
    velocity_time_coor=[]

    finger_x_acceleration=[]
    finger_y_acceleration=[]
    finger_z_acceleration=[]
    acceleration_time_coor=[]

    #duration=204445
    duration=numpy_time_stamp.shape[1]

    for i in range(duration):
        print('Velocity computing progress: ' + str( round( (i/duration)*100, 3) )+' %' )
        
        if ( i<duration-1 ):
            velocity=( numpy_finger_pos[0][i+1] - numpy_finger_pos[0][i] ) / ( numpy_time_stamp[0][i+1]-numpy_time_stamp[0][i] )
            finger_x_velocity.append(velocity)

            velocity=( numpy_finger_pos[1][i+1] - numpy_finger_pos[1][i] ) / ( numpy_time_stamp[0][i+1]-numpy_time_stamp[0][i] )
            finger_y_velocity.append(velocity)

            velocity=( numpy_finger_pos[2][i+1] - numpy_finger_pos[2][i] ) / ( numpy_time_stamp[0][i+1]-numpy_time_stamp[0][i] )
            finger_z_velocity.append(velocity)

            velocity_time_coor.append( numpy_time_stamp[0][i] )

        else:        
            finger_x_velocity.append(0)
            finger_y_velocity.append(0)
            finger_z_velocity.append(0)
            velocity_time_coor.append(0)
    
    finger_x_velocity=np.array(finger_x_velocity)
    finger_x_velocity=finger_x_velocity.astype(np.float64)
    
    finger_y_velocity=np.array(finger_y_velocity)
    finger_y_velocity=finger_y_velocity.astype(np.float64)

    finger_z_velocity=np.array(finger_z_velocity)
    finger_z_velocity=finger_z_velocity.astype(np.float64)

    velocity_time_coor=np.array(velocity_time_coor)

    for i in range(duration):
        print('Aceeleration computing progress '+ str( round( (i/duration)*100, 3) )+' %')
        if(i<duration-1):
            acceleration=(finger_x_velocity[i+1]-finger_x_velocity[i])/(velocity_time_coor[i+1]-velocity_time_coor[i])
            finger_x_acceleration.append(acceleration)

            acceleration=(finger_y_velocity[i+1]-finger_y_velocity[i])/(velocity_time_coor[i+1]-velocity_time_coor[i])
            finger_y_acceleration.append(acceleration)

            acceleration=(finger_z_velocity[i+1]-finger_z_velocity[i])/(velocity_time_coor[i+1]-velocity_time_coor[i])
            finger_z_acceleration.append(acceleration)

            acceleration_time_coor.append(velocity_time_coor[i])
        else:
            finger_x_acceleration.append(0)
            finger_y_acceleration.append(0)
            finger_z_acceleration.append(0)
            acceleration_time_coor.append(0)
    
    finger_x_acceleration=np.array(finger_x_acceleration)
    finger_x_acceleration=finger_x_acceleration.astype(np.float64)
    
    finger_y_acceleration=np.array(finger_y_acceleration)
    finger_y_acceleration=finger_y_acceleration.astype(np.float64)

    finger_z_acceleration=np.array(finger_z_acceleration)
    finger_z_acceleration=finger_z_acceleration.astype(np.float64)

    acceleration_time_coor=np.array(acceleration_time_coor)

print('shape of finger_x_velocity: ', finger_x_velocity.shape )

plot.figure(figsize=(15,5))
plot.scatter(velocity_time_coor, finger_x_velocity, s=1)
plot.title('X axis velocity with respect to time')
plot.xlabel('Time')
plot.ylabel('X velocity')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig(path+'X_axis_velocity.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
plot.scatter(velocity_time_coor, finger_y_velocity, s=1)
plot.title('Y axis velocity with respect to time')
plot.xlabel('Time')
plot.ylabel('Y velocity')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig(path+'Y_axis_velocity.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
plot.scatter(velocity_time_coor, finger_z_velocity, s=1)
plot.title('Z axis velocity with respect to time')
plot.xlabel('Time')
plot.ylabel('Z velocity')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig(path+'Z_axis_velocity.png' )
        
plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
plot.scatter(acceleration_time_coor, finger_x_acceleration, s=1)
plot.title('X axis acceleration with respect to time')
plot.xlabel('Time')
plot.ylabel('X acceleration')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig(path+'X_axis_acceleration.png' )


plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
plot.scatter(acceleration_time_coor, finger_y_acceleration, s=1)
plot.title('y axis acceleration with respect to time')
plot.xlabel('Time')
plot.ylabel('y acceleration')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig(path+'Y_axis_acceleration.png' )

plot.cla()
plot.clf()

plot.figure(figsize=(15,5))
plot.scatter(acceleration_time_coor, finger_z_acceleration, s=1)
plot.title('z axis acceleration with respect to time')
plot.xlabel('Time')
plot.ylabel('z acceleration')
axes = plot.gca()
axes.set_xlim([60, 890])
#plot.show()
plot.savefig(path+'Z_axis_acceleration.png' )