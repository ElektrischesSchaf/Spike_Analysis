# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 

with h5py.File('indy_20160407_02.mat', 'r') as mat_file:

    finger_pos = mat_file['finger_pos']
    time_stamp=mat_file['t']

    print('finger_pos shape: ',end='')
    print(finger_pos.shape) # (3, 204446)
    print('time_stamp shape: ',end='')
    print(time_stamp.shape) # (1, 204446)

    finger_x_coor=[]
    finger_y_coor=[]
    finger_z_coor=[]

    finger_x_velocity=[]
    finger_y_velocity=[]
    finger_z_velocity=[]

    time_coor=[]

    #duration=1000
    duration=finger_pos.shape[1]

    for i in range(duration):
        print('progress: ' + str( (i/duration)*100 )+'%' )

        finger_x_coor.append(  finger_pos[0][i]  )
        finger_y_coor.append(  finger_pos[1][i]  )
        finger_z_coor.append(  finger_pos[2][i]  )
        time_coor.append( time_stamp[0][i] )

        if i<= duration-1:
            velocity=( finger_pos[0][i+1] - finger_pos[0][i] ) / ( time_stamp[0][i+1]-time_stamp[0][i] )
            finger_x_velocity.append(velocity)

            velocity=( finger_pos[1][i+1] - finger_pos[1][i] ) / ( time_stamp[0][i+1]-time_stamp[0][i] )
            finger_y_velocity.append(velocity)

            velocity=( finger_pos[2][i+1] - finger_pos[2][i] ) / ( time_stamp[0][i+1]-time_stamp[0][i] )
            finger_z_velocity.append(velocity)

print('length of finger_x_coor: ',end='')
print(len(finger_x_coor))

plot.scatter(finger_x_coor, finger_y_coor)
plot.title('X-Y plane')
plot.xlabel('X coordinate')
plot.ylabel('Y coordinate')
#plot.show()
plot.savefig('X-Y_plane_trajectory.png' )

plot.cla()
plot.clf()

plot.scatter(finger_x_velocity, finger_y_velocity)
plot.title('X-Y velocity')
plot.xlabel('X velocity')
plot.ylabel('Y velocity')
#plot.show()
print('line 74')
plot.savefig('X-Y_plane_velocity.png' )