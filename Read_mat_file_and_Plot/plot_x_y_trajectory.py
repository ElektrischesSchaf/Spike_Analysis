# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 
path=r'''../Kinematic_Variables_Plots/'''
with h5py.File('../Sorted_Spike_Dataset/indy_20160407_02.mat', 'r') as mat_file:

    '''
    finger_pos = mat_file['finger_pos']
    time_stamp=mat_file['t']

    print('finger_pos shape: ',end='')
    print(finger_pos.shape) # (3, 204446)
    print('time_stamp shape: ',end='')
    print(time_stamp.shape) # (1, 204446)
    '''

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

plot.scatter(finger_x_coor, finger_y_coor, s=1)
plot.title('X-Y plane')
plot.xlabel('X coordinate')
plot.ylabel('Y coordinate')
#plot.show()
plot.savefig(path+'X-Y_plane_trajectory.png')

plot.cla()
plot.clf()

plot.scatter(finger_x_coor, finger_z_coor, s=1)
plot.title('X-Z plane')
plot.xlabel('X coordinate')
plot.ylabel('Z coordinate')
#plot.show()
plot.savefig(path+'X-Z_plane_trajectory.png')

plot.cla()
plot.clf()

plot.scatter(finger_y_coor, finger_z_coor, s=1)
plot.title('Y-Z plane')
plot.xlabel('Y coordinate')
plot.ylabel('Z coordinate')
#plot.show()
plot.savefig(path+'Y-Z_plane_trajectory.png')