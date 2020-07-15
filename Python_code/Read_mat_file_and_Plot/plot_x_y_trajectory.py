# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 
path=r'''../../Figures/Kinematic_Variables_Plots/'''
session_name='indy_20161025_04'
with h5py.File('../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat', 'r') as mat_file:

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

    numpy_finger_target=mat_file.get('target_pos')
    numpy_finger_target=np.array(numpy_finger_target)

    numpy_time_stamp=mat_file.get('t')
    numpy_time_stamp=np.array(numpy_time_stamp)

    print('numpy_finger_pos shape: ',end='')
    print(numpy_finger_pos.shape) #  (3, 204446)

    print('numpy_time_stamp: ',end='')
    print(numpy_time_stamp.shape)  #  (1, 204446)

    print('numpy_finger_target: ',end='')
    print(numpy_finger_target.shape)  #  (2, 204446)

    finger_z_coor=numpy_finger_pos[0][:]*-10
    finger_x_coor=numpy_finger_pos[1][:]*-10
    finger_y_coor=numpy_finger_pos[2][:]*-10

    target_x_coor=numpy_finger_target[0][:]
    target_y_coor=numpy_finger_target[1][:]

my_fontsize = 30
my_height = 25
my_width = 9

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

plot.cla()
plot.clf()


plot.figure(figsize=(my_height,my_width))
plot.title(session_name, fontsize=my_fontsize, color='black')
plot.plot( numpy_time_stamp[0,10000:20000],finger_x_coor[10000:20000] ,'b' , linewidth=5, alpha=0.7, label='x-axis recording')
plot.plot( numpy_time_stamp[0,10000:20000],target_x_coor[10000:20000] ,'b--', linewidth=5, alpha=0.8, label='x-axis target cue')
plot.plot( numpy_time_stamp[0,10000:20000],finger_y_coor[10000:20000] ,'g' , linewidth=5, alpha=0.7, label='y-axis recording')
plot.plot( numpy_time_stamp[0,10000:20000],target_y_coor[10000:20000] ,'g--', linewidth=5, alpha=0.8, label='y-axis target cue')
plot.ylabel('Position (mm)', fontsize=my_fontsize)
plot.xlabel('Time (second)', fontsize=my_fontsize)
plot.xticks(fontsize=my_fontsize*0.5)
plot.yticks(fontsize=my_fontsize*0.5)
plot.xlim([ numpy_time_stamp[0,10000], numpy_time_stamp[0,20000]])
plot.legend(loc='upper right', fontsize=my_fontsize*0.8)
plot.tight_layout()
plot.savefig(path+'x_and_y_trajectory_and_cue.png')