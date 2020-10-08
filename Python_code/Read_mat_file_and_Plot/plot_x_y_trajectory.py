# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import numpy
import matplotlib.pyplot as plot

path=r'''../../Figures/Kinematic_Variables_Plots/'''
if not os.path.exists(path):
    os.mkdir(path)

session_name='indy_20161025_04' #indy_20161025_04 8X8, indy_20160411_02 8X17, loco_20170214_02 6X6
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
my_height = 9
my_width = 25

plot.scatter(finger_x_coor, finger_y_coor, s=1)
plot.title('X-Y plane')
plot.xlabel('X coordinate')
plot.ylabel('Y coordinate')
#plot.show()
plot.savefig(path+'X-Y_plane_trajectory.png')

plot.cla()
plot.clf()
plot.close()

plot.scatter(finger_x_coor, finger_z_coor, s=1)
plot.title('X-Z plane')
plot.xlabel('X coordinate')
plot.ylabel('Z coordinate')
#plot.show()
plot.savefig(path+'X-Z_plane_trajectory.png')

plot.cla()
plot.clf()
plot.close()

plot.scatter(finger_y_coor, finger_z_coor, s=1)
plot.title('Y-Z plane')
plot.xlabel('Y coordinate')
plot.ylabel('Z coordinate')
#plot.show()
plot.savefig(path+'Y-Z_plane_trajectory.png')

plot.cla()
plot.clf()
plot.close()


# Plot virtual panel layout
plot.figure(figsize=(8, 8))
plot.scatter(target_x_coor, target_y_coor, color='black' , s=80)
# plot.title('X-Y plane', fontsize=my_fontsize, color='black')
plot.xlabel('mm', fontsize=my_fontsize, color='black')
plot.ylabel('mm', fontsize=my_fontsize, color='black')
plot.xticks(fontsize=my_fontsize*0.8)
plot.yticks(fontsize=my_fontsize*0.8)
plot.tight_layout()
plot.savefig(path+'X-Y_VR_panel.png')

plot.cla()
plot.clf()
plot.close()


# Plot trajectory in VR panel
plot_start_sample_time = 9840 #10125
plot_end_sample_time = plot_start_sample_time + 590 # 1830 before for more trials

plot.figure(figsize=(9,9))
plot.scatter( target_x_coor, target_y_coor, color='black', s=80)
# for i in range(plot_start_sample_time, plot_end_sample_time+1):
plot.scatter( target_x_coor[plot_start_sample_time:plot_end_sample_time], target_y_coor[plot_start_sample_time:plot_end_sample_time], marker='D', color='#A37E2C', s=300)

plot.scatter( finger_x_coor[plot_start_sample_time:plot_end_sample_time], finger_y_coor[plot_start_sample_time:plot_end_sample_time], marker='.', color='#006039', s=50, alpha=0.6)

plot.scatter( finger_x_coor[plot_start_sample_time], finger_y_coor[plot_start_sample_time], marker='$A$', color='green', s=700, alpha=1)
plot.scatter( finger_x_coor[plot_end_sample_time], finger_y_coor[plot_end_sample_time], marker='$B$', color='green', s=700, alpha=1)

plot.title('Session ' + session_name, fontsize=my_fontsize*0.8, color='black')
plot.xlabel('mm', fontsize=my_fontsize*0.8, color='black')
plot.ylabel('mm', fontsize=my_fontsize*0.8, color='black')
plot.xticks(fontsize=my_fontsize*0.8)
plot.yticks(fontsize=my_fontsize*0.8)
plot.tight_layout()
plot.savefig(path+'X-Y_trajectory_in_VR_panel.png')

plot.cla()
plot.clf()
plot.close()


plot.figure( figsize=( my_width, my_height*0.7 ))
plot.title('Session ' + session_name, fontsize=my_fontsize, color='black')
plot.plot( numpy_time_stamp[0,plot_start_sample_time:plot_end_sample_time], finger_x_coor[plot_start_sample_time:plot_end_sample_time] ,'b--' , linewidth=5, alpha=0.7, label='x-axis recording')
# plot.plot( numpy_time_stamp[0,plot_start_sample_time:plot_end_sample_time], target_x_coor[plot_start_sample_time:plot_end_sample_time] ,'b', linewidth=5, alpha=0.8, label='x-axis target cue')
plot.plot( numpy_time_stamp[0,plot_start_sample_time:plot_end_sample_time], finger_y_coor[plot_start_sample_time:plot_end_sample_time] ,'g--' , linewidth=5, alpha=0.7, label='y-axis recording')
# plot.plot( numpy_time_stamp[0,plot_start_sample_time:plot_end_sample_time], target_y_coor[plot_start_sample_time:plot_end_sample_time] ,'g', linewidth=5, alpha=0.8, label='y-axis target cue')

the_x = target_x_coor[plot_start_sample_time:plot_end_sample_time]
the_y = target_y_coor[plot_start_sample_time:plot_end_sample_time]
change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

change_points_x_set = set(change_points_x)
change_points_y_set = set(change_points_y)

print('change_points_x_set= ', change_points_x_set)
for ele in list(change_points_x_set.union( change_points_y_set )):
    if ele != 0:
        plot.axvline( numpy_time_stamp[0, plot_start_sample_time+ele] , color='black' , linewidth=5, alpha=0.4 )

A_spot = (finger_x_coor[plot_start_sample_time] + finger_y_coor[plot_start_sample_time] ) /2
B_spot = (finger_x_coor[plot_end_sample_time] + finger_y_coor[plot_end_sample_time] ) /2

plot.scatter( numpy_time_stamp[0,plot_start_sample_time], A_spot, marker='$A$', color='black', s= 1000, alpha=1)
plot.scatter( numpy_time_stamp[0,plot_end_sample_time], B_spot, marker='$B$', color='black', s= 1000, alpha=1)

plot.ylabel('Position (mm)', fontsize=my_fontsize*0.8)
plot.xlabel('Time (second)', fontsize=my_fontsize*0.8)
plot.xticks(fontsize=my_fontsize*0.8)
plot.yticks(fontsize=my_fontsize*0.8)
# plot.xlim([ numpy_time_stamp[0,plot_start_sample_time], numpy_time_stamp[0,plot_end_sample_time]])
plot.legend(loc='upper right', fontsize=my_fontsize*0.8)
plot.tight_layout()
plot.savefig(path+'x_and_y_trajectory_and_cue.png')