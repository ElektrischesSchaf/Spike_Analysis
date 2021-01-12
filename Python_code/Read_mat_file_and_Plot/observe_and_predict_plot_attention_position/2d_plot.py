# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import os
import numpy
import matplotlib.pyplot as plt

my_fontsize=50

CWD = os.getcwd()

# plotting timeline setting
duration = 50  # time bins
start_time_bin = 1
end_time_bin = start_time_bin + duration


trajectory_2D_path = os.path.join(CWD, '2D_trajectory')
if not os.path.exists(trajectory_2D_path):
    os.mkdir(trajectory_2D_path)

position_reconstuction_path = os.path.join(CWD, 'position_reconstrution_vs_time')
if not os.path.exists(position_reconstuction_path):
    os.mkdir(position_reconstuction_path)

x_ground_truth = pd.read_csv("Ground_Truth_x_pos.csv", header=None)
x_ground_truth=np.array(x_ground_truth)

y_ground_truth = pd.read_csv("Ground_Truth_y_pos.csv", header=None)
y_ground_truth = np.array(y_ground_truth)

plotting_time_elapsed = pd.read_csv("plotting_time_elapsed.csv", header=None)
plotting_time_elapsed = np.array(plotting_time_elapsed)

prediction_x_attention = pd.read_csv("attention/my_prediction_x_pos.csv", header=None)
prediction_x_attention = np.array(prediction_x_attention)

prediction_y_attention = pd.read_csv("attention/my_prediction_y_pos.csv", header=None)
prediction_y_attention = np.array(prediction_y_attention)

prediction_x_no_attention = pd.read_csv("no_attention/my_prediction_x_pos.csv", header=None)
prediction_x_no_attention = np.array(prediction_x_no_attention)

prediction_y_no_attention = pd.read_csv("no_attention/my_prediction_y_pos.csv", header=None)
prediction_y_no_attention = np.array(prediction_y_no_attention)




while(end_time_bin < int( prediction_y_no_attention.shape[0] )-duration ):
    
    plt.figure(figsize=(12, 12))
    plt.plot(x_ground_truth[start_time_bin-1:end_time_bin], y_ground_truth[start_time_bin-1:end_time_bin], color='black', linestyle='dashed', linewidth=5 , label='actual')
    plt.plot(prediction_x_attention[start_time_bin-1:end_time_bin], prediction_y_attention[start_time_bin-1:end_time_bin], color='blue', linewidth=5 , alpha=0.7, label='with attention')
    plt.plot(prediction_x_no_attention[start_time_bin-1:end_time_bin], prediction_y_no_attention[start_time_bin-1:end_time_bin], color='green', linewidth=5 , alpha=0.7, label='without attention')

    plt.scatter(x_ground_truth[start_time_bin-1], y_ground_truth[start_time_bin-1], color='black', marker='o', s=300)
    plt.scatter(prediction_x_attention[start_time_bin-1], prediction_y_attention[start_time_bin-1], color='blue' , marker='o', s=300)
    plt.scatter(prediction_x_no_attention[start_time_bin-1], prediction_y_no_attention[start_time_bin-1], color='green' , marker='o', s=300)

    plt.scatter(x_ground_truth[end_time_bin-1], y_ground_truth[end_time_bin-1], color='black', marker='X', s=300)
    plt.scatter(prediction_x_attention[end_time_bin-1], prediction_y_attention[end_time_bin-1], color='blue' , marker='X', s=300)
    plt.scatter(prediction_x_no_attention[end_time_bin-1], prediction_y_no_attention[end_time_bin-1], color='green' , marker='X', s=300)

    # plot.title('X-Y plane', fontsize=my_fontsize, color='black')
    plt.xlabel('x (mm)', fontsize=my_fontsize, color='black')
    plt.ylabel('y (mm)', fontsize=my_fontsize, color='black')
    plt.xticks(fontsize=my_fontsize*0.5)
    plt.yticks(fontsize=my_fontsize*0.5)
    plt.legend(fontsize=my_fontsize*0.5, loc='upper right')

    plt.xlim([-60,70])
    plt.ylim([-10,120])
    plt.tight_layout()

    plt.savefig( trajectory_2D_path+ '/x_y_trajectory_with_attention'+ '_from_' + str( int(plotting_time_elapsed[start_time_bin,0]) )+'_to_'+str( int(plotting_time_elapsed[end_time_bin,0]) ) +'.png')

    plt.cla()
    plt.clf()
    plt.close()

    start_time_bin=end_time_bin
    end_time_bin=end_time_bin+duration





start_time_bin = 1
duration = 50*3
end_time_bin = start_time_bin + duration


while(end_time_bin < int( prediction_y_no_attention.shape[0] )-duration ):
    
    plt.figure(figsize=(20, 5))
    plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], x_ground_truth[start_time_bin-1:end_time_bin], color='black',linestyle='--', linewidth=5 , label='actual')
    # plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], y_ground_truth[start_time_bin-1:end_time_bin], color='black', linewidth=5 , label='actual y pos')  
    plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_x_attention[start_time_bin-1:end_time_bin], color='blue', linewidth=5 , alpha=0.7, label='with attention')
    # plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_y_attention[start_time_bin-1:end_time_bin], color='blue', linewidth=5 , alpha=0.7, label='with attention y pos')
    plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_x_no_attention[start_time_bin-1:end_time_bin], color='green', linewidth=5 , alpha=0.7, label='without attention')
    # plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_y_no_attention[start_time_bin-1:end_time_bin], color='green', linewidth=5 , alpha=0.7, label='without attention y pos')

    plt.xlim([plotting_time_elapsed[start_time_bin-1], plotting_time_elapsed[end_time_bin] ])

    plt.title('x-axis', fontsize=my_fontsize*0.7, color='black')
    plt.xlabel('Time (second)', fontsize=my_fontsize*0.5, color='black')
    plt.ylabel('Position (mm)', fontsize=my_fontsize*0.5, color='black')
    plt.xticks(fontsize=my_fontsize*0.5, color='black')
    plt.yticks(fontsize=my_fontsize*0.5, color='black')
    plt.legend(fontsize=my_fontsize*0.3, loc='upper right')
    plt.tight_layout()

    plt.savefig( position_reconstuction_path+ '/x_pos_vs_time'+ '_from_' + str( int(plotting_time_elapsed[start_time_bin,0]) )+'_to_'+str( int(plotting_time_elapsed[end_time_bin,0]) ) +'.png')
    # plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    plt.figure(figsize=(20, 5))
    # plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], x_ground_truth[start_time_bin-1:end_time_bin], color='black', linewidth=5 , label='actual x pos')
    plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], y_ground_truth[start_time_bin-1:end_time_bin], color='black',linestyle='--', linewidth=5 , label='actual')  
    # plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_x_attention[start_time_bin-1:end_time_bin], color='blue', linewidth=5 , alpha=0.7, label='with attention x pos')
    plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_y_attention[start_time_bin-1:end_time_bin], color='blue', linewidth=5 , alpha=0.7, label='with attention')
    # plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_x_no_attention[start_time_bin-1:end_time_bin], color='green', linewidth=5 , alpha=0.7, label='without attention x pos')
    plt.plot(plotting_time_elapsed[start_time_bin-1:end_time_bin], prediction_y_no_attention[start_time_bin-1:end_time_bin], color='green', linewidth=5 , alpha=0.7, label='without attention')

    plt.xlim([plotting_time_elapsed[start_time_bin-1], plotting_time_elapsed[end_time_bin] ])

    plt.title('y-axis', fontsize=my_fontsize*0.7, color='black')
    plt.xlabel('Time (second)', fontsize=my_fontsize*0.5, color='black')
    plt.ylabel('Position (mm)', fontsize=my_fontsize*0.5, color='black')
    plt.xticks(fontsize=my_fontsize*0.5, color='black')
    plt.yticks(fontsize=my_fontsize*0.5, color='black')
    plt.legend(fontsize=my_fontsize*0.3, loc='upper right')
    plt.tight_layout()

    plt.savefig( position_reconstuction_path+ '/y_pos_vs_time'+ '_from_' + str( int(plotting_time_elapsed[start_time_bin,0]) )+'_to_'+str( int(plotting_time_elapsed[end_time_bin,0]) ) +'.png')
    # plt.show()

    plt.cla()
    plt.clf()
    plt.close()



    start_time_bin=end_time_bin
    end_time_bin=end_time_bin+duration


