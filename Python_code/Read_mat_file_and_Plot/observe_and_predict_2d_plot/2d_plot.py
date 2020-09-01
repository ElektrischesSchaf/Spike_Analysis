# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import os
import numpy
import matplotlib.pyplot as plt
my_fontsize=30
x_ground_truth = pd.read_csv("Ground_Truth_x_pos.csv", header=None)
x_ground_truth=np.array(x_ground_truth)
y_ground_truth = pd.read_csv("Ground_Truth_y_pos.csv", header=None)
y_ground_truth = np.array(y_ground_truth)

prediction_x_attention = pd.read_csv("attention/my_prediction_x_pos.csv", header=None)
prediction_x_attention = np.array(prediction_x_attention)

prediction_y_attention = pd.read_csv("attention/my_predictiony_y_pos.csv", header=None)
prediction_y_attention = np.array(prediction_y_attention)

prediction_x_no_attention = pd.read_csv("no_attention/my_prediction_x_pos.csv", header=None)
prediction_x_no_attention = np.array(prediction_x_no_attention)

prediction_y_no_attention = pd.read_csv("no_attention/my_predictiony_y_pos.csv", header=None)
prediction_y_no_attention = np.array(prediction_y_no_attention)

start_time_bin = 50
duration = 50
end_time_bin = start_time_bin+duration

while(end_time_bin < int( prediction_y_no_attention.shape[0] )-duration ):
    
    plt.figure(figsize=(12, 12))
    plt.plot(x_ground_truth[start_time_bin:end_time_bin], y_ground_truth[start_time_bin:end_time_bin], color='black', linewidth=5 , label='Actual')
    plt.plot(prediction_x_attention[start_time_bin:end_time_bin], prediction_y_attention[start_time_bin:end_time_bin], color='blue', linewidth=5 , alpha=0.7, label='with attention')
    plt.plot(prediction_x_no_attention[start_time_bin:end_time_bin], prediction_y_no_attention[start_time_bin:end_time_bin], color='green', linewidth=5 , alpha=0.7, label='no attention')

    plt.scatter(x_ground_truth[start_time_bin], y_ground_truth[start_time_bin], color='black', marker='D', s=300)
    plt.scatter(prediction_x_attention[start_time_bin], prediction_y_attention[start_time_bin], color='blue' , marker='D', s=300)
    plt.scatter(prediction_x_no_attention[start_time_bin], prediction_y_no_attention[start_time_bin], color='green' , marker='D', s=300)

    plt.scatter(x_ground_truth[end_time_bin-1], y_ground_truth[end_time_bin-1], color='black', marker='D', s=300)
    plt.scatter(prediction_x_attention[end_time_bin-1], prediction_y_attention[end_time_bin-1], color='blue' , marker='D', s=300)
    plt.scatter(prediction_x_no_attention[end_time_bin-1], prediction_y_no_attention[end_time_bin-1], color='green' , marker='D', s=300)

    # plot.title('X-Y plane', fontsize=my_fontsize, color='black')
    plt.xlabel('mm', fontsize=my_fontsize, color='black')
    plt.ylabel('mm', fontsize=my_fontsize, color='black')
    plt.xticks(fontsize=my_fontsize*0.8)
    plt.yticks(fontsize=my_fontsize*0.8)
    plt.legend(fontsize=my_fontsize*0.8, loc='upper right')
    plt.tight_layout()

    plt.savefig('2D_trajectory/x_y_trajectory_with_attention'+ '_from_' + str(start_time_bin)+'_to_'+str(end_time_bin) +'.png')
    # plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    start_time_bin=end_time_bin
    end_time_bin=end_time_bin+duration