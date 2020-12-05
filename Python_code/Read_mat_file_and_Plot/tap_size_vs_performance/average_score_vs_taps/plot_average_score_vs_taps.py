import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np

my_fontsize=35


plt.figure(figsize=(16,9))
FILE_PATH = './tap_sizes_results_position' # tap_sizes_results_position , tap_sizes_results_velocity
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
file_list=List_FILE

kinematci_types = 'x_and_y_vel'

for sess_name in file_list:
    file_name = FILE_PATH+ '/' +sess_name+'/'+ 'csv_files' + '/' + 'attn_weight_matrix_all' + '.csv'

    df = pd.read_csv(file_name, header=None )
    attn_weight_matrix_all=df.to_numpy()
    mean = np.mean(attn_weight_matrix_all, axis=0)

    this_tap_sizes = len(mean)



    x_ticklabels=[]
    for ticks_label in range(this_tap_sizes):

        if ticks_label == int(attn_weight_matrix_all.shape[1]) -1 :
            x_ticklabels.append( 't')
        else:
            x_ticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )

    plt.figure(figsize=(16,9))

    plt.plot(x_ticklabels, mean, linewidth=5, color='blue')


    plt.xlabel('taps' , fontsize=my_fontsize)
    plt.xticks(fontsize=my_fontsize*0.5, rotation=45)
    plt.ylabel('Average attention score', fontsize=my_fontsize )
    plt.yticks(fontsize=my_fontsize*0.8)
    # plt.legend(loc='lower right', fontsize=my_fontsize*0.8 )
    plt.xlim([x_ticklabels[0], x_ticklabels[-1]])
    plt.grid(True, 'major', 'x')
    plt.tight_layout()
    if FILE_PATH == './tap_sizes_results_position':
        plt.savefig('../tap_sizes_position_'+ str(sess_name) +'.png')
    if FILE_PATH == './tap_sizes_results_velocity':
        plt.savefig('../tap_sizes_velocity_'+ str(sess_name) +'.png')
    if FILE_PATH == './tap_sizes_results_acceleration':
        plt.savefig('../tap_sizes_acceleration_'+ str(sess_name) +'.png')

    # plt.savefig()