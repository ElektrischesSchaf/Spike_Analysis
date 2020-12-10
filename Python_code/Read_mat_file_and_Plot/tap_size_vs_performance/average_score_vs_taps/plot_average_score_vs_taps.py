import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np

my_fontsize=35


plt.figure(figsize=(16,9))
FILE_PATH = './tap_sizes_results_position' # tap_sizes_results_position , tap_sizes_results_velocity , tap_sizes_results_acceleration
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
file_list=List_FILE

CWD = os.getcwd()


for sess_name in file_list:
    file_name = FILE_PATH+ '/' +sess_name+'/'+ 'csv_files' + '/' + 'attn_weight_matrix_all' + '.csv'

    df = pd.read_csv(file_name, header=None )
    attn_weight_matrix_all=df.to_numpy()


    max_tap_sizes =  int(attn_weight_matrix_all.shape[1])

    # trip the beginning, because I have 0 padding
    attn_weight_matrix_all = attn_weight_matrix_all[max_tap_sizes: , :]

    mean = np.mean(attn_weight_matrix_all, axis=0)

    this_tap_sizes = len(mean)

    x_ticklabels=[]

    for ticks_label in reversed(range(this_tap_sizes)):

        if ticks_label == max_tap_sizes -1 :
            x_ticklabels.append( 't')
        else:
            x_ticklabels.append( 't-' + str( max_tap_sizes -1 - ticks_label) )

    mean = np.flip(mean)

    plt.figure(figsize=(16,9))

    plt.plot(x_ticklabels, mean, linewidth=5, color='blue')

    plt.xlabel('taps' , fontsize=my_fontsize)
    plt.xticks(fontsize=my_fontsize*0.3, rotation=45)
    plt.ylabel('Average attention score', fontsize=my_fontsize )
    plt.yticks(fontsize=my_fontsize*0.8)
    # plt.legend(loc='lower right', fontsize=my_fontsize*0.8 )
    plt.xlim([x_ticklabels[0], x_ticklabels[-1]])
    plt.grid(True, 'major', 'x')


    plt.tight_layout()

    plot_path_1 = os.path.join('..', 'Plots')
    if not os.path.exists(plot_path_1):
        os.mkdir(plot_path_1)

    if FILE_PATH == './tap_sizes_results_position':
        plot_path = os.path.join( plot_path_1 , 'Plots_bar_tap_sizes_position')
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plt.savefig(plot_path+'/'+ str(sess_name) +'.png')

    if FILE_PATH == './tap_sizes_results_velocity':
        plot_path = os.path.join( plot_path_1 , 'Plots_bar_tap_sizes_velocity')
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plt.savefig(plot_path+'/'+ str(sess_name) +'.png')

    if FILE_PATH == './tap_sizes_results_acceleration':
        plot_path = os.path.join( plot_path_1 , 'Plots_bar_tap_sizes_acceleration')
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plt.savefig(plot_path+'/'+ str(sess_name) +'.png')


    plt.cla()
    plt.clf()
    plt.close()

    min_score = np.min(mean)
    mean_minus_min = mean - min_score

    sum_score = 0
    sum_score=float(sum_score)
    mean_sum = []
    for i in mean_minus_min:
        sum_score = i + sum_score
        mean_sum.append(sum_score)

    mean_sum = np.array(mean_sum)
    
    # for k in mean_sum:
        # k = k /sum_score
    mean_sum = mean_sum/sum_score

    plt.figure(figsize=(16,9))

    plt.plot(x_ticklabels, mean_sum, linewidth=5, color='blue')

    plt.xlabel('taps' , fontsize=my_fontsize)
    plt.xticks(fontsize=my_fontsize*0.3, rotation=45)
    plt.ylabel('Score', fontsize=my_fontsize )
    plt.yticks(fontsize=my_fontsize*0.8)
    # plt.legend(loc='lower right', fontsize=my_fontsize*0.8 )
    plt.xlim([x_ticklabels[0], x_ticklabels[-1]])
    plt.grid(True, 'major', 'x')

    plt.tight_layout()

    plot_path_1 = os.path.join('..', 'Plots')
    if not os.path.exists(plot_path_1):
        os.mkdir(plot_path_1)

    if FILE_PATH == './tap_sizes_results_position':
        plot_path = os.path.join( plot_path_1 , 'Plots_bar_tap_sizes_position')
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plt.savefig(plot_path+'/'+ str(sess_name)+'_sum_score' +'.png')

    if FILE_PATH == './tap_sizes_results_velocity':
        plot_path = os.path.join( plot_path_1 , 'Plots_bar_tap_sizes_velocity')
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plt.savefig(plot_path+'/'+ str(sess_name)+'_sum_score' +'.png')

    if FILE_PATH == './tap_sizes_results_acceleration':
        plot_path = os.path.join( plot_path_1 , 'Plots_bar_tap_sizes_acceleration')
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plt.savefig(plot_path+'/'+ str(sess_name)+'_sum_score' +'.png')
    
    
    # plt.savefig()