# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import matplotlib.ticker as ticker
my_fontsize=20
def histc( X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r


FILE_PATH = '../../Dataset/Chewie/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

for session_k in range(len(session_file_list)):
    session_name = str(session_file_list[session_k])[:-4]
    file_name='../../Dataset/Chewie/'+ session_name +'.mat'
    print('session_name: ',session_name, '\n')
    annots = loadmat(file_name)
    print(annots.keys()) # out_struct

    targets_corner = annots['out_struct']['targets'][0][0][0][0][0]
    targets_rotation = annots['out_struct']['targets'][0][0][0][0][1]
    pos = annots['out_struct']['pos'][0][0]
    vel = annots['out_struct']['vel'][0][0]
    acc = annots['out_struct']['acc'][0][0]

    time_stamp = pos[:,0]
    down_sampling_index=64
    down_sampling_pos = pos[::down_sampling_index][:,1:]
    down_sampling_vel = vel[::down_sampling_index][:,1:]
    down_sampling_acc = acc[::down_sampling_index][:,1:]
    time_stamp_64ms = time_stamp[::down_sampling_index]
    print('time_stamp_64ms shape = ', time_stamp_64ms.shape, '\n')
    print('down_sampling_pos shape = ', down_sampling_pos.shape, '\n')
    print('down_sampling_vel shape = ', down_sampling_vel.shape, '\n')
    print('down_sampling_acc shape = ', down_sampling_acc.shape, '\n')

    print('Session Duration: ', str(  round( (time_stamp_64ms[-1]-time_stamp_64ms[0]) /60 ,3)  ) + ' minutes'  )

    units = annots['out_struct']['units'][0][0] # 1x174
    total_unit_numbers = units.shape[1]
    print('\ntotal_unit_numbers= ', total_unit_numbers, '\n')

    print(pos.shape)
    print(vel.shape)
    print(acc.shape)
    print(units.shape)

    print('-'*30)

    print(targets_corner.shape, '\n')
    print(targets_rotation.shape, '\n')

    print('-'*30)
    unit_no = 52 # for 1 to total_unit_numbers
    # print('unit No. '+str(unit_no+1)+' spike train = ', units[0][unit_no][1], ', shape= ', units[0][unit_no][1].shape, '\n')
    # print('unit full ID: ' , units[0][unit_no][0], ', shape = ', units[0][unit_no][0].shape, '\n' )

    print('-'*30)
    firing_rate_cell=[[]]   
    for i in range(total_unit_numbers):
        temp = units[0][i][1]
        yee = histc(temp, time_stamp_64ms)
        firing_rate_cell.append(yee[:-1])
        firing_rate_cell.append([])

    firing_rate_final=[] # not[[]]
    for row_index in range( len( firing_rate_cell) ):   
        if len(firing_rate_cell[row_index]):
            firing_rate_final.append( firing_rate_cell[row_index] )

    # print('firing_rate_final= ', firing_rate_final, '\n')

    cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.05 }
    f, ax = plt.subplots( 3,1,gridspec_kw={'height_ratios': [4,1,1],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05},  figsize = (16, 9), constrained_layout=True)

    # sns.set()
    sns.heatmap(firing_rate_final ,cmap='YlGnBu_r', ax=ax[0] ,vmax=5, cbar_kws=cbar_kws_attention, xticklabels=False)
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[0].set_title('Session '+session_name, fontsize=my_fontsize)
    ax[0].set_ylabel( 'Units', rotation=90, fontsize=my_fontsize*0.5)

    ax[1].plot(time_stamp_64ms, down_sampling_pos[:,0], label='x-axis')
    ax[1].plot(time_stamp_64ms, down_sampling_pos[:,1], label='y-axis')
    ax[1].set_xlim([ time_stamp_64ms[0],time_stamp_64ms[-1] ])
    ax[1].legend(loc='upper center', fontsize=my_fontsize*0.5)
    ax[1].set_ylabel( 'Position (mm)', rotation=90, fontsize=my_fontsize*0.5)

    ax[2].plot(time_stamp_64ms, down_sampling_vel[:,0], label='x-axis')
    ax[2].plot(time_stamp_64ms, down_sampling_vel[:,1], label='y-axis')
    ax[2].set_xlim([ time_stamp_64ms[0],time_stamp_64ms[-1] ])
    ax[2].legend(loc='upper center', fontsize=my_fontsize*0.5)
    ax[2].set_ylabel( 'Velocity (mm/s)', rotation=90, fontsize=my_fontsize*0.5)
    ax[2].set_xlabel( 'Time (second)', fontsize=my_fontsize*0.8)
    ax[2].set_ylim([-30,30])

    # plt.show()
    plt.savefig('firing_rate_heatmap_and_kinematic_variable_session_'+session_name+'.png')

    plt.cla()
    plt.clf()
    plt.close()
