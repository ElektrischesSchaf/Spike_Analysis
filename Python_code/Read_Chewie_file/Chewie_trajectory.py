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

    if session_name =='Chewie_10032013':
        fig=plt.figure(figsize = (9, 17))
    if session_name=='Chewie_12192013':
        fig=plt.figure(figsize = (9, 17))
    target_grid_width=2

    plt.title('Session: '+session_name, fontsize=my_fontsize)
    plt.scatter( targets_corner[:,1], targets_corner[:,2], color='blue' , s=20)
    plt.scatter( targets_corner[:,3], targets_corner[:,4], color='green', s=20)
    for index in range(len(targets_corner[:,1])):
        x1=targets_corner[index,1]
        y1=targets_corner[index,2]
        x2=targets_corner[index,3]-target_grid_width
        y2=targets_corner[index,2]-target_grid_width
        x3=targets_corner[index,3]
        y3=targets_corner[index,4]
        x4=targets_corner[index,1]+target_grid_width
        y4=targets_corner[index,4]+target_grid_width

        plt.plot( [x1,x2],[y1,y2] ,color='magenta')
        plt.plot( [x1,x4],[y1,y4] ,color='magenta')
        plt.plot( [x2,x3],[y2,y3] ,color='magenta')
        plt.plot( [x3,x4],[y3,y4] ,color='magenta')

    plt.scatter( down_sampling_pos[:,0], down_sampling_pos[:,1], color='black' , s=5)
    plt.xlim([-10, 15])
    plt.ylim([-46, 11])
    plt.xlabel('mm', fontsize=my_fontsize)
    plt.xticks(fontsize=my_fontsize*0.5)
    plt.ylabel('mm', fontsize=my_fontsize)
    plt.yticks(fontsize=my_fontsize*0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig('xy_plane_tragectory_session_'+session_name+'.png')

    plt.cla()
    plt.clf()
    plt.close()
