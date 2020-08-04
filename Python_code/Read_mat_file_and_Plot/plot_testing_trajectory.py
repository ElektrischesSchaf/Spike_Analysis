import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt


CWD_origin=os.getcwd()

FILE_PATH = '../../Dataset/Sorted_Spike_Dataset/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

my_height = 32
my_width = 4.5
my_fontsize = 30

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

CWD = os.path.join(CWD_origin, 'testing_data_trace')
if not os.path.exists(CWD):
    os.mkdir(CWD)

for session_k in range(len(session_file_list)):
    session_name = str(session_file_list[session_k])[:-4]
    file_name_1='../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'

    with h5py.File(file_name_1, 'r') as mat_file:        
        time_stamp = mat_file['t'] # or time_stamp=mat_file.get('t') => time_stamp=np.array(time_stamp)
        time_stamp = time_stamp[0][:]
        duration =  str(   round((time_stamp[-1] - time_stamp[0])/60,1)    )

        numpy_finger_pos = mat_file.get('finger_pos')
        numpy_finger_pos = np.array(numpy_finger_pos)

        finger_x_coor = numpy_finger_pos[1][:]*-10
        finger_y_coor = numpy_finger_pos[2][:]*-10
    
        finger_x_coor = finger_x_coor[::16]
        finger_y_coor = finger_y_coor[::16]
        time_stamp_64ms = time_stamp[::16]


        finger_x_coor_testing = finger_x_coor[5000:]
        finger_y_coor_testing = finger_y_coor[5000:]

        finger_x_coor_testing = finger_x_coor_testing[1:]
        finger_y_coor_testing = finger_y_coor_testing[1:]

        time_stamp_64ms_testing = time_stamp_64ms[5000:-1]

        original_testing_data_length = len( time_stamp_64ms_testing )

        aa_dict={
            "indy_20160407_02":	7777,
            "indy_20160411_01":	9894,
            "indy_20160411_02":	8749,
            "indy_20160418_01":	16212,
            "indy_20160419_01":	3187,
            "indy_20160420_01":	19268,
            "indy_20160426_01":	22532,
            "indy_20160622_01":	33276,
            "indy_20160624_03":	2812,
            "indy_20160627_01":	47546,
            "indy_20160630_01":	17863,
            "indy_20160915_01":	953,
            "indy_20160916_01":	2059,
            "indy_20160921_01":	627,
            "indy_20160927_04":	1083,
            "indy_20160927_06":	1578,
            "indy_20160930_02":	2199,
            "indy_20160930_05":	1343,
            "indy_20161005_06":	843,
            "indy_20161006_02":	2843,
            "indy_20161007_02":	2677,
            "indy_20161011_03":	5527,
            "indy_20161013_03":	3085,
            "indy_20161014_04":	3109,
            "indy_20161017_02":	2747,
            "indy_20161024_03":	2380,
            "indy_20161025_04":	2875,
            "indy_20161026_03":	2772,
            "indy_20161027_03":	4046,
            "indy_20161206_02":	6529,
            "indy_20161207_02":	1954,
            "indy_20161212_02":	3761,
            "indy_20161220_02":	4005,
            "indy_20170123_02":	4527,
            "indy_20170124_01":	4218,
            "indy_20170127_03":	6461,
            "indy_20170131_02":	7749,
            "loco_20170210_03":	2500,
            "loco_20170213_02":	26200,
            "loco_20170214_02":	5859,
            "loco_20170215_02":	12090,
            "loco_20170216_02":	7031,
            "loco_20170217_02":	5979,
            "loco_20170227_04":	25703,
            "loco_20170228_02":	15078,
            "loco_20170301_05":	4218,
            "loco_20170302_02":	17656
        }


        for session_name_from_list in  aa_dict:
            if session_name == session_name_from_list:
                length_difference = abs( original_testing_data_length - aa_dict[session_name] )
                print('length_difference with Makin2018 = ', length_difference, '\n')
                # start timming
                if length_difference != 0:
                    split_spot = time_stamp_64ms_testing[-length_difference]
                else:
                    split_spot = 0
                break

        plt.figure(figsize=(my_height,my_width))
        plt.title(session_name + ' testing data actual trajectory ', fontsize=my_fontsize, color='black')
        plt.plot( time_stamp_64ms_testing,finger_x_coor_testing ,'b' , linewidth=5, alpha=0.5, label='x-axis')
        plt.plot( time_stamp_64ms_testing,finger_y_coor_testing ,'g', linewidth=5, alpha=0.5, label='y-axis')

        if split_spot != 0:
            # plt.axvline(split_spot, color='red', linewidth=5, alpha=1)
            plt.axvspan(split_spot, time_stamp_64ms_testing[-1], color='red', linewidth=5, alpha=0.4)

        plt.ylabel('Position (mm)', fontsize=my_fontsize*0.8)
        plt.xlabel('Time (second)', fontsize=my_fontsize*0.8)
        plt.xticks(fontsize=my_fontsize*0.5)
        plt.yticks(fontsize=my_fontsize*0.5)
        plt.xlim([ time_stamp_64ms_testing[0],time_stamp_64ms_testing[-1] ])
        plt.legend(loc='upper left', fontsize=my_fontsize*0.8)
        plt.tight_layout()
        plt.savefig( CWD+'/'+ session_name +'_x_y_testing_data_trajectory.png' )

        plt.cla()
        plt.clf()
        plt.close()