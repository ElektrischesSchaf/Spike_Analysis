
# -*- coding: utf-8 -*-
import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.signal import hilbert
import seaborn as sns
import os
import math
import pandas as pd
# Load my module
import sys
sys.path.append("..") # Adds higher directory to python modules path
import Inter_Channel_Module.parameters as my_parameters
import Inter_Channel_Module.buttersworth_filter as buttersworth_filter

my_parameters=my_parameters.my_parameters()
buttersworth_filter=buttersworth_filter.butterworth_filter()


channel_number=my_parameters.channel_number
# start_second=my_parameters.start_second
start_second=-1 # initial
plot_time_duration=my_parameters.plot_time_duration
end_second=-2
last_mat_timestep=-1

band_start=my_parameters.band_start
band_cutoff=my_parameters.band_start
session_name=my_parameters.session_name

# 'pos' or 'vel'
kinematic_variable_type=my_parameters.kinematic_variable_type

# nwb file
nwb_filename = '../../../Dataset/The_nwb_Raw_Dataset/'+session_name+'.nwb'
nwb_file = h5py.File(nwb_filename, 'r')
data = nwb_file['/acquisition/timeseries/broadband/data']
conversion = data.attrs['conversion']
electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']
nwb_timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']

# mat file
mat_file_name_1='../../../Dataset/Sorted_Spike_Dataset/'+session_name+'.mat'
mat_file=h5py.File(mat_file_name_1, 'r')
mat_timestamp=mat_file.get('t')
mat_timestamp=np.array(mat_timestamp)
print('YEEE shape of mat_timestamp', mat_timestamp.shape, '\n')

start_second=math.floor(mat_timestamp[0][0])
last_mat_timestep=math.floor(mat_timestamp[0][-1])
end_second=start_second+plot_time_duration

while(end_second<last_mat_timestep):


    mat_time_interval=np.where(np.logical_and(mat_timestamp[0,:]>start_second, mat_timestamp[0,:]<end_second ) )
    # print('mat_timestamp np.where result = ', end='')
    # print(mat_time_interval, '\n')
    # print('type of mat_time_interval=', type(mat_time_interval),'\n')
    # print('mat_time_interval start time index = ', mat_time_interval[0][0],'\n')
    # print('mat_time_interval end time index = ', mat_time_interval[0][-1],'\n')

    # new timestamp in mat file
    new_mat_time_stamp=mat_timestamp[0,mat_time_interval[0][0]:mat_time_interval[0][-1]]
    print('new_mat_time_stamp = ', new_mat_time_stamp,'\n')


    # Extract time interval from nwb file
    nwb_time_interval=np.where(np.logical_and(nwb_timestamp[:,]>start_second, nwb_timestamp[:,]<end_second ) )
    # print('nwb_timestamp np.where result = ', end='')    
    # print('type of nwb_time_interval = ', type(nwb_time_interval),'\n')
    # print('nwb_time_interval start time index = ', nwb_time_interval[0][0],'\n')
    # print('nwb_time_interval end time index = ', nwb_time_interval[0][-1],'\n')
    new_nwb_time_stamp= nwb_timestamp[nwb_time_interval[0][0]:nwb_time_interval[0][-1],]
    print('new_nwb_time_stamp = ', new_nwb_time_stamp, '\n')
    sampling_rate=1/( nwb_timestamp[1,]- nwb_timestamp[0,])

    channel_1=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], 0+channel_number]
    channel_2=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], 0+69]
    channel_3=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], 0+50]


    instance_phase_all_channels=[]
    # good_channel_list_start_from_one=[39,41,76,42,26,29,33,93,21,2,54]
    for channel_number_yee in range(96):
    # for channel_number_yee in good_channel_list_start_from_one:
        # channel_number_yee=channel_number_yee-1
        channel_1=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], 0+channel_number_yee]
        filtered_data_1=buttersworth_filter.butter_highpass_filter(channel_1, band_start, sampling_rate, order=2)
        filtered_data_1=buttersworth_filter.butter_lowpass_filter(filtered_data_1, band_cutoff, sampling_rate, order=2)
        analytic_signal_1 = hilbert(filtered_data_1)
        instantaneous_phase = np.angle(analytic_signal_1)
        instance_phase_all_channels.append(instantaneous_phase)

    instance_phase_all_channels=np.array(instance_phase_all_channels)

    ITPC_angle=[]
    ITPC_abs=[]
    for itpc_loop in range( instance_phase_all_channels.shape[1] ) :
        itpc_angle=0
        itpc_abs=0
        # itpc = abs(mean(exp( i* 1-D_signal_array )))
        # print('1: ', instance_phase_all_channels[:][itpc_loop:itpc_loop+1],'\n')
        # print('1j * instance_phase_all_channels[:,itpc_loop]=', 1j * instance_phase_all_channels[:,itpc_loop],'\n')
        itpc_angle=np.angle( np.mean (np.exp( 1j * instance_phase_all_channels[:,itpc_loop]  )))
        itpc_abs=np.abs( np.mean (np.exp( 1j * instance_phase_all_channels[:,itpc_loop]  )))

        ITPC_angle.append(itpc_angle)
        ITPC_abs.append(itpc_abs)

    ITPC_angle=np.array(ITPC_angle).transpose()
    ITPC_abs=np.array(ITPC_abs).transpose()

    print('---'*30)
    print('ITPC_angle shape= ', ITPC_angle.shape, '\n')
    print('ITPC_abs shape= ', ITPC_abs.shape, '\n')

    print('len of new_nwb_time_stamp= ', len(new_nwb_time_stamp), '\n')
    # Write result to csv
    CWD = os.getcwd()

    if 'Inter-Channel_Clustering_Output_Table' not in CWD:
        CWD=os.path.join(CWD, 'Inter-Channel_Clustering_Output_Table')
        if not os.path.exists(CWD):
                os.mkdir(CWD)

    if '0_5-40Hz' not in CWD:
        CWD=os.path.join(CWD, '0_5-40Hz')
        if not os.path.exists(CWD):
                os.mkdir(CWD)

    csv_path=os.path.join(CWD, '24kHz')
    if not os.path.exists(csv_path):
        os.mkdir(str(csv_path))

    print('csv_path= ', csv_path, '\n')

    # https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file
    df = pd.DataFrame(ITPC_angle)
    df.to_csv(os.path.join(csv_path,'24kHz_angle_0_5-40Hz.csv'), mode='a', index=False, header=False)

    df = pd.DataFrame(ITPC_abs)
    df.to_csv(os.path.join(csv_path,'24kHz_abs_0_5-40Hz.csv'), mode='a', index=False, header=False)

    df = pd.DataFrame(new_nwb_time_stamp)
    df.to_csv(os.path.join(csv_path,'24kHz_nwb_time_stamp.csv'), mode='a', index=False, header=False)

    start_second+=plot_time_duration
    end_second+=plot_time_duration
