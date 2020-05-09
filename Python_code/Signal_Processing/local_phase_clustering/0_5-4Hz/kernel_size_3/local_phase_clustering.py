
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
import gc
import math
import pandas as pd
import time
tStart=time.time()
# Load my module
import sys
sys.path.append('..') # Adds higher directory to python modules path
import Inter_Channel_Module.parameters as my_parameters
import Inter_Channel_Module.buttersworth_filter as buttersworth_filter
sys.path.append('../..') 
from data_processing.electrode_map_conv import map_conv_2D


my_parameters=my_parameters.my_parameters()
buttersworth_filter=buttersworth_filter.butterworth_filter()

this_cwd=os.getcwd()
channel_number=my_parameters.channel_number
# start_second=my_parameters.start_second
start_second=-1 # initial
plot_time_duration=my_parameters.plot_time_duration
end_second=-2
last_mat_timestep=-1

band_start=0.5
band_cutoff=4
kenel_size=3
# session_name=my_parameters.session_name
session_file_list=my_parameters.List_FILE
# print(session_file_list)

# Cross Sessions Control Start

for k in range(len(session_file_list)):

    ########################################################################## Raw to csv

    session_name=str(session_file_list[k])[:-4]
    print('session_name=', session_name)
    if band_start ==0.5:
        bandwidth_token='0_5'+'-'+ str(band_cutoff) + 'Hz'

    else:
        bandwidth_token=str(band_start) +'-'+ str(band_cutoff) + 'Hz'

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
    last_mat_timestep=mat_timestamp[0][-1]
    end_second=start_second+plot_time_duration

    is_first_loop=True # Optimize write file system
    is_last_loop=False

    while(is_last_loop==False):

        if end_second>last_mat_timestep:
            is_last_loop=True

        # Extract time interval from nwb file
        nwb_time_interval=np.where(np.logical_and(nwb_timestamp[:,]>start_second, nwb_timestamp[:,]<end_second ) )
        # print('nwb_timestamp np.where result = ', end='')    
        # print('type of nwb_time_interval = ', type(nwb_time_interval),'\n')
        # print('nwb_time_interval start time index = ', nwb_time_interval[0][0],'\n')
        # print('nwb_time_interval end time index = ', nwb_time_interval[0][-1],'\n')
        new_nwb_time_stamp= nwb_timestamp[nwb_time_interval[0][0]:nwb_time_interval[0][-1],]
        print('new_nwb_time_stamp = ', new_nwb_time_stamp, '\n')
        sampling_rate=1/( nwb_timestamp[1,]- nwb_timestamp[0,])



        instance_phase_all_channels=[]
        for channel_number_yee in range(96):
        # for channel_number_yee in good_channel_list_start_from_one:
            # channel_number_yee=channel_number_yee-1
            channel_1=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], 0+channel_number_yee]
            # filtered_data_1=buttersworth_filter.butter_highpass_filter(channel_1, band_start, sampling_rate, order=2)
            # filtered_data_1=buttersworth_filter.butter_lowpass_filter(filtered_data_1, band_cutoff, sampling_rate, order=2)
            filtered_data_1=buttersworth_filter.butter_bandpass_filter(channel_1, band_start, band_cutoff, sampling_rate, order=2)
            analytic_signal_1 = hilbert(filtered_data_1)
            instantaneous_phase = np.angle(analytic_signal_1)
            instance_phase_all_channels.append(instantaneous_phase)

        instance_phase_all_channels=np.array(instance_phase_all_channels)

        my_2d_conv=map_conv_2D(kenel_size, instance_phase_all_channels)
        [ITPC_angle, ITPC_abs] = my_2d_conv.conv2d_phase_clustering()

        print('---'*30)
        print('ITPC_angle shape= ', ITPC_angle.shape, '\n')
        print('ITPC_abs shape= ', ITPC_abs.shape, '\n')

        print('len of new_nwb_time_stamp= ', len(new_nwb_time_stamp), '\n')
        # Write result to csv
        CWD = this_cwd
        # CWD= os.path.join('..')
        if 'Tables_'+str(kenel_size)+'_kernel_size' not in CWD:
            CWD=os.path.join( CWD, 'Tables_'+str(kenel_size)+'_kernel_size' )
            if not os.path.exists(CWD):
                    os.mkdir(CWD)

        if session_name not in CWD:
            CWD=os.path.join(CWD, session_name)
            if not os.path.exists(CWD):
                os.mkdir(CWD)


        if bandwidth_token not in CWD:
            CWD=os.path.join(CWD, bandwidth_token)
            if not os.path.exists(CWD):
                    os.mkdir(CWD)

        csv_path=os.path.join(CWD, '24kHz')
        if not os.path.exists(csv_path):
            os.mkdir(str(csv_path))

        print('csv_path= ', csv_path, '\n')

        if is_first_loop==True:
            is_first_loop=False
            try:
                os.remove(os.path.join(csv_path,'24kHz_angle' + '.csv'))
                os.remove(os.path.join(csv_path,'24kHz_abs' +'.csv'))
                os.remove(os.path.join(csv_path,'24kHz_nwb_time_stamp.csv'))
                print('\nOld file deleted\n')
            except:
                print('\nNo old files\n')
        # https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file
        df = pd.DataFrame(ITPC_angle)
        df.to_csv(os.path.join(csv_path,'24kHz_angle'+ '.csv'), mode='a', index=False, header=False)

        df = pd.DataFrame(ITPC_abs)
        df.to_csv(os.path.join(csv_path,'24kHz_abs'+ '.csv'), mode='a', index=False, header=False)

        df = pd.DataFrame(new_nwb_time_stamp)
        df.to_csv(os.path.join(csv_path,'24kHz_nwb_time_stamp.csv'), mode='a', index=False, header=False)

        start_second+=plot_time_duration
        end_second+=plot_time_duration

    tEnd=time.time()
    print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )

    ########################################################################## csv to mat csv


    delta= nwb_timestamp[1,]- nwb_timestamp[0,]


    High_angle = 'Tables_'+str(kenel_size)+'_kernel_size' '/' + session_name +'/'+bandwidth_token +'/24kHz/24kHz_angle.csv'
    High_abs = 'Tables_'+str(kenel_size)+'_kernel_size' + '/' + session_name +'/'+bandwidth_token +'/24kHz/24kHz_abs.csv'
    new_nwb_time_stamp = 'Tables_'+str(kenel_size)+'_kernel_size' + '/' + session_name +'/'+bandwidth_token +'/24kHz/24kHz_nwb_time_stamp.csv'
    chunksize = 1e5

    is_first_loop=True # Optimize write file system

    # https://stackoverflow.com/questions/9394803/python-combine-two-for-loops
    # https://stackoverflow.com/questions/1663807/how-to-iterate-through-two-lists-in-parallel
    # https://stackoverflow.com/questions/28138392/skip-iterations-in-enumerated-list-object-python
    # https://kite.com/python/answers/how-to-skip-the-first-element-of-a-for-loop-in-python
    # https://stackoverflow.com/questions/10079216/skip-first-entry-in-for-loop-in-python

    iterator=enumerate(zip( pd.read_csv(new_nwb_time_stamp, chunksize=chunksize), pd.read_csv(High_angle, chunksize=chunksize), pd.read_csv(High_abs, chunksize=chunksize) ))

    skip=0
    for nwb_loop_index, (chunk_new_nwb_time_stamp, chunk_High_angle, chunk_High_abs) in iterator:

        chunk_new_nwb_time_stamp=np.array(chunk_new_nwb_time_stamp)
        chunk_High_angle=np.array(chunk_High_angle)
        chunk_High_abs=np.array(chunk_High_abs)
        print('nwb_loop_index = ', nwb_loop_index, '\n')

        for mat_loop_index in range(skip, mat_timestamp.shape[1]):

            target_start=time.time()

            nwb_timestamp_to_mat_timestamp=[]
            ITPC_abs_250Hz=[]
            ITPC_angle_250Hz=[]
            target=mat_timestamp[0][mat_loop_index]
            print('-'*50, '\ntarget= ', target, '\n')
            
            # for _ in range(skip):
                # next(iterator, None)
            # for nwb_loop_index, (chunk_new_nwb_time_stamp, chunk_High_angle, chunk_High_abs) in iterator:  

            print('enumerate number= ', nwb_loop_index, '\n')
            print('first= ', chunk_new_nwb_time_stamp[0], '\n')
            print('last= ', chunk_new_nwb_time_stamp[-1], '\n')

            if chunk_new_nwb_time_stamp[-1]<target:
                skip=mat_loop_index
                break

            # if chunk_new_nwb_time_stamp[0]>target:
            #     break

            # print('shape of chunk_new_nwb_time_stamp: ', chunk_new_nwb_time_stamp.shape, '\n')
            chunk_index=np.where( np.logical_and(target>chunk_new_nwb_time_stamp[:,0], target-delta<chunk_new_nwb_time_stamp[:,0] ))
            # print(type(chunk_index[0]))
            if chunk_index[0].size==0:
                # breakpoint() # TODO explain this
                chunk_index=np.where( target<chunk_new_nwb_time_stamp[:,0] ) # correct
                # pass # wrong

            if chunk_index[0].size>0:
                

                # if chunk_index[0].size>1:
                #     print(chunk_index[0])
                #     breakpoint()

                # print('chunk_index= ', chunk_index, ' \nchunk_new_nwb_time_stamp value=', chunk_new_nwb_time_stamp[chunk_index], '\nchunk_High_angle value=',chunk_High_angle[chunk_index], '\nchunk_High_abs value=',chunk_High_abs[chunk_index] ,'\n')
                
                nwb_timestamp_to_mat_timestamp.append(target)
                ITPC_abs_250Hz.append(chunk_High_abs[chunk_index[0][0]])
                ITPC_angle_250Hz.append( chunk_High_angle[chunk_index[0][0]] )



                nwb_timestamp_to_mat_timestamp=np.array(nwb_timestamp_to_mat_timestamp).transpose()
                ITPC_abs_250Hz=np.array(ITPC_abs_250Hz).transpose()
                ITPC_angle_250Hz=np.array(ITPC_angle_250Hz).transpose()

                # Write result to csv
                CWD = this_cwd
                # CWD= os.path.join('..')
                
                if 'Tables' not in CWD:
                    CWD=os.path.join(CWD, 'Tables')
                    if not os.path.exists(CWD):
                        os.mkdir(CWD)

                if session_name not in CWD:
                    CWD=os.path.join(CWD, session_name)
                    if not os.path.exists(CWD):
                        os.mkdir(CWD)

                if bandwidth_token not in CWD:
                    CWD=os.path.join(CWD, bandwidth_token )
                    if not os.path.exists(CWD):
                        os.mkdir(CWD)

                csv_path=os.path.join(CWD, '250Hz')
                if not os.path.exists(csv_path):
                    os.mkdir(str(csv_path))

                # print('csv_path= ', csv_path, '\n')

                if is_first_loop==True:
                    is_first_loop=False
                    try:
                        os.remove(os.path.join(csv_path,'nwb_timestamp_to_mat_timestamp.csv'))
                        os.remove(os.path.join(csv_path,'ITPC_abs_250Hz.csv'))
                        os.remove(os.path.join(csv_path,'ITPC_angle_250Hz.csv'))
                        print('\nOld file deleted\n')
                    except:
                        print('\nNo old files\n')

                df=pd.DataFrame(nwb_timestamp_to_mat_timestamp)
                df.to_csv(os.path.join(csv_path,'nwb_timestamp_to_mat_timestamp.csv'), mode='a', index=False, header=False)

                df=pd.DataFrame(ITPC_abs_250Hz)
                df.to_csv(os.path.join(csv_path,'ITPC_abs_250Hz.csv'), mode='a', index=False, header=False)

                df=pd.DataFrame(ITPC_angle_250Hz)
                df.to_csv(os.path.join(csv_path,'ITPC_angle_250Hz.csv'), mode='a', index=False, header=False)

                print('end one target search\n')
                target_end=time.time()
                print('target processing time: '+ str ( round( (target_end-target_start) , 7) )+' seconds' )
            
        del chunk_new_nwb_time_stamp
        del chunk_High_angle
        del chunk_High_abs
        gc.collect()

    os.remove(High_angle)
    os.remove(High_abs)
    os.remove(new_nwb_time_stamp)


    tEnd=time.time()
    print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )


# Cross Sessions Control End