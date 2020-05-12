
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

band_start=0.5
band_cutoff=4
kenel_size=7

# Make file list
kinematic_variable_type='x_vel' # x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc
FILE_PATH = '../Phase_all_Channels/Tables/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

# Cross Sessions Control Start

for k in range(len(session_file_list)):

    session_name=str(session_file_list[k])
    print('session_name=', session_name)
    if band_start ==0.5:
        bandwidth_token='0_5'+'-'+ str(band_cutoff) + 'Hz'

    else:
        bandwidth_token=str(band_start) +'-'+ str(band_cutoff) + 'Hz'

    phase_of_firing_all_channel=[]
    for PoF_channel_index in range(0, 96):
        # TODO use np.concatenate instead
        file_phase_of_firing = FILE_PATH + session_name+'/'+bandwidth_token+'/250Hz/'+str(PoF_channel_index) +'/instance_phase_a_channel_250Hz.csv'
        PoF_one_channel=pd.read_csv(file_phase_of_firing, dtype=float)
        PoF_one_channel=np.array(PoF_one_channel)
        PoF_one_channel=PoF_one_channel.flatten() # important
        phase_of_firing_all_channel.append( PoF_one_channel )

    instance_phase_all_channels=np.array(phase_of_firing_all_channel)
    print( 'phase_of_firing_all_channel shape= ', instance_phase_all_channels.shape)

    my_2d_conv=map_conv_2D(kenel_size, instance_phase_all_channels)
    [ITPC_angle, ITPC_abs] = my_2d_conv.conv2d_phase_clustering()

    print('---'*30)
    print('phase_conv_angle shape= ', ITPC_angle.shape, '\n')
    print('phase_conv_abs shape= ', ITPC_abs.shape, '\n')

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

    csv_path=os.path.join(CWD, bandwidth_token)
    if not os.path.exists(csv_path):
        os.mkdir(str(csv_path))

    print('csv_path= ', csv_path, '\n')

    # https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file
    df = pd.DataFrame(ITPC_angle)
    df.to_csv(os.path.join(csv_path,'phase_conv_angle'+ '.csv'), mode='w', index=False, header=False)

    df = pd.DataFrame(ITPC_abs)
    df.to_csv(os.path.join(csv_path,'phase_conv_abs'+ '.csv'), mode='w', index=False, header=False)
    tEnd=time.time()
    
print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )