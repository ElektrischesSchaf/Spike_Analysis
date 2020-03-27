
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
import itertools
from itertools import islice
import time
tStart=time.time()
# Load my module
import sys
sys.path.append("..") # Adds higher directory to python modules path
import Inter_Channel_Module.parameters as my_parameters
import Inter_Channel_Module.buttersworth_filter as buttersworth_filter
my_parameters=my_parameters.my_parameters()

session_name=my_parameters.session_name
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
print('nwb_loop_indexE shape of mat_timestamp', mat_timestamp.shape, '\n')

delta= nwb_timestamp[1,]- nwb_timestamp[0,]

bandwidth_token='4-8Hz'


High_angle='Inter-Channel_Clustering_Output_Table/'+bandwidth_token+'/24kHz/24kHz_angle.csv'
High_abs='Inter-Channel_Clustering_Output_Table/'+bandwidth_token+'/24kHz/24kHz_abs.csv'
new_nwb_time_stamp='Inter-Channel_Clustering_Output_Table/0_5-40Hz/24kHz/24kHz_nwb_time_stamp.csv'
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
            chunk_index=np.where( target<chunk_new_nwb_time_stamp[:,0] )

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
            CWD = os.path.abspath(__file__)
            # CWD= os.path.join('..')
            if 'Inter-Channel_Clustering_Output_Table' not in CWD:
                CWD=os.path.join(CWD, 'Inter-Channel_Clustering_Output_Table')
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


tEnd=time.time()
print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )