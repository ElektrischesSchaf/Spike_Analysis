import numpy as np
import pandas as pd
import h5py
import os
CWD_origin=os.getcwd()

FILE_PATH = '../../Dataset/Sorted_Spike_Dataset/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

duration_across_all_sessions=[]
units_per_channel_all_sessions=[]
has_S1_array_all_sessions = []
original_testing_data_length_all_session = []

CWD=os.path.join(CWD_origin, 'sessions_data')
if not os.path.exists(CWD):
    os.mkdir(CWD)

for session_k in range(len(session_file_list)):
    session_name = str(session_file_list[session_k])[:-4]
    file_name_1='../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'

    with h5py.File(file_name_1, 'r') as mat_file:        
        time_stamp = mat_file['t'] # or time_stamp=mat_file.get('t') => time_stamp=np.array(time_stamp)
        time_stamp = time_stamp[0][:]
        duration =  str(   round((time_stamp[-1] - time_stamp[0])/60,1)    )
        duration_across_all_sessions.append(duration)

        numpy_spikes_array = mat_file.get('spikes')
        numpy_spikes_array = np.array(numpy_spikes_array)
        units_per_channel_all_sessions.append(numpy_spikes_array.shape[0])

        if numpy_spikes_array.shape[1] == 192:
            has_S1_array_all_sessions.append('True')
        else:
            has_S1_array_all_sessions.append('False')

        time_stamp_64ms = time_stamp[::16]
        original_testing_data_length = len(time_stamp_64ms[5000:]) - 1 
        original_testing_data_length_all_session.append(original_testing_data_length)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'session length':[x for x in duration_across_all_sessions] })
df.to_csv(os.path.join(CWD, 'duration_across_all_sessions.csv'), index=False, header=True)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'unit num':[x for x in units_per_channel_all_sessions] })
df.to_csv(os.path.join(CWD, 'units_per_channel_all_sessions.csv'), index=False, header=True)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'S1':[x for x in has_S1_array_all_sessions] })
df.to_csv(os.path.join(CWD, 'has_S1_array_all_sessions.csv'), index=False, header=True)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'original test length':[x for x in original_testing_data_length_all_session] })
df.to_csv(os.path.join(CWD, 'original_testing_data_length_all_session.csv'), index=False, header=True)