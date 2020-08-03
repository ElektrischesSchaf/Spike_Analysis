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
real_units_all_sessions = []

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

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


        # calcuate units
        spikes = mat_file['spikes']
        firing_rate_cell=[[]]            

        print('spikes shape: ', spikes.shape) #  (3, 192) in indy_20160407_02
        unit_number=spikes.shape[0]
        
        # use S1 array data
        channel_number = spikes.shape[1]

        # don't use S1 array data
        # channel_number = 96

        actual_channel_number = 96

        time_stamp_64ms=np.asarray(time_stamp_64ms)
        time_stamp_64ms=time_stamp_64ms.flatten()

        # Making spike counts matrix
        # Handling M1 array data
        include_hash_unit = True
        for channel_index in range(actual_channel_number):
            for unit_index in range( spikes.shape[0] ):
                temp_spike_cell=[]
                temp_spike_cell=mat_file[( spikes[unit_index][channel_index] )][()]
                temp_spike_cell=np.asarray(temp_spike_cell)
                temp_spike_cell=temp_spike_cell.flatten()

                if temp_spike_cell.shape[0] != 2 and include_hash_unit==True:

                    yee=histc(temp_spike_cell, time_stamp_64ms)
                    #print('shape of yee:  ',yee.shape)
                    firing_rate_cell.append(yee[:-1])
                    #print('yee: ',yee)

                # else:
                #     r = np.zeros( len(time_stamp_64ms)-1 )
                #     firing_rate_cell.append(r)

                firing_rate_cell.append([])

        # Handling S1 array data
        if channel_number == actual_channel_number*2:
            print('Has S1')
            for channel_index in range(actual_channel_number, actual_channel_number*2):

                for unit_index in range( spikes.shape[0] ):
                    temp_spike_cell=[]
                    temp_spike_cell=mat_file[( spikes[unit_index][channel_index] )][()]
                    temp_spike_cell=np.asarray(temp_spike_cell)
                    temp_spike_cell=temp_spike_cell.flatten()

                    if temp_spike_cell.shape[0] != 2 and include_hash_unit==True:

                        yee=histc(temp_spike_cell, time_stamp_64ms)
                        #print('shape of yee:  ',yee.shape)
                        firing_rate_cell.append(yee[:-1])
                        #print('yee: ',yee)

                    # else:
                    #     r = np.zeros( len(time_stamp_64ms)-1 )
                    #     firing_rate_cell.append(r)

                    firing_rate_cell.append([])

        units_have_value = 0
        firing_rate_final=[] # not[[]]
        for row_index in range( len( firing_rate_cell) ):   
            if len(firing_rate_cell[row_index]):
                firing_rate_final.append( firing_rate_cell[row_index] )
                units_have_value += 1

        number_of_real_units = len(firing_rate_final)
        print('number_of_real_units= ', number_of_real_units, '  units_have_value= ', units_have_value, '\n')
        real_units_all_sessions.append(number_of_real_units)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'session length':[x for x in duration_across_all_sessions] })
df.to_csv(os.path.join(CWD, 'duration_across_all_sessions.csv'), index=False, header=True)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'unit num':[x for x in units_per_channel_all_sessions] })
df.to_csv(os.path.join(CWD, 'units_per_channel_all_sessions.csv'), index=False, header=True)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'S1':[x for x in has_S1_array_all_sessions] })
df.to_csv(os.path.join(CWD, 'has_S1_array_all_sessions.csv'), index=False, header=True)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'original test length':[x for x in original_testing_data_length_all_session] })
df.to_csv(os.path.join(CWD, 'original_testing_data_length_all_session.csv'), index=False, header=True)

df = pd.DataFrame({ 'session': [ str(x)[:-4] for x in session_file_list], 'units number':[x for x in real_units_all_sessions] })
df.to_csv(os.path.join(CWD, 'real_units_all_sessions.csv'), index=False, header=True)