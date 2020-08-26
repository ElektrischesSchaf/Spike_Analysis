# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import numpy
import matplotlib.pyplot as plt 

path=r'''../../Figures/Spike_Train_Plots/'''
if not os.path.exists(path):
    os.mkdir(path)


def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

session_name='indy_20160407_02'
with h5py.File('../../Dataset/Sorted_Spike_Dataset/'+session_name+'.mat', 'r') as mat_file:
    # <KeysViewHDF5 ['#refs#', 'chan_names', 'cursor_pos', 'finger_pos', 'spikes', 't', 'target_pos', 'wf']>
    print('print all keys: ',end='')
    print( list( mat_file.keys() ) )

    # ['#refs#', 'chan_names', 'cursor_pos', 'finger_pos', 'spikes', 't', 'target_pos', 'wf']
    #chan_names=list (mat_file['chan_names'] )
    chan_names = mat_file['chan_names'] 
    spikes = mat_file['spikes']

    print('chan_names shape:  ',end='')
    print(chan_names.shape)

    print('spikes shape:  ',end='')
    print(spikes.shape)
    unit_number = spikes.shape[0]
    channel_number = spikes.shape[1]

    print('chan_names dtype:  ',end='')
    print(chan_names.dtype)

    print('spikes dtype: ',end='')
    print(spikes.dtype)

    print('len of chan_names:  ',end='')
    print( len(chan_names) ) # print 1

    print('type of chan_names[0]:  ',end='' )
    print( type( chan_names[0] ) )

    print('type of spikes[0]:  ',end='' )
    print( type( spikes[0] ) )

    print('type of mat_file[ (chan_names[0][0]) ]:  ',end='')
    print( type(mat_file[ (chan_names[0][0]) ]))

    data_chan_names_1 = mat_file[ (chan_names[0][0]) ][()]
    print('shape of data_chan_names_1: ',end='')
    print(data_chan_names_1.shape)
    for i in range(data_chan_names_1.shape[0]):
        print( chr(data_chan_names_1[i][0] ))

    print('shape of mat_file[ ( spikes[0][0] ) ]: ',end='')
    print(mat_file[(spikes[0][0])].shape )

    spikes_data_1_1=mat_file[ ( spikes[0][0] ) ][()]
    print('shape of spikes_data_1_1: ',end='')
    print( spikes_data_1_1.shape )
    
    print( 'value of spikes_data_1_1[0][1000]: ',end='')
    print( spikes_data_1_1[0][1000] )

    # Making spike counts matrix
    # Handling M1 array data
    firing_rate_cell=[[]]
    m1_raster=[[]]
    s1_raster=[[]]

    time_stamp=mat_file['t'] # or time_stamp=mat_file.get('t') => time_stamp=np.array(time_stamp)
    sampling_rate=16 # because 64ms
    time_stamp_64ms=time_stamp[0][::sampling_rate]
    print('lenght of time_stamp_64ms: ', len(time_stamp_64ms))
    time_stamp_64ms=np.asarray(time_stamp_64ms)
    time_stamp_64ms=time_stamp_64ms.flatten()

    for channel_index in range(96):
        for unit_index in range( spikes.shape[0] ):
            temp_spike_cell=[]
            temp_spike_cell=mat_file[( spikes[unit_index][channel_index] )][()]
            temp_spike_cell=np.asarray(temp_spike_cell)
            temp_spike_cell=temp_spike_cell.flatten()

            m1_raster.append([])
            if temp_spike_cell.shape[0] != 2:
                for i in range ( len(temp_spike_cell) ):
                    m1_raster[-1].append( temp_spike_cell[i] )
                yee=histc(temp_spike_cell, time_stamp_64ms)
                #print('shape of yee:  ',yee.shape)
                firing_rate_cell.append(yee[:-1])
                #print('yee: ',yee)

            else:
                r = np.zeros( len(time_stamp_64ms)-1 )
                firing_rate_cell.append(r)
                # m1_raster[-1].append(0)

            firing_rate_cell.append([])

    if channel_number==192:
        for channel_index in range(96,96*2):
            for unit_index in range( spikes.shape[0] ):
                temp_spike_cell=[]
                temp_spike_cell=mat_file[( spikes[unit_index][channel_index] )][()]
                temp_spike_cell=np.asarray(temp_spike_cell)
                temp_spike_cell=temp_spike_cell.flatten()

                s1_raster.append([])
                if temp_spike_cell.shape[0] != 2:
                    for i in range ( len(temp_spike_cell) ):
                        s1_raster[-1].append( temp_spike_cell[i] )
                    yee=histc(temp_spike_cell, time_stamp_64ms)
                    #print('shape of yee:  ',yee.shape)
                    firing_rate_cell.append(yee[:-1])
                    #print('yee: ',yee)

                else:
                    r = np.zeros( len(time_stamp_64ms)-1 )
                    firing_rate_cell.append(r)
                    # s1_raster[-1].append(0)

                firing_rate_cell.append([])


firing_rate_final=[] # not[[]]
for row_index in range( len( firing_rate_cell) ):   
    if len(firing_rate_cell[row_index]):
        firing_rate_final.append( firing_rate_cell[row_index] )

remove_index=[]
for i in range( len(m1_raster) ):
    # print(len(m1_raster[i]))
    if len(m1_raster[i]) <= 1:
        remove_index.append(i)
        # print('hi')

for i in range(len(remove_index)):
    m1_raster.pop(i)
    
for i in range( len(m1_raster) ):
    print(len(m1_raster[i]))

# remove_index_s1=[]
# for i in range( len(s1_raster) ):
    # if len(s1_raster[i]) <= 1:
        # remove_index_s1.append(i)
# for i in range(len(remove_index_s1)):
    # s1_raster.pop(i)

my_plot_width = 16
my_plot_height = 9
f ,ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 1],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05}, constrained_layout=True , figsize=(my_plot_width, my_plot_height*1.2))
ax[0].eventplot(m1_raster, linelengths=0.3)
ax[0].set_xlim([ 100,200 ])
ax[1].eventplot(s1_raster, linelengths=0.3)
ax[1].set_xlim([ 100,200 ])
# plt.show()
plt.savefig('yee.png')