# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import numpy
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import seaborn as sns

path=r'''../../Figures/Spike_and_Heatmap/'''
if not os.path.exists(path):
    os.mkdir(path)


def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

session_name='indy_20160411_02'
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

    for channel_index in range(0,96):
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


start_time_bin=100
end_time_bin=200
start_plot_time = time_stamp_64ms[start_time_bin]
end_plot_time = time_stamp_64ms[end_time_bin]

firing_rate_final=[] # not[[]]
for row_index in range( len( firing_rate_cell) ):   
    if len(firing_rate_cell[row_index]):
        firing_rate_final.append( firing_rate_cell[row_index] )

firing_rate_final=np.array(firing_rate_final)

# Eliniate empty units
valid_rows=[]
for row_idx in range(firing_rate_final.shape[0]):
    if not np.all( firing_rate_final[row_idx,:] ==0 ):
        valid_rows.append(row_idx)
data = firing_rate_final[valid_rows,:]

for rows in m1_raster[:]:
    if len(rows)==0:
        m1_raster.remove(rows)

for rows in s1_raster[:]:
    if len(rows)==0:
        s1_raster.remove(rows)

for i in range(len(m1_raster)):
    print(len(m1_raster[i]))

print(len(m1_raster), ' ', len(s1_raster), '\n')

my_plot_width = 20
my_plot_height = 20
my_fontsize = 45



# f ,ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,1],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05}, constrained_layout=True , figsize=(my_plot_width, my_plot_height*2))

plt.figure( figsize=(my_plot_width, my_plot_height) )
cbar_kws={"orientation": "horizontal", "shrink": 0.5, "aspect":50, "use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ 0, 4 ]}
ax = sns.heatmap( data=data, vmax=4 , xticklabels=False, yticklabels=True, cbar=False, cmap='YlGnBu_r') # important, not ax[0] = sns.heatmap(...)
# ax[0].set_xticklabels(ax[0].get_xmajorticklabels(), fontsize = my_fontsize, rotation=0)
ax.set_title('Firing rate', fontsize=my_fontsize)


ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = my_fontsize)

ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.set_xlim([ start_time_bin, end_time_bin ])
ax.set_xlabel('Time bins', fontsize=my_fontsize, color="black")
ax.set_ylabel('Units', fontsize=my_fontsize, color="black")

plt.tight_layout()

# plt.show()
plt.savefig(path+'/'+'the_result_heatmap.png')

plt.cla()
plt.clf()
plt.close()


f ,ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 1],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05}, constrained_layout=True , figsize=(my_plot_width, my_plot_height))

ax[0].set_title('M1 spike train', fontsize=my_fontsize)
ax[0].eventplot(m1_raster, linelengths=1 , color='black')
ax[0].set_xlim([ start_plot_time, end_plot_time ])

ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].get_xaxis().set_ticks([])
ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")
ax[0].tick_params(axis='y', labelsize= my_fontsize*0.5)


ax[1].set_title('S1 spike train', fontsize=my_fontsize)
ax[1].eventplot(s1_raster, linelengths=1 , color='black' )
ax[1].set_xlim([ start_plot_time, end_plot_time ])
ax[1].set_ylabel('Units', fontsize=my_fontsize, color="black")
ax[1].tick_params(axis='x', labelsize= my_fontsize*0.5)
ax[1].tick_params(axis='y', labelsize= my_fontsize*0.5)
ax[1].yaxis.set_major_locator(ticker.MultipleLocator(50))
ax[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1].set_xlabel('Time (seconds)', fontsize=my_fontsize )

# plt.show()
plt.savefig( path+'/'+'s1_and_m1_spike_train.png')

plt.cla()
plt.clf()
plt.close()

