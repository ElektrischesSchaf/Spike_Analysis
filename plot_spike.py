# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 

with h5py.File('indy_20160407_02.mat', 'r') as mat_file:
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

    data_chan_names_1=mat_file[ (chan_names[0][0]) ][()]
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



    # plot each channel start
    for channel_index in range(94,96):
        #channel_index=0

        temp_spike_cell_1=[]
        temp_spike_cell_2=[]
        temp_spike_cell_3=[]
        temp_spike_cell_4=[]
        temp_spike_cell_5=[]
        temp_spike_cell_6=[]

        plot_row = [[]]  

        temp_spike_cell_1=mat_file[ ( spikes[0][channel_index] ) ][()]
        temp_spike_cell_2=mat_file[ ( spikes[1][channel_index] ) ][()]
        temp_spike_cell_3=mat_file[ ( spikes[2][channel_index] ) ][()]
        temp_spike_cell_4=mat_file[ ( spikes[0][channel_index+96] ) ][()]
        temp_spike_cell_5=mat_file[ ( spikes[1][channel_index+96] ) ][()]
        temp_spike_cell_6=mat_file[ ( spikes[2][channel_index+96] ) ][()]

        
        plot_row.append([])
        if temp_spike_cell_1.shape[0] != 2:
            for i in range (temp_spike_cell_1.shape[1]):
                plot_row[-1].append( temp_spike_cell_1[0][i] )
        else:
            plot_row[-1].append(0)

        plot_row.append([])

        if temp_spike_cell_2.shape[0] != 2:
            for i in range (temp_spike_cell_2.shape[1]):
                plot_row[-1].append( temp_spike_cell_2[0][i] )
        else:
            plot_row[-1].append(0)

        plot_row.append([])

        if temp_spike_cell_3.shape[0] != 2:
            for i in range (temp_spike_cell_3.shape[1]):
                plot_row[-1].append( temp_spike_cell_3[0][i] )
        else:
            plot_row[-1].append(0)

        plot_row.append([])

        if temp_spike_cell_4.shape[0] != 2:
            for i in range (temp_spike_cell_4.shape[1]):
                plot_row[-1].append( temp_spike_cell_4[0][i] )
        else:
            plot_row[-1].append(0)

        plot_row.append([])

        if temp_spike_cell_5.shape[0] != 2:
            for i in range (temp_spike_cell_5.shape[1]):
                plot_row[-1].append( temp_spike_cell_5[0][i] )
        else:
            plot_row[-1].append(0)

        plot_row.append([])

        if temp_spike_cell_6.shape[0] != 2:
            for i in range (temp_spike_cell_6.shape[1]):
                plot_row[-1].append( temp_spike_cell_6[0][i] )
        else:
            plot_row[-1].append(0)

        # Set different colors for each neuron
        colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1] ])

        # Set spike colors for each neuron
        lineSize = [0.9]

        plot.figure(figsize=(15,5))
        plot.margins(0,0)
        axes = plot.gca()
        axes.set_xlim([50, 900])
        axes.set_ylim([0.5, 6.5])        

        # Draw a spike raster plot
        plot.eventplot(plot_row, color=colorCodes, linelengths = lineSize)

        # Provide the title for the spike raster plot
        title_text='Spike Train Plot Channel ' + str(channel_index+1)
        plot.title(  title_text )

        # Give x axis label for the spike raster plot
        plot.xlabel('Time')

        # Give y axis label for the spike raster plot
        plot.ylabel('Unit')

        # Display the spike raster plot
        #plot.show()

        plot.savefig('Spike_Train_Channel_'+ str(channel_index+1) +'.png' )

    # plot spike train end

