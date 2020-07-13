# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import numpy
import matplotlib.pyplot as plot 

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

    # plot each channel start
    plot_all_channels=[[]]
    for channel_index in range(96):
        #channel_index=0
        
        temp_spike_cell_1=[]
        plot_row = [[]]  
        for unit_index in range(unit_number):
            temp_spike_cell_1=mat_file[ ( spikes[unit_index][channel_index] ) ][()]
            plot_row.append([])
            plot_all_channels.append([])
            if temp_spike_cell_1.shape[0] != 2:
                for i in range (temp_spike_cell_1.shape[1]):
                    plot_row[-1].append( temp_spike_cell_1[0][i] )
                    plot_all_channels[-1].append( temp_spike_cell_1[0][i] )
            else:
                plot_row[-1].append(0)


        # Set different colors for each neuron
        # colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1] ])
        if unit_number == 3:
            colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1] ])
        if unit_number == 4:
            colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0] ])
        if unit_number == 5:
            colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1] ])
        if unit_number == 6:
            colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1] ])
        # Set spike colors for each neuron
        lineSize = [0.9]

        plot.figure(figsize=(16, 9))
        plot.margins(0,0)
        axes = plot.gca()
        # axes.set_xlim([50, 900])
        axes.set_xlim([spikes_data_1_1[0][1000], spikes_data_1_1[0][1000]+15])
        axes.set_ylim([0.5, 3.5])

        # Draw a spike raster plot
        plot.eventplot(plot_row, color=colorCodes, linelengths = lineSize)

        # Provide the title for the spike raster plot
        title_text='M1 Spike Train in Session '+ session_name + ' Channel ' + str(channel_index+1)
        plot.title(  '', fontsize=30)

        # Give x axis label for the spike raster plot
        plot.xlabel('Time (Second)', fontsize=25)
        plot.xticks(fontsize=25)

        # Give y axis label for the spike raster plot
        # plot.ylabel('Units')
        if unit_number == 3:
            plot.yticks([1, 2, 3], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2'], fontsize=25, rotation=0)
        if unit_number == 4:
            plot.yticks([1, 2, 3, 4], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2', 'Sorted Unit 3'], fontsize=25, rotation=0)
        if unit_number == 5:
            plot.yticks([1, 2, 3, 4, 5], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2', 'Sorted Unit 3', 'Sorted Unit 4'], fontsize=25, rotation=0)
        if unit_number == 6:
            plot.yticks([1, 2, 3, 4, 5, 6], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2', 'Sorted Unit 3', 'Sorted Unit 4', 'Sorted Unit 5'], fontsize=25, rotation=0)
        # Display the spike raster plot

        path=r'''../../Figures/Spike_Train_Plots/M1_Spike_Train_Channel_'''

        plot.tight_layout()
        plot.savefig(path+ str( f"{channel_index+1:03}" )  +'.png')
        # plot.show()

    # plot each channel start
    if channel_number==96*2:
        for channel_index in range(96, 96*2):
            temp_spike_cell_1=[]
            plot_row_S1 = [[]]  

            for unit_index in range(unit_number):
                temp_spike_cell_1=mat_file[ ( spikes[unit_index][channel_index] ) ][()]
                plot_row_S1.append([])
                if temp_spike_cell_1.shape[0] != 2:
                    for i in range (temp_spike_cell_1.shape[1]):
                        plot_row_S1[-1].append( temp_spike_cell_1[0][i] )
                else:
                    plot_row_S1[-1].append(0) 

            # Set different colors for each neuron
            # colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1] ])
            if unit_number == 3:
                colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1] ])
            if unit_number == 4:
                colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0] ])
            if unit_number == 5:
                colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1] ])
            if unit_number == 6:
                colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1] ])
            # Set spike colors for each neuron
            lineSize = [0.9]

            plot.figure(figsize=(16, 9))
            plot.margins(0,0)
            axes = plot.gca()
            # axes.set_xlim([50, 900])
            axes.set_xlim([spikes_data_1_1[0][1000], spikes_data_1_1[0][1000]+15])
            axes.set_ylim([0.5, 3.5])

            # Draw a spike raster plot
            plot.eventplot(plot_row_S1, color=colorCodes, linelengths = lineSize)

            # Provide the title for the spike raster plot
            title_text='S1 Spike Train in Session '+session_name+' Channel ' + str(channel_index+1 -96)
            plot.title(  '', fontsize=30)

            # Give x axis label for the spike raster plot
            plot.xlabel('Time (Second)', fontsize=25)
            plot.xticks(fontsize=25)

            # Give y axis label for the spike raster plot
            # plot.ylabel('Units')
            if unit_number == 3:
                plot.yticks([1, 2, 3], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2'], fontsize=25, rotation=0)
            if unit_number == 4:
                plot.yticks([1, 2, 3, 4], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2', 'Sorted Unit 3'], fontsize=25, rotation=0)
            if unit_number == 5:
                plot.yticks([1, 2, 3, 4, 5], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2', 'Sorted Unit 3', 'Sorted Unit 4'], fontsize=25, rotation=0)
            if unit_number == 6:
                plot.yticks([1, 2, 3, 4, 5, 6], ['Hash Unit', 'Sorted Unit 1', 'Sorted Unit 2', 'Sorted Unit 3', 'Sorted Unit 4', 'Sorted Unit 5'], fontsize=25, rotation=0)
            # Display the spike raster plot

            path=r'''../../Figures/Spike_Train_Plots/S1_Spike_Train_Channel_'''

            plot.tight_layout()
            plot.savefig(path+ str( f"{channel_index+1 -96:03}" )  +'.png')
            # plot.show()
    # plot spike train end

