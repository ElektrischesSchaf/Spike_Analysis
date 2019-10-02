# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plot 
import h5py

with h5py.File('../Sorted_Spike_Dataset/indy_20160407_02.mat', 'r') as mat_file:

    numpy_channel_names_array=mat_file.get('chan_names')
    numpy_channel_names_array=np.array(numpy_channel_names_array)

    numpy_spikes_array=mat_file.get('spikes')
    numpy_spikes_array=np.array(numpy_spikes_array)

    numpy_wf_array=mat_file.get('wf')
    numpy_wf_array=np.array(numpy_wf_array)

    chan_names = mat_file['chan_names'] 
    spikes = mat_file['spikes']
    wf = mat_file['wf']

    channel_number=int(numpy_spikes_array.shape[1] / 2) # 96 in indy_20160407_02

    print('numpy_wf_array shape: ',end='')
    print(numpy_wf_array.shape)  #  (3, 192) in indy_20160407_02.mat

    for channel_index in range(0, 13):
        temp_wf_cell_1=mat_file[ ( wf[0][channel_index] ) ][()]
        temp_wf_cell_2=mat_file[ ( wf[1][channel_index] ) ][()]
        temp_wf_cell_3=mat_file[ ( wf[2][channel_index] ) ][()]

        print('shape of temp_wf_cell_2: ', temp_wf_cell_2.shape)
        
        #temp_wf_cell_2=temp_wf_cell_1[0,:] # TODO figure out this

        print('shape of temp_wf_cell_2: ', temp_wf_cell_2.shape)

        temp_wf_cell_1=temp_wf_cell_1.flatten()
        temp_wf_cell_2=temp_wf_cell_2.flatten()
        temp_wf_cell_3=temp_wf_cell_3.flatten()

        print('shape of temp_wf_cell_2: ', temp_wf_cell_2.shape)

        # Set different colors for each neuron
        #colorCodes = np.array([ [1, 1, 0],[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1] ])
        colorCodes = np.array([ [0, 0, 0] ])
        # Set spike colors for each neuron
        lineSize = [0.9]
        
        plot.figure(figsize=(15,5))
        plot.margins(0,0)
        axes = plot.gca()
        #axes.set_xlim([50, 900])
        #axes.set_ylim([0.5, 3.5])

        x=[]

        for i in range( temp_wf_cell_2.shape[0]):
            x.append(i)

        # Draw a spike raster plot
        plot.scatter(x, temp_wf_cell_2, color=colorCodes)

        # Provide the title for the spike raster plot
        title_text='Wave form Plot Channel ' + str(channel_index+1) +' unit 2' 
        plot.title(  title_text )

        # Give x axis label for the spike raster plot
        plot.xlabel('Time')

        # Give y axis label for the spike raster plot
        plot.ylabel('Amptitude')

        # Display the spike raster plot
        #plot.show()
        path=r'''../Wave_Form_Plot/'''
        plot.savefig(path+ 'unit_2_'+'channel_' +str( f"{channel_index+1:03}" )  +'.png')
