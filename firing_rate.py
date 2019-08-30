# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 
import copy
with h5py.File('indy_20160407_02.mat', 'r') as mat_file:

    time_stamp=mat_file['t']
    spikes = mat_file['spikes']
    duration=100
    #duration=finger_pos.shape[1]

    # plot each channel start
    for channel_index in range(10):
        #channel_index=0

        temp_spike_cell_1=[]
        temp_spike_cell_2=[]
        temp_spike_cell_3=[]
        temp_spike_cell_4=[]
        temp_spike_cell_5=[]
        temp_spike_cell_6=[]

        firing_rate_cell_1=[[]]
        firing_rate_cell_2=[[]]
        firing_rate_cell_3=[[]]
        firing_rate_cell_4=[[]]
        firing_rate_cell_5=[[]]
        firing_rate_cell_6=[[]]

        plot_row = [[]]  

        temp_spike_cell_1=mat_file[ ( spikes[0][channel_index] ) ][()]
        temp_spike_cell_2=mat_file[ ( spikes[1][channel_index] ) ][()]
        temp_spike_cell_3=mat_file[ ( spikes[2][channel_index] ) ][()]
        temp_spike_cell_4=mat_file[ ( spikes[0][channel_index+96] ) ][()]
        temp_spike_cell_5=mat_file[ ( spikes[1][channel_index+96] ) ][()]
        temp_spike_cell_6=mat_file[ ( spikes[2][channel_index+96] ) ][()]
        
        if temp_spike_cell_1.shape[0] != 2:
            for a in range (temp_spike_cell_1.shape[1]):
                plot_row[-1].append( temp_spike_cell_1[0][a] )

            # firing rate
            i=0
            index=0
            k=0
            #for k in range(temp_spike_cell_1.shape[1]-1):
            while i<duration :
                print('i= ',end='')      
                print(i)
                print('\n')

                print('index_1=',end='')
                print(index)
                print('\n')

                print('k=',end='')
                print(k)
                print('\n')

                print('time target: ',end='')
                print(time_stamp[0][i])
                print('\n')

                print('length of firing_rate_cell_1: ',end='')
                print(len(firing_rate_cell_1[-1]))
                print('\n')

                if time_stamp[0][i] < temp_spike_cell_1[0][k] and time_stamp[0][i] > temp_spike_cell_1[0][k-1] :
                    firing_rate_cell_1[-1].append(k-index)
                    index=copy.deepcopy(k)
                    k=k-1

                    print('index_2= ',end='')
                    print(index)
                    print('\n')

                    #k=0
                    #if(i>duration):
                        #break
                    #else:
                        #i=i+1
                    i+=1
                    
                else:
                    #i=i+1
                    k=k+1
            #end firing rate

        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell_1: ',end='')
        print(len(firing_rate_cell_1[-1]))        
        firing_rate_cell_1.append([])

        plot_row.append([])

        if temp_spike_cell_2.shape[0] != 2:
            for i in range (temp_spike_cell_2.shape[1]):
                plot_row[-1].append( temp_spike_cell_2[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell_2: ',end='')
        print(len(firing_rate_cell_2[-1]))        
        firing_rate_cell_2.append([])

        plot_row.append([])

        if temp_spike_cell_3.shape[0] != 2:
            for i in range (temp_spike_cell_3.shape[1]):
                plot_row[-1].append( temp_spike_cell_3[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell_3: ',end='')
        print(len(firing_rate_cell_3[-1]))        
        firing_rate_cell_3.append([])

        plot_row.append([])

        if temp_spike_cell_4.shape[0] != 2:
            for i in range (temp_spike_cell_4.shape[1]):
                plot_row[-1].append( temp_spike_cell_4[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell_4: ',end='')
        print(len(firing_rate_cell_4[-1]))        
        firing_rate_cell_4.append([])

        plot_row.append([])

        if temp_spike_cell_5.shape[0] != 2:
            for i in range (temp_spike_cell_5.shape[1]):
                plot_row[-1].append( temp_spike_cell_5[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell_5: ',end='')
        print(len(firing_rate_cell_5[-1]))        
        firing_rate_cell_5.append([])

        plot_row.append([])


        if temp_spike_cell_6.shape[0] != 2:
            for i in range (temp_spike_cell_6.shape[1]):
                plot_row[-1].append( temp_spike_cell_6[0][i] )
        else:
            plot_row[-1].append(0)   

        print('length of firing_rate_cell_6: ',end='')
        print(len(firing_rate_cell_6[-1]))    