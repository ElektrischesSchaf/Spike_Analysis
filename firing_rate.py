# -*- coding: utf-8 -*-
import numpy as np
import h5py

import numpy
import matplotlib.pyplot as plot 
import copy
with h5py.File('indy_20160407_02.mat', 'r') as mat_file:

    time_stamp=mat_file['t']
    spikes = mat_file['spikes']
    firing_rate_cell=[[]]
    duration=1000
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
            i=0    #i is the index for time_stemp
            index=0
            k=0    #k is the index for spikes
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

                print('length of firing_rate_cell[-1]: ',end='')
                print(len(firing_rate_cell[-1]))
                print('\n')

                if time_stamp[0][i] < temp_spike_cell_1[0][k] and time_stamp[0][i] > temp_spike_cell_1[0][k-1] :
                    firing_rate_cell[-1].append(k-index)
                    index=k
                    k=k-1

                    print('index_2= ',end='')
                    print(index)
                    print('\n')

                    #k=0
                    #if(i>duration):
                        #break
                    #else:
                        #i=i+1
                    i+=16
                    
                else:
                    #i=i+1
                    k=k+1
            #end firing rate

        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell_1[-1]: ',end='')
        print(len(firing_rate_cell[-1]))

        firing_rate_cell.append([])

        plot_row.append([])

        if temp_spike_cell_2.shape[0] != 2:

            for i in range (temp_spike_cell_2.shape[1]):
                plot_row[-1].append( temp_spike_cell_2[0][i] )

            # firing rate
            i=0    #i is the index for time_stemp
            index=0
            k=0    #k is the index for spikes
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

                print('length of firing_rate_cell[-1]: ',end='')
                print(len(firing_rate_cell[-1]))
                print('\n')

                if time_stamp[0][i] < temp_spike_cell_2[0][k] and time_stamp[0][i] > temp_spike_cell_2[0][k-1] :
                    firing_rate_cell[-1].append(k-index)
                    index=k
                    k=k-1

                    print('index_2= ',end='')
                    print(index)
                    print('\n')

                    #k=0
                    #if(i>duration):
                        #break
                    #else:
                        #i=i+1
                    i+=16
                    
                else:
                    #i=i+1
                    k=k+1
            #end firing rate


        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell[-1]: ',end='')
        print(len(firing_rate_cell[-1]))
        firing_rate_cell.append([])

        plot_row.append([])

        if temp_spike_cell_3.shape[0] != 2:
            for i in range (temp_spike_cell_3.shape[1]):
                plot_row[-1].append( temp_spike_cell_3[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell[-1]: ',end='')
        print(len(firing_rate_cell[-1]))        
        firing_rate_cell.append([])

        plot_row.append([])

        if temp_spike_cell_4.shape[0] != 2:
            for i in range (temp_spike_cell_4.shape[1]):
                plot_row[-1].append( temp_spike_cell_4[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell[-1]: ',end='')
        print(len(firing_rate_cell[-1]))        
        firing_rate_cell.append([])

        plot_row.append([])

        if temp_spike_cell_5.shape[0] != 2:
            for i in range (temp_spike_cell_5.shape[1]):
                plot_row[-1].append( temp_spike_cell_5[0][i] )
        else:
            plot_row[-1].append(0)

        print('length of firing_rate_cell[-1]: ',end='')
        print(len(firing_rate_cell[-1]))        
        firing_rate_cell.append([])

        plot_row.append([])


        if temp_spike_cell_6.shape[0] != 2:
            for i in range (temp_spike_cell_6.shape[1]):
                plot_row[-1].append( temp_spike_cell_6[0][i] )
        else:
            plot_row[-1].append(0)

        print('row numbers of firing_rate_cell: ',end='')
        print( len( firing_rate_cell) )

        for row_index in range( len( firing_rate_cell) ):
            print('length of firing_rate_cell['+ str(row_index) +']: ',end='')
            print(len(firing_rate_cell[row_index]))
        