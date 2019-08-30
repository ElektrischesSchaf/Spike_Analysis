# -*- coding: utf-8 -*-
import numpy as np
import h5py

with h5py.File('indy_20160407_02.mat', 'r') as mat_file:
    # <KeysViewHDF5 ['#refs#', 'chan_names', 'cursor_pos', 'finger_pos', 'spikes', 't', 'target_pos', 'wf']>
    print('print all keys: ',end='')
    print( list( mat_file.keys() ) )

    data=mat_file.get('chan_names')[()]
    print( (data[0][5]) )

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
    print(mat_file[ ( spikes[0][0] ) ].shape )

    spikes_data_1=mat_file[ ( spikes[0][0] ) ][()]
    print('shape of spikes_data_1: ',end='')
    print( spikes_data_1.shape )
    
    print( 'value of spikes_data_1[0][1000]: ',end='')
    print( spikes_data_1[0][1000] )

    '''
    # invalid
    print('list all elements in spikes(1X1): ')
    for i in range( spikes.shape[1] ):
        data=''.join([ chr(v[0]) for v in mat_file[ ( spikes[0][i] ) ]  ])
        print(data)
    '''    

    '''
    #valid
    print('list all elements in chan_names: ')
    for i in range( chan_names.shape[1] ):
        # print  <HDF5 object reference>
        data=''.join([ chr(v[0]) for v in mat_file[ (chan_names[0][i]) ] ])
        print(data)
    '''
