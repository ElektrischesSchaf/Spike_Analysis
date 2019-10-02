# -*- coding: utf-8 -*-
import numpy as np
import h5py

with h5py.File('../Sorted_Spike_Dataset/indy_20160407_02.mat', 'r') as mat_file:
    # <KeysViewHDF5 ['#refs#', 'chan_names', 'cursor_pos', 'finger_pos', 'spikes', 't', 'target_pos', 'wf']>
    print('print all keys: ',end='')
    print( list( mat_file.keys() ) )

    data=mat_file.get('chan_names')[()]
    print( (data[0][5]) )

    # ['#refs#', 'chan_names', 'cursor_pos', 'finger_pos', 'spikes', 't', 'target_pos', 'wf']
    #chan_names=list (mat_file['chan_names'] )
    numpy_channel_names_array=mat_file.get('chan_names')
    numpy_channel_names_array=np.array(numpy_channel_names_array)

    numpy_spikes_array=mat_file.get('spikes')
    numpy_spikes_array=np.array(numpy_spikes_array)

    chan_names = mat_file['chan_names'] 
    spikes = mat_file['spikes']

    print('numpy_channel_names_array shape: ',end='')
    print(numpy_channel_names_array.shape) # (1, 192) in indy_20160407_02.mat

    print('numpy_spikes_array shape: ',end='')
    print(numpy_spikes_array.shape)  #  (3, 192) in indy_20160407_02.mat

    print('chan_names shape:  ',end='')
    print(chan_names.shape) # (1, 192) in indy_20160407_02.mat

    print('spikes shape:  ',end='')
    print(spikes.shape) # (3, 192) in indy_20160407_02.mat

    print('chan_names dtype:  ',end='')
    print(chan_names.dtype)  # object 

    print('spikes dtype: ',end='')
    print(spikes.dtype)   # object

    print('len of chan_names:  ',end='')
    print( len(chan_names) ) # len(chan_names) =1

    print('type of chan_names[0]:  ',end='' )
    print( type( chan_names[0] ) )  #  <class 'numpy.ndarray'>

    print('type of spikes[0]:  ',end='' )
    print( type( spikes[0] ) )  #  <class 'numpy.ndarray'>

    print('type of mat_file[ (chan_names[0][0]) ]:  ',end='')
    print( type(mat_file[ (chan_names[0][0]) ]))  #  <class 'h5py._hl.dataset.Dataset'>

    data_chan_names_1=mat_file[ (chan_names[0][0]) ][()]

    print('shape of data_chan_names_1: ',end='')  # (6, 1) in indy_20160407_02.mat
    print(data_chan_names_1.shape)

    for i in range(data_chan_names_1.shape[0]):
        print( chr(data_chan_names_1[i][0] ))

    print('shape of mat_file[ ( spikes[0][0] ) ]: ',end='')
    print(mat_file[ ( spikes[0][0] ) ].shape ) # (1, 5595) in indy_20160407_02.mat

    spikes_data_1=mat_file[ ( spikes[0][0] ) ][()]
    print('shape of spikes_data_1: ',end='')
    print( spikes_data_1.shape )  # (1, 5595) in indy_20160407_02.mat
    
    print( 'value of spikes_data_1[0][1000]: ',end='')
    print( spikes_data_1[0][1000] )  #  224.40043620355942 in indy_20160407_02.mat

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
