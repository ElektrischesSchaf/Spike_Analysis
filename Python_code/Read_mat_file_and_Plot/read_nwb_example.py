# -*- coding: utf-8 -*-
import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import os

session_name='indy_20161007_02'

nwb_filename = '../../Dataset/The_nwb_Raw_Dataset/'+session_name+'.nwb'
nwb_file = h5py.File(nwb_filename, 'r')
data = nwb_file['/acquisition/timeseries/broadband/data']

# yee=data.attrs['session_description']
conversion = data.attrs['conversion']
electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']
nwb_timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']
# electrode_name = nwb_file['/acquisition/timeseries/broadband/electrode_name']

print('print all nwb_file keys: ',end='')
print( list( nwb_file.keys() ) )


print('shape of data ', data.shape, '\n') # (12695457, 96) in indy_20160624_03

print('shape of electrode_map ', electrode_map.shape, '\n') # (96, 3) in indy_20160624_03
print('shape of nwb_timestamp ', nwb_timestamp.shape, '\n') #  (12695457,) in indy_20160624_03
print('first of nwb_timestamp= ', nwb_timestamp[0,],'\n')

