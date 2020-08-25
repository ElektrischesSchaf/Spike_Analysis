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
#print('shape of conversion', conversion.data, '\n')
print('shape of electrode_map ', electrode_map.shape, '\n') # (96, 3) in indy_20160624_03
print('shape of nwb_timestamp ', nwb_timestamp.shape, '\n') #  (12695457,) in indy_20160624_03
print('first of nwb_timestamp= ', nwb_timestamp[0,],'\n')

for i in range( electrode_map.shape[1] ):
    print('electrode_map= \n', electrode_map[:,i])
# print(electrode_map[:,1].shape)

m1=np.zeros(shape=(96,3))
for i in range(96):
    m1[i][0]=electrode_map[:,0][i]*1000 # x
    m1[i][1]=electrode_map[:,1][i]*1000 # y
    m1[i][2]=i+1

plt.figure(figsize=(15,15))
my_fontsize=30
ax = plt.subplot(111)
for i in range(96):
    plt.title('M1 electrodes from session '+session_name, fontsize=my_fontsize)
    plt.scatter( m1[i][0], m1[i][1] , s=80)
    plt.annotate( str( int( m1[i][2] ) ), xy=(m1[i][0], m1[i][1]) , fontsize=my_fontsize)
    plt.ylim([-1.5,-1.5+4])
    plt.xlabel('mm', fontsize=my_fontsize)
    plt.xticks( fontsize=my_fontsize*0.8)
    plt.ylabel('mm', fontsize=my_fontsize)
    plt.yticks( fontsize=my_fontsize*0.8)


path=r'''../../Figures/Electorde_map/'''
if not os.path.exists(path):
    os.mkdir(path)

plt.tight_layout()
plt.savefig(path+'_'+session_name+'.png')

yee = nwb_file['/acquisition/timeseries/broadband/electrode_names']
print('yee=', yee)
# print('yee=', yee.keys())
# for i in range(96):
    # print('yee=', yee[i])
