# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

def histc( X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

session_name='Chewie_10032013' # Chewie_10032013, Chewie_12192013,indy_20160407_02

annots = loadmat('../../Dataset/Chewie/'+ session_name  +'.mat')
print(annots.keys()) # out_struct

targets_corner = annots['out_struct']['targets'][0][0][0][0][0]
targets_rotation = annots['out_struct']['targets'][0][0][0][0][1]
pos = annots['out_struct']['pos'][0][0]
vel = annots['out_struct']['vel'][0][0]
acc = annots['out_struct']['acc'][0][0]

time_stamp = pos[:,0]
down_sampling_index=64
down_sampling_pos = pos[::down_sampling_index][:,1:]
down_sampling_vel = vel[::down_sampling_index][:,1:]
down_sampling_acc = acc[::down_sampling_index][:,1:]
time_stamp_64ms = time_stamp[::down_sampling_index]
print('time_stamp_64ms shape = ', time_stamp_64ms.shape, '\n')
print('down_sampling_pos shape = ', down_sampling_pos.shape, '\n')
print('down_sampling_vel shape = ', down_sampling_vel.shape, '\n')
print('down_sampling_acc shape = ', down_sampling_acc.shape, '\n')
print(time_stamp_64ms)

units = annots['out_struct']['units'][0][0] # 1x174
total_unit_numbers = units.shape[1]
print('\ntotal_unit_numbers= ', total_unit_numbers, '\n')

print(pos.shape)
print(vel.shape)
print(acc.shape)
print(units.shape)

print('-'*30)

print(targets_corner.shape, '\n')
print(targets_rotation.shape, '\n')

print('-'*30)
unit_no = 52 # for 1 to total_unit_numbers
# print('unit No. '+str(unit_no+1)+' spike train = ', units[0][unit_no][1], ', shape= ', units[0][unit_no][1].shape, '\n')
# print('unit full ID: ' , units[0][unit_no][0], ', shape = ', units[0][unit_no][0].shape, '\n' )

print('-'*30)
firing_rate_cell=[[]]   
for i in range(total_unit_numbers):
    temp = units[0][i][1]
    yee = histc(temp, time_stamp_64ms)
    firing_rate_cell.append(yee[:-1])
    firing_rate_cell.append([])

firing_rate_final=[] # not[[]]
for row_index in range( len( firing_rate_cell) ):   
    if len(firing_rate_cell[row_index]):
        firing_rate_final.append( firing_rate_cell[row_index] )

# print('firing_rate_final= ', firing_rate_final, '\n')

cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.1 }
f, (ax) = plt.subplots(figsize = (16, 9),nrows=1)

sns.set()
ax = sns.heatmap(firing_rate_final , vmax=5, cbar_kws=cbar_kws_attention)

plt.show()
