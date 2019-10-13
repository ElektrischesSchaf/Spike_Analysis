# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:16:18 2019

@author: KimUyen
"""

import h5py
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='highpass')
    return b, a

def butter_highpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_highpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

#https://stackoverflow.com/questions/13728392/moving-average-or-running-mean?answertab=votes
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
   
# Read data
filename = '../../Dataset/The_nwb_Raw_Dataset/indy_20160624_03.nwb'
nwb_file = h5py.File(filename, 'r')
data = nwb_file['/acquisition/timeseries/broadband/data']
conversion = data.attrs['conversion']

electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']

timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']

print('print all nwb_file keys: ',end='')
print( list( nwb_file.keys() ) )

print('shape of data ', data.shape, '\n') # (12695457, 96) in indy_20160624_03
#print('shape of conversion', conversion.data, '\n')
print('shape of electrode_map ', electrode_map.shape, '\n') # (96, 3) in indy_20160624_03
print('shape of timestamp ', timestamp.shape, '\n') #  (12695457,) in indy_20160624_03

plt.scatter(timestamp[:100000], data[:100000, 0])
plt.show()

#####################
sample_rate = 24400
duration = 10
start = 0
channel = 96
signal_length = (int) (duration * sample_rate)
################################

#butterworth filter
cutoff_frequency = 300.0
# Downsample data
new_sample_rate = 1000
new_signal_length = (int) (duration*new_sample_rate)

#preprocessing to achieve LFP
start_idx = start*sample_rate
end_idx = start*sample_rate + signal_length
pre_data = data[start_idx:end_idx]
pos_data = np.empty([new_signal_length,channel])

#preprocessing for all channels
for i in range(channel):
    low_pass = butter_lowpass_filter(pre_data[:, i], cutoff_frequency, sample_rate/2)
    down_sample = signal.resample(low_pass, new_signal_length)
    pos_data[:,i] = down_sample

car_data = np.mean(pos_data, axis=1)

# box plot for all channels

new_pos_data = np.empty([new_signal_length,channel + 1])
new_pos_data[:, 0:channel] = pos_data
new_pos_data[:, channel] = car_data


# remove outlier channels and channels have small variation
# find min and max for all channels
max_96_chans = np.max(pos_data, axis=0)
min_96_chans = np.min(pos_data, axis=0)
median_96_chans = np.median(pos_data, axis=0)
median_car =  np.median(car_data, axis=0)
diff_median = np.abs(median_96_chans - median_car)
diff_min_max = np.abs(max_96_chans-min_96_chans)
# remove outlier channels for all diff median > 1500
# remove channels have small variation for all diff_min_max < 1500
idx_to_delete_outlier = np.argwhere((diff_median > 1000))
idx_to_delete_small_vari = np.argwhere((diff_min_max < 3000))
# selected 
idx_to_selected_1 = np.argwhere((diff_median < 1000))
idx_to_selected_2 = np.argwhere((diff_min_max > 3000))
idx_to_select = np.intersect1d(idx_to_selected_1, idx_to_selected_2)

# car from selected channels
CAR_selected = np.mean(pos_data[:, idx_to_select], axis=1)
chan = 36
plt.figure(figsize=(11, 9))
j = 0
for i in idx_to_select: # range(channel):
    low_pass = butter_lowpass_filter(pos_data[:, i], 4, new_sample_rate/2)
    plt.plot(low_pass + j * 1000, label = i)
    j = j + 1
#plt.plot(CAR_selected, color='r', label = "car from selected channels")
plt.title("Selected channels to be kept")
plt.xlabel('{} Samples of {}s'.format(new_signal_length, duration))
plt.ylabel('Amplitude (dB)')
#plt.legend(idx_to_select, loc = "best")
plt.legend()
plt.show()

plt.figure(figsize=(11, 9))
plt.title("Box plot for selected channel")
plt.xlabel("Channel")
plt.ylabel("amplitude value")
plt.boxplot(new_pos_data[:, idx_to_select], labels = idx_to_select)
plt.show()