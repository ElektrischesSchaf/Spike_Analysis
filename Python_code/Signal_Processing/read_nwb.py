# -*- coding: utf-8 -*-
import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
# Low pass
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# High pass
def butter_highpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='highpass')
    return b, a

def butter_highpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_highpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Band pass
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#https://stackoverflow.com/questions/13728392/moving-average-or-running-mean?answertab=votes
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# Read data and plot raw waveform
channel_number=50
start_second=311
end_second=317

# nwb file
nwb_filename = '../../Dataset/The_nwb_Raw_Dataset/indy_20161007_02.nwb'
nwb_file = h5py.File(nwb_filename, 'r')

data = nwb_file['/acquisition/timeseries/broadband/data']
conversion = data.attrs['conversion']
electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']
nwb_timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']
print('print all nwb_file keys: ',end='')
print( list( nwb_file.keys() ) )



print('shape of data ', data.shape, '\n') # (12695457, 96) in indy_20160624_03
#print('shape of conversion', conversion.data, '\n')
print('shape of electrode_map ', electrode_map.shape, '\n') # (96, 3) in indy_20160624_03
print('shape of nwb_timestamp ', nwb_timestamp.shape, '\n') #  (12695457,) in indy_20160624_03
print('first of nwb_timestamp= ', nwb_timestamp[0,],'\n')

# mat file
mat_file_name_1='../../Dataset/Sorted_Spike_Dataset/indy_20161007_02.mat'
mat_file=h5py.File(mat_file_name_1, 'r')
mat_timestamp=mat_file.get('t')
mat_timestamp=np.array(mat_timestamp)
print('shape of mat_timestamp', mat_timestamp.shape, '\n')
mat_time_interval=np.where(np.logical_and(mat_timestamp[0,:]>start_second, mat_timestamp[0,:]<end_second ) )
print('mat_timestamp np.where result = ', end='')
print(mat_time_interval, '\n')
print('type of mat_time_interval=', type(mat_time_interval),'\n')
print('mat_time_interval start time index = ', mat_time_interval[0][0],'\n')
print('mat_time_interval end time index = ', mat_time_interval[0][-1],'\n')
new_mat_time_stamp=mat_timestamp[0,mat_time_interval[0][0]:mat_time_interval[0][-1]]

print('new_mat_time_stamp = ', new_mat_time_stamp,'\n')


spikes = mat_file['spikes']
temp_spike_cell_1=mat_file[ ( spikes[0][channel_number] ) ][()]
temp_spike_cell_2=mat_file[ ( spikes[1][channel_number] ) ][()]
temp_spike_cell_3=mat_file[ ( spikes[2][channel_number] ) ][()]

#print('shape of temp_spike_cell_1 = ', temp_spike_cell_1.shape, '\n')
#print('temp_spike_cell_1 = ', temp_spike_cell_1, '\n')

spike_cell_1_interval=np.where(np.logical_and( temp_spike_cell_1[0,:]>start_second, temp_spike_cell_1[0,:]<end_second ))
print('spike_cell_1_interval = ', spike_cell_1_interval, '\n')
temp_spike_cell_1=temp_spike_cell_1[0, spike_cell_1_interval[0][0]:spike_cell_1_interval[0][-1]]

spike_cell_2_interval=np.where(np.logical_and( temp_spike_cell_2[0,:]>start_second, temp_spike_cell_2[0,:]<end_second ))
print('spike_cell_2_interval = ', spike_cell_2_interval, '\n')
temp_spike_cell_2=temp_spike_cell_2[0, spike_cell_2_interval[0][0]:spike_cell_2_interval[0][-1]]

#spike_cell_3_interval=np.where(np.logical_and( temp_spike_cell_3[0,:]>start_second, temp_spike_cell_3[0,:]<end_second ))
#print('spike_cell_3_interval = ', spike_cell_3_interval, '\n')
#temp_spike_cell_3=temp_spike_cell_3[0, spike_cell_3_interval[0][0]:spike_cell_3_interval[0][-1]]

print('temp_spike_cell_1 = ', temp_spike_cell_1, '\n')
print('temp_spike_cell_2 = ', temp_spike_cell_2, '\n')


# Extract time interval from nwb file
nwb_time_interval=np.where(np.logical_and(nwb_timestamp[:,]>start_second, nwb_timestamp[:,]<end_second ) )
print('nwb_timestamp np.where result = ', end='')
print(nwb_time_interval, '\n')
print('type of nwb_time_interval = ', type(nwb_time_interval),'\n')
print('nwb_time_interval start time index = ', nwb_time_interval[0][0],'\n')
print('nwb_time_interval end time index = ', nwb_time_interval[0][-1],'\n')

new_nwb_time_stamp= nwb_timestamp[nwb_time_interval[0][0]:nwb_time_interval[0][-1],]
new_data=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1],0+channel_number]

print('new_nwb_time_stamp = ', new_nwb_time_stamp,'\n')

#plt.scatter(nwb_timestamp[:100000], data[:100000, 0])
#plt.show()


plt.scatter(new_nwb_time_stamp, new_data, s=1, color= 'black')
plt.title("indy_20161007_02 raw record in Channel "+ str(channel_number+1),fontsize=30, color="black")
plt.xlabel("Time (s)", fontsize=25, color="black")
plt.ylabel("Amp. (mV)", fontsize=25, color="black")
plt.xlim(start_second, end_second)
plt.xticks(fontsize=20, color="black")
plt.yticks(fontsize=20, color="black")
plt.show()

sampling_rate=1/( nwb_timestamp[1,]- nwb_timestamp[0,])
print('sampling rate= ',sampling_rate, '\n')
#butterworth filter
cutoff_frequency = 300.0
# Downsample data
new_sample_rate = 1000

plt.close()
plt.clf()

plt.figure(figsize=(29,7))
spike_signal=butter_bandpass_filter(new_data, 500, 5000, sampling_rate, order=4)
plt.plot(new_nwb_time_stamp, spike_signal, color= 'black', zorder=2, linewidth=0.5)
plt.eventplot(temp_spike_cell_1, color='red', linelengths=20, lineoffsets=100)
plt.eventplot(temp_spike_cell_2, color='blue', linelengths=20, lineoffsets=90)
plt.eventplot(temp_spike_cell_3, color='red', linelengths=10)
plt.title("indy_20161007_02 Spike Signal (500Hz-5000Hz) in Channel "+ str(channel_number+1),fontsize=30, color="black")
plt.xlabel("Time (s)", fontsize=25, color="black")
plt.ylabel("Amp. (mV)", fontsize=25, color="black")
plt.xlim(start_second, end_second)
plt.ylim(120,-120)
plt.xticks(fontsize=20, color="black")
plt.yticks(fontsize=20, color="black")
#plt.show()
plt.savefig('Filtered_raw_data_and_spike_label_on_Channel_' + str(channel_number+1) + '.png')
plt.close()
plt.clf()


'''

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

'''