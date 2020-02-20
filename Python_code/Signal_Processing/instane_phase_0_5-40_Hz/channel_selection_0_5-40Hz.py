# -*- coding: utf-8 -*-
import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.signal import hilbert
import seaborn as sns


#https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
# Low pass
def butter_lowpass(cutoff, fs, order=4):
    nyq_freq = 0.5 * fs
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, fs, order=4):

    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# High pass
def butter_highpass(cutoff, fs, order=4):
    nyq_freq = 0.5 * fs
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='highpass')
    return b, a

def butter_highpass_filter(data, cutoff_freq, fs, order=4):
    b, a = butter_highpass(cutoff_freq, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Band pass
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#https://stackoverflow.com/questions/13728392/moving-average-or-running-mean?answertab=votes
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# Read data and plot raw waveform
channel_number=31
start_second=309
plot_time_duration=3
end_second=start_second+plot_time_duration

band_start=0.5
band_cutoff=40
session_name='indy_20161007_02'

# 'pos' or 'vel'
kinematic_variable_type='vel'

# start_second & end_second loop control
for i in range(100):


    # nwb file
    nwb_filename = '../../../Dataset/The_nwb_Raw_Dataset/'+session_name+'.nwb'
    nwb_file = h5py.File(nwb_filename, 'r')

    data = nwb_file['/acquisition/timeseries/broadband/data']
    conversion = data.attrs['conversion']
    electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']
    nwb_timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']
    # print('print all nwb_file keys: ',end='')
    # print( list( nwb_file.keys() ) )



    # print('shape of data ', data.shape, '\n') # (12695457, 96) in indy_20160624_03
    #print('shape of conversion', conversion.data, '\n')
    # print('shape of electrode_map ', electrode_map.shape, '\n') # (96, 3) in indy_20160624_03
    # print('shape of nwb_timestamp ', nwb_timestamp.shape, '\n') #  (12695457,) in indy_20160624_03
    # print('first of nwb_timestamp= ', nwb_timestamp[0,],'\n')

   

    # Extract time interval from nwb file
    nwb_time_interval=np.where(np.logical_and(nwb_timestamp[:,]>start_second, nwb_timestamp[:,]<end_second ) )

    # 出圖比例
    my_plot_width=30
    my_plot_height=50
    figure_path='../../../Figures/Raw_data_and_Spike/instantaneous_phase/channel_selection/0_5-40Hz/'
    my_fontsize=30

    sampling_rate=1/( nwb_timestamp[1,]- nwb_timestamp[0,])
    fig=plt.figure(1, figsize=(my_plot_width, my_plot_height) )    

    plt.title(session_name + ' signal from '+ str(band_start) +'Hz to '+ str(band_cutoff) + 'Hz', fontsize=30, color="black")
    result=[]
    
    for channel_number_yee in range(96):
        channel_1=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], 0+channel_number_yee]
        # filtered_data_1=butter_highpass_filter(channel_1, band_start, sampling_rate, order=5)
        # filtered_data_1=butter_lowpass_filter(filtered_data_1, band_cutoff, sampling_rate, order=5)
        filtered_data_1=butter_bandpass_filter(channel_1, band_start, band_cutoff, sampling_rate, order=2) # order more than 3 doesn't work
        analytic_signal_1 = hilbert(filtered_data_1)
        instantaneous_phase_1 = np.angle(analytic_signal_1)
        result.append(instantaneous_phase_1)

    result=np.array(result)
    print('result shape = ', result.shape, '\n')

    sns.set()
    ax = sns.heatmap(result, xticklabels=False,  yticklabels=True, cbar=False, cmap='seismic')

    result=None

    # plt.colorbar(ax.get_children()[0], orientation="horizontal", size=0.5)
    # plt.xlim(start_second, end_second)

    # plt.xticks([], [])

    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values

    plt.yticks(fontsize=my_fontsize, rotation=20)

    plt.ylabel('Channels', fontsize=my_fontsize, color="black")

  

    plt.tight_layout()

    # plt.show()

    plt.savefig(figure_path+'channel_selection_'+ str(start_second)  +'_to_'+str(end_second) + '.png')

    start_second+=plot_time_duration
    end_second+=plot_time_duration

    plt.clf()
    plt.cla()
    plt.close()