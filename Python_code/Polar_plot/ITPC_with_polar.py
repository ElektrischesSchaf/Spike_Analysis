# -*- coding: utf-8 -*-
import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.signal import hilbert
import seaborn as sns

import statistics
import math


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
start_second=300
plot_time_duration=5
end_second=start_second+plot_time_duration

band_start=0.5
band_cutoff=40
session_name='indy_20161007_02'

# 'pos' or 'vel'
kinematic_variable_type='vel'


# start_second & end_second loop control
for i in range(100):


    # nwb file
    nwb_filename = '../../Dataset/The_nwb_Raw_Dataset/'+session_name+'.nwb'
    nwb_file = h5py.File(nwb_filename, 'r')

    data = nwb_file['/acquisition/timeseries/broadband/data']
    conversion = data.attrs['conversion']
    electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']
    nwb_timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']


    # Extract time interval from nwb file
    nwb_time_interval=np.where(np.logical_and(nwb_timestamp[:,]>start_second, nwb_timestamp[:,]<end_second ) )
    print(nwb_time_interval, '\n')
    new_nwb_time_stamp= nwb_timestamp[nwb_time_interval[0][0]:nwb_time_interval[0][-1],]

    # 出圖比例
    my_plot_width=29
    my_plot_height=7
    figure_path='../../../Figures/Raw_data_and_Spike/instantaneous_phase/ITPC-angle/ITPC_0_5-40Hz/11/'
    my_fontsize=30

    sampling_rate=1/( nwb_timestamp[1,]- nwb_timestamp[0,])

    instance_phase_all_channels=[]

    # good_channel_list_start_from_one=[39,41,76,42,26,29,33,93,77,58,54]
    # for channel_number_yee in good_channel_list_start_from_one:
    #     channel_number_yee=channel_number_yee-1
    #     channel_1=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], channel_number_yee]        
    #     filtered_data_1=butter_highpass_filter(channel_1, band_start, sampling_rate, order=3) # must order 3
    #     filtered_data_1=butter_lowpass_filter(filtered_data_1, band_cutoff, sampling_rate, order=3) # must order 3
    #     analytic_signal = hilbert(filtered_data_1)
    #     instantaneous_phase = np.angle(analytic_signal)
    #     instance_phase_all_channels.append(instantaneous_phase)

    
    # bad_channels=[15,19,46,57,58,59,60,64,65,68,70,77,78,8,81,93,94]
    bad_channels=[]
    for channel_number_yee in range(96):
        if channel_number_yee not in bad_channels:
            channel_1=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], channel_number_yee]
            # filtered_data_1=butter_bandpass_filter(channel_1, band_start, band_cutoff, sampling_rate, order=3) # must order 3
            filtered_data_1=butter_highpass_filter(channel_1, band_start, sampling_rate, order=3) # must order 3
            filtered_data_1=butter_lowpass_filter(filtered_data_1, band_cutoff, sampling_rate, order=3) # must order 3
            analytic_signal = hilbert(filtered_data_1)
            instantaneous_phase = np.angle(analytic_signal)

            # plt.scatter(new_nwb_time_stamp, instantaneous_phase, s=0.1, c='black')
            
            instance_phase_all_channels.append(instantaneous_phase)


    instance_phase_all_channels=np.array(instance_phase_all_channels)
    ITPC=[]

    for itpc_loop in range( instance_phase_all_channels.shape[1] ) :

        # Start the polar plot

        # radar green, solid grid lines
        plt.rc('grid', color='#316931', linewidth=1, linestyle='-')
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=0)

        # force square figure and square axes looks better for polar, IMO
        width, height = matplotlib.rcParams['figure.figsize']
        size = min(width, height)
        # make a square figure
        # fig = plt.figure(figsize=(size, size))
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.7], polar=True, facecolor='#d5de9c') # ax1 = fig.add_axes([left, bottom, width, height])

        # r = np.arange(0, 3.0, 0.01)
        # theta = 2*np.pi*r
        # ax.plot(theta, r, color='#ee8d18', lw=3)
        # ax.set_rmax(1.0)

        # plt.grid(True)


        itpc_angle=0
        itpc_abs=0

        for i in range(len( instance_phase_all_channels[:,itpc_loop] )):
            ax.set_title(session_name+'\nt= ' + str( new_nwb_time_stamp[i] ), fontsize=20)  
            
            plt.arrow( instance_phase_all_channels[:,itpc_loop][i], 0, 0, 1, alpha = 0.3, width = 0.005,  edgecolor = 'blue', facecolor = 'blue', lw = 1, zorder = 0)

        itpc_angle=np.angle( np.mean (np.exp( 1j * instance_phase_all_channels[:,itpc_loop]  )))        
        itpc_abs=np.abs( np.mean (np.exp( 1j * instance_phase_all_channels[:,itpc_loop]  )))
        print('itpc_angle:', itpc_angle,'  ',' itpc_abs:', itpc_abs, '\n' )
        ITPC.append(itpc_angle)

        #This is the line I added:
        plt.arrow( itpc_angle, 0, 0, itpc_abs, alpha = None, width = 0.015,  edgecolor = 'red', facecolor = 'red', lw = 4, zorder = 50)
        plt.show()
        plt.close()

    # End the polar plot

    start_second+=plot_time_duration
    end_second+=plot_time_duration
    del ITPC
    del instance_phase_all_channels
