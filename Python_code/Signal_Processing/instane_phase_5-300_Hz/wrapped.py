# -*- coding: utf-8 -*-
import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.signal import hilbert

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
channel_number=49
start_second=309
plot_time_duration=3
end_second=start_second+plot_time_duration

band_start=5
band_cutoff=300

# start_second & end_second loop control
for i in range(100):


    # nwb file
    nwb_filename = '../../../Dataset/The_nwb_Raw_Dataset/indy_20161007_02.nwb'
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
    mat_file_name_1='../../../Dataset/Sorted_Spike_Dataset/indy_20161007_02.mat'
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
    # new timestamp in mat file
    new_mat_time_stamp=mat_timestamp[0,mat_time_interval[0][0]:mat_time_interval[0][-1]]
    print('new_mat_time_stamp = ', new_mat_time_stamp,'\n')
    # finger position in mat file
    numpy_finger_pos=mat_file.get('finger_pos')
    finger_z_coor=numpy_finger_pos[0][:]
    finger_z_coor=finger_z_coor[mat_time_interval[0][0]:mat_time_interval[0][-1]]
    finger_x_coor=numpy_finger_pos[1][:]
    finger_x_coor=finger_x_coor[mat_time_interval[0][0]:mat_time_interval[0][-1]]
    finger_y_coor=numpy_finger_pos[2][:]
    finger_y_coor=finger_y_coor[mat_time_interval[0][0]:mat_time_interval[0][-1]]
    # sourted spikes in mat file
    spikes = mat_file['spikes']
    temp_spike_cell_1=mat_file[ ( spikes[0][channel_number] ) ][()]
    temp_spike_cell_2=mat_file[ ( spikes[1][channel_number] ) ][()]
    temp_spike_cell_3=mat_file[ ( spikes[2][channel_number] ) ][()]
    temp_spike_cell_4=mat_file[ ( spikes[3][channel_number] ) ][()]

    #print('shape of temp_spike_cell_1 = ', temp_spike_cell_1.shape, '\n')
    #print('temp_spike_cell_1 = ', temp_spike_cell_1, '\n')

    spike_cell_1_interval=np.where(np.logical_and( temp_spike_cell_1[0,:]>start_second, temp_spike_cell_1[0,:]<end_second ))
    print('spike_cell_1_interval = ', spike_cell_1_interval, '\n')
    if spike_cell_1_interval[0]!=[]:
        temp_spike_cell_1=temp_spike_cell_1[0, spike_cell_1_interval[0][0]:spike_cell_1_interval[0][-1]]

    spike_cell_2_interval=np.where(np.logical_and( temp_spike_cell_2[0,:]>start_second, temp_spike_cell_2[0,:]<end_second ))
    print('spike_cell_2_interval = ', spike_cell_2_interval, '\n')
    if spike_cell_2_interval[0]!=[]:
        temp_spike_cell_2=temp_spike_cell_2[0, spike_cell_2_interval[0][0]:spike_cell_2_interval[0][-1]]

    spike_cell_3_interval=np.where(np.logical_and( temp_spike_cell_3[0,:]>start_second, temp_spike_cell_3[0,:]<end_second ))
    print('spike_cell_3_interval = ', spike_cell_3_interval, '\n')
    if spike_cell_3_interval[0]!=[]:
        temp_spike_cell_3=temp_spike_cell_3[0, spike_cell_3_interval[0][0]:spike_cell_3_interval[0][-1]]

    spike_cell_4_interval=np.where(np.logical_and( temp_spike_cell_4[0,:]>start_second, temp_spike_cell_4[0,:]<end_second ))
    print('spike_cell_4_interval = ', spike_cell_4_interval, '\n')
    if spike_cell_4_interval[0]!=[]:
        temp_spike_cell_4=temp_spike_cell_4[0, spike_cell_4_interval[0][0]:spike_cell_4_interval[0][-1]]

    print('temp_spike_cell_1 = ', temp_spike_cell_1, '\n')
    print('temp_spike_cell_2 = ', temp_spike_cell_2, '\n')
    print('temp_spike_cell_3 = ', temp_spike_cell_3, '\n')
    print('temp_spike_cell_4 = ', temp_spike_cell_4, '\n')

    # Extract time interval from nwb file
    nwb_time_interval=np.where(np.logical_and(nwb_timestamp[:,]>start_second, nwb_timestamp[:,]<end_second ) )
    print('nwb_timestamp np.where result = ', end='')
    print(nwb_time_interval, '\n')
    print('type of nwb_time_interval = ', type(nwb_time_interval),'\n')
    print('nwb_time_interval start time index = ', nwb_time_interval[0][0],'\n')
    print('nwb_time_interval end time index = ', nwb_time_interval[0][-1],'\n')

    new_nwb_time_stamp= nwb_timestamp[nwb_time_interval[0][0]:nwb_time_interval[0][-1],]
    new_data=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], 0+channel_number]

    print('new_nwb_time_stamp = ', new_nwb_time_stamp,'\n')

    #plt.scatter(nwb_timestamp[:100000], data[:100000, 0])
    #plt.show()

    # 出圖比例
    my_plot_width=29
    my_plot_height=7
    figure_path='../../../Figures/Raw_data_and_Spike/instantaneous_phase/wrapped_in_band_5-300Hz/'
    # figure_path='../../Figures/Raw_data_and_Spike/instantaneous_phase/unwrapped/'

    plt.figure(figsize=(my_plot_width, my_plot_height))
    plt.scatter(new_nwb_time_stamp, new_data, s=1, color= 'black')
    plt.title("indy_20161007_02 raw record in Channel "+ str(channel_number+1),fontsize=30, color="black")
    plt.xlabel("Time (s)", fontsize=25, color="black")
    plt.ylabel("Amp. (mV)", fontsize=25, color="black")

    plt.xlim(start_second, end_second)
    plt.xticks(fontsize=20, color="black")
    plt.yticks(fontsize=20, color="black")
    #plt.show()
    #plt.savefig(figure_path+'Raw_data_on_Channel_' + str(channel_number+1) + '.png')

    plt.clf()
    plt.cla()
    plt.close()


    sampling_rate=1/( nwb_timestamp[1,]- nwb_timestamp[0,])
    print('sampling rate= ',sampling_rate, '\n')

    plt.figure(figsize=(my_plot_width, my_plot_height))
    plt.gca().invert_yaxis()

    spike_signal=butter_bandpass_filter(new_data, 500, 5000, sampling_rate, order=3)
    plt.plot(new_nwb_time_stamp, spike_signal, color= 'black', zorder=2, linewidth=0.5)

    spike_line_width=4
    spike_line_length=18
    spike_line_offlet=150
    plt.eventplot(temp_spike_cell_1, color='red', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet-2*spike_line_length, linestyles='dotted')
    plt.eventplot(temp_spike_cell_2, color='blue', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet, linestyles='dotted')
    plt.eventplot(temp_spike_cell_3, color='green', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet-1*spike_line_length, linestyles='dotted')
    plt.eventplot(temp_spike_cell_4, color='yellow', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet-3*spike_line_length, linestyles='dotted')

    plt.title("indy_20161007_02 Spike Signal (500Hz-5000Hz) in Channel "+ str(channel_number+1), fontsize=10, color="black")

    plt.xlabel("Time (s)", fontsize=25, color="black")
    plt.ylabel("Amp. (mV)", fontsize=25, color="black")

    plt.xlim(start_second, end_second)
    plt.ylim(0, 200)
    plt.xticks(fontsize=20, color="black")
    plt.yticks(fontsize=20, color="black")
    #plt.show()
    #plt.savefig(figure_path+'Filtered_raw_data_and_spike_label_on_Channel_' + str(channel_number+1) + '.png')

    plt.clf()
    plt.cla()
    plt.close()

    # Combining the above two into one figure

    plt.figure(1, figsize=(my_plot_width, my_plot_height*1.5) )

    plt.subplot(411)
    plt.scatter(new_nwb_time_stamp, new_data, s=1, color= 'black')
    plt.title("indy_20161007_02 raw record in Channel "+ str(channel_number+1), fontsize=10, color="black")

    #plt.xlabel("Time (s)", fontsize=10, color="black")
    plt.ylabel("Amp. (mV)", fontsize=10, color="black")

    plt.xlim(start_second, end_second)
    #plt.xticks(fontsize=20, color="black")
    plt.xticks([], [])
    plt.yticks(fontsize=10, color="black")

    plt.subplot(412)

    filtered_data=butter_bandpass_filter(new_data, band_start, band_cutoff, sampling_rate, order=3)
    
    analytic_signal = hilbert(filtered_data)

    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_phase = np.angle(analytic_signal)

    plt.scatter(new_nwb_time_stamp, instantaneous_phase, s=1, c='black')
    plt.xlim(start_second, end_second)
    #plt.ylim(np.pi, -np.pi)

    plt.xticks([], [])

    # tick_pos= [0, np.pi , 2*np.pi, -np.pi, -2*np.pi]
    # labels = ['0', '$\pi$', '$2\pi$', '$-\pi$', '$-2\pi$']

    tick_pos= [0, np.pi , -np.pi]
    labels = ['0', '$\pi$', '$-\pi$']
    plt.yticks(tick_pos, labels)

    plt.ylabel('Wrapped Phase', fontsize=10, color="black")

    plt.subplot(413)
    plt.gca().invert_yaxis()

    spike_signal=butter_bandpass_filter(new_data, 500, 5000, sampling_rate, order=3)
    plt.plot(new_nwb_time_stamp, spike_signal, color= 'black', zorder=2, linewidth=0.5)

    spike_line_width=4
    spike_line_length=18
    spike_line_offlet=190
    plt.eventplot(temp_spike_cell_1, color='red', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet-2*spike_line_length, linestyles='solid')
    plt.eventplot(temp_spike_cell_2, color='blue', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet, linestyles='solid')
    plt.eventplot(temp_spike_cell_3, color='green', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet-1*spike_line_length, linestyles='solid')
    plt.eventplot(temp_spike_cell_4, color='yellow', linewidths=spike_line_width, linelengths=spike_line_length, lineoffsets=spike_line_offlet-3*spike_line_length, linestyles='solid')

    #plt.title("indy_20161007_02 Spike Signal (500Hz-5000Hz) in Channel "+ str(channel_number+1),fontsize=30, color="black")

    #plt.xlabel("Time (s)", fontsize=10, color="black")
    plt.xticks([], [])
    plt.ylabel("Amp. (mV)", fontsize=10, color="black")

    plt.xlim(start_second, end_second)
    plt.ylim(0, 200)
    plt.xticks(fontsize=10, color="black")
    plt.yticks(fontsize=10, color="black")

    plt.subplot(414)
    #print('\nin subplot 414, new_nwb_time_stamp.shape=', new_nwb_time_stamp.shape, ' finger_x_coor.shape=', finger_x_coor.shape, '\n')
    x_pos=plt.scatter(new_mat_time_stamp, finger_x_coor, s=5, c='blue')
    y_pos=plt.scatter(new_mat_time_stamp, finger_y_coor, s=5, c='green')
    z_pos=plt.scatter(new_mat_time_stamp, finger_z_coor, s=5, c='orange')
    plt.xlim(start_second, end_second)
    plt.legend((x_pos, y_pos, z_pos), ('x', 'y', 'z'),loc='lower left')
    plt.xlabel("Time (second)", fontsize=10, color="black")
    plt.ylabel("Position (cm)", fontsize=10, color="black")

    # plt.show()
    plt.savefig(figure_path+'raw-data_vs_instantaneous-phase_vs_spike-train_vs_position_on_Channel_' + str(channel_number+1)+'_from_'+ str(start_second)  +'_to_'+str(end_second) + '.png')

    start_second+=plot_time_duration
    end_second+=plot_time_duration

    plt.clf()
    plt.cla()
    plt.close()