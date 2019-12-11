import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



channel_number=49
start_second=310
end_second=330



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

sampling_frequency=0 # sampling frequency


# nwb file
nwb_filename = '../../Dataset/The_nwb_Raw_Dataset/indy_20161007_02.nwb'
nwb_file = h5py.File(nwb_filename, 'r')
data = nwb_file['/acquisition/timeseries/broadband/data']
conversion = data.attrs['conversion']

electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']

nwb_timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']
sampling_frequency = 1 / (nwb_timestamp[1]-nwb_timestamp[0]) # 1 / Δt 
print('sampling_frequency= ', sampling_frequency, '\n')

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

numpy_finger_pos=mat_file.get('finger_pos')
finger_z_coor=numpy_finger_pos[0][:]
finger_z_coor=finger_z_coor[mat_time_interval[0][0]:mat_time_interval[0][-1]]
finger_x_coor=numpy_finger_pos[1][:]
finger_x_coor=finger_x_coor[mat_time_interval[0][0]:mat_time_interval[0][-1]]
finger_y_coor=numpy_finger_pos[2][:]
finger_y_coor=finger_y_coor[mat_time_interval[0][0]:mat_time_interval[0][-1]]


for selected_channel in range(0, 96):    

    # ploting with fixed duration
    test_sampling_duration=1000000
    s=data[:test_sampling_duration, selected_channel]
    spike_signal=butter_bandpass_filter(s, 300, 3000, sampling_frequency, order=5)
    local_field_potential_signal=butter_lowpass_filter(s,3000, sampling_frequency/2)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(17, 17), dpi=100)
    print('Duration= ', nwb_timestamp[test_sampling_duration]-nwb_timestamp[0],'\n')

    axes[0,0].scatter(nwb_timestamp[:test_sampling_duration], s, s=1)
    axes[0, 0].set_title('Signal')

    axes[1,0].set_title("Magnitude Spectrum")
    axes[1,0].magnitude_spectrum(s, Fs=sampling_frequency)

    axes[1, 1].set_title("Logistic Magnitude Spectrum")
    axes[1, 1].magnitude_spectrum(s, Fs=sampling_frequency, scale='dB', color='C1')

    axes[2, 0].set_title("Phase Spectrum ")
    axes[2, 0].phase_spectrum(s, Fs=sampling_frequency, color='C2')

    axes[2, 1].set_title("Angle Spectrum")
    axes[2, 1].angle_spectrum(s, Fs=sampling_frequency, color='C2')

    axes[0, 1].remove()  # don't display empty ax

    fig.tight_layout()

    #plt.show()
    plt.savefig('../../Figures/nwb_Data_Plot/spectrum_on_1_session.png')

    plt.clf()
    plt.close()
    plt.title('Magnitude Spectrum in Channel '+str(selected_channel+1))
    plt.magnitude_spectrum(spike_signal, Fs=sampling_frequency)
    #plt.show()
    plt.savefig('../../Figures/nwb_Data_Plot/magnitude_spectrum_channel_'+str(selected_channel+1)+'.png')

    plt.clf()
    plt.close()
    

    new_data=data[ nwb_time_interval[0][0]:nwb_time_interval[0][-1], selected_channel]

    my_plot_width=30
    my_plot_height=30
    new_sample_rate_spike = 6010
    new_sample_rate_LFP = 610

    spike_signal=butter_bandpass_filter(new_data, 300, 3000, new_sample_rate_spike, order=5)
    local_field_potential_signal=butter_lowpass_filter(new_data, 300, new_sample_rate_LFP/2)

    '''
    #plot power spectral density Spike spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, mode='psd', cmap='tab20c', Fs=sampling_frequency, noverlap=0) 
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)

    Pxx=Pxx[300:3000,:]
    c.set_title('Spike Power Spectral Density Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([300, 3000])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)

    
    #plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    #plt.margins(0,0)
    
    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/Spike/psd/spike_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()

    '''

    '''
    #plot phase Spike spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, mode='phase', scale='linear', cmap='tab20c', Fs=sampling_frequency, noverlap=0) 
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)

    Pxx=Pxx[300:3000,:]
    c.set_title('Spike Phase Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([300, 3000])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)


    #plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    #plt.margins(0,0)

    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/Spike/phase/spike_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()
    '''

    '''
    #plot magnitude Spike spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, mode='magnitude', scale='dB', cmap='tab20c', Fs=sampling_frequency, noverlap=0) 
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)

    Pxx=Pxx[300:3000,:]
    c.set_title('Magnitude Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([300, 3000])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)


    #plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    #plt.margins(0,0)

    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/Spike/magnitude/spike_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()
    '''

    #plot angle Spike spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, mode='angle', scale='linear', cmap='tab20c', Fs=sampling_frequency, noverlap=0) 
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)

    Pxx=Pxx[300:3000,:]
    c.set_title('Spike Angle Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([300, 3000])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)


    #plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    #plt.margins(0,0)

    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/Spike/angle/spike_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()

    '''
    #save spectrogram for CNN
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

    fig, c = plt.subplots(figsize=(freqs_cnn, bins_cnn), dpi=1, frameon=False)
    Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, Fs=sampling_frequency, noverlap=0)
    print('len of Pxx: ', len(Pxx), '\n')
    print('len of Pxx[0]=', len(Pxx[0]), '\n')
    c.set_ylim([300, 3000])

    #c.get_xaxis().set_visible(False)
    #c.get_yaxis().set_visible(False)
    c.axis('off')
    plt.axis('off')
    mpl.rcParams['savefig.pad_inches'] = 0
    #plt.box(on=None)
    plt.autoscale(tight=True)

    plt.savefig('../../Figures/nwb_Data_Plot/spike_spectrum_on_1_session.png')

    # Convert to matrix form
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    print('len(X)= ', len(X), '\n')
    print('len(X[0])', len(X[0]), '\n')

    plt.clf()
    plt.close()
    '''

    '''
    #plot power spectral density LFP spectrogram spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=1024, cmap='tab20c', mode='psd', Fs=sampling_frequency, noverlap=0)
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)
    Pxx=Pxx[:300,:]
    c.set_title('LFP Power Spectral Density Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([0, 300])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)

    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/LFP/psd/LFP_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()
    '''

    '''
    #plot phase LFP spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=1024, cmap='tab20c', mode='phase', scale='linear', Fs=sampling_frequency, noverlap=0)
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)
    Pxx=Pxx[:300,:]
    c.set_title('LFP Phase Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([0, 300])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)

    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/LFP/phase/LFP_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()
    '''

    '''
    #plot magnitude LFP spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=1024, cmap='tab20c', mode='magnitude', scale='dB', Fs=sampling_frequency, noverlap=0)
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)
    Pxx=Pxx[:300,:]
    c.set_title('LFP Magnitude Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([0, 300])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)

    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/LFP/magnitude/LFP_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()
    '''

    #plot angle LFP spectrogram for inspection
    plt.figure()
    fig, (c, pos_x, pos_y, pos_z) = plt.subplots(4, 1, sharex=False, figsize=(my_plot_width, my_plot_height), gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=1024, cmap='tab20c', mode='angle', scale='linear', Fs=sampling_frequency, noverlap=0)
    print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
    print('bins= ', len(bins), ' im= ', im, '\n')
    print('bins[0]=', bins[0],'\n')
    print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
    freqs_cnn=len(freqs)
    bins_cnn=len(bins)
    Pxx=Pxx[:300,:]
    c.set_title('LFP Angle Spectrogram in Channel '+str(selected_channel+1))
    c.set_ylim([0, 300])
    c.set_xlabel('Time (Seconds)')
    c.set_ylabel('Frequency')
    #fig.colorbar(im)

    pos_x.scatter(new_mat_time_stamp, finger_x_coor)
    pos_x.set_xlim(start_second, end_second)

    pos_y.scatter(new_mat_time_stamp, finger_y_coor)
    pos_y.set_xlim(start_second, end_second)

    pos_z.scatter(new_mat_time_stamp, finger_z_coor)
    pos_z.set_xlim(start_second, end_second)

    plt.subplots_adjust(top=0.95, bottom=0.05, left= 0.05, right=0.95, hspace=0.05, wspace=0)
    plt.savefig('../../Figures/nwb_Data_Plot/Spectrogram_all_Channels/LFP/angle/LFP_spectrum_on_'+str(selected_channel+1)+'_channel.png')
    
    plt.clf()
    plt.close()

    '''
    #save spectrogram for CNN
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

    fig, c = plt.subplots(figsize=(freqs_cnn, bins_cnn), dpi=1, frameon=False)
    Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=2*1024, Fs=sampling_frequency, noverlap=0)
    print('len of Pxx: ', len(Pxx), '\n')
    print('len of Pxx[0]=', len(Pxx[0]), '\n')
    c.set_ylim([0, 300])

    #c.get_xaxis().set_visible(False)
    #c.get_yaxis().set_visible(False)
    c.axis('off')
    plt.axis('off')
    mpl.rcParams['savefig.pad_inches'] = 0
    #plt.box(on=None)
    plt.autoscale(tight=True)


    plt.savefig('../../Figures/nwb_Data_Plot/LFP_spectrum_on_1_session.png')

    # Convert to matrix form
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    print('len(X) = ', len(X), '\n')
    print('len(X[0]) = ', len(X[0]), '\n')
    print(X[0][500])

    plt.clf()
    plt.close()
    '''