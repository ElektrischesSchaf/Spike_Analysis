import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
selected_channel=0


filename = '../../Dataset/The_nwb_Raw_Dataset/indy_20160624_03.nwb'
nwb_file = h5py.File(filename, 'r')
data = nwb_file['/acquisition/timeseries/broadband/data']
conversion = data.attrs['conversion']

electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']

timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']
sampling_frequency = 1 / (timestamp[1]-timestamp[0]) # 1 / Δt 
print('sampling_frequency= ', sampling_frequency, '\n')

s=data[:1000000, selected_channel]

'''
s=butter_highpass_filter(s, 300, sampling_frequency/2)
#s=butter_lowpass_filter(s, 3000, sampling_frequency/2)
'''
'''
s=butter_lowpass_filter(s,3000, sampling_frequency/2)
s=butter_highpass_filter(s, 300, sampling_frequency/2)
'''

spike_signal=butter_bandpass_filter(s, 300, 3000, sampling_frequency, order=5)
local_field_potential_signal=butter_lowpass_filter(s,3000, sampling_frequency/2)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(17, 17))

print('Duration= ', timestamp[1000000]-timestamp[0],'\n')

axes[0,0].scatter(timestamp[:1000000], s, s=1)
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
#plt.magnitude_spectrum(spike_signal, Fs=sampling_frequency)
#plt.show()

plt.clf()
plt.close()

plt.figure(figsize=(17, 17), dpi=200)
c = plt.subplot(111)
Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, Fs=6000, noverlap=0)
c.set_title('Spike Spectrogram in Channel '+str(selected_channel+1))
c.set_ylim([300, 3000])
c.set_xlabel('Time (Sample)')
c.set_ylabel('Frequency')
#plt.show()
plt.savefig('../../Figures/nwb_Data_Plot/spike_spectrum_on_1_session.png')
#width, height = plt.size()
print('bins= ', len(bins), ' im= ', im, '\n')

plt.clf()
plt.close()
#
plt.figure(figsize=(17, 17), dpi=200)
c = plt.subplot(111)
Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=2*1024, Fs=600, noverlap=0)
print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im)  )
print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
c.set_title('LFP Spectrogram in Channel '+str(selected_channel+1))
c.set_ylim([0, 300])
c.set_xlabel('Time (Sample)')
c.set_ylabel('Frequency')
#plt.show()
plt.savefig('../../Figures/nwb_Data_Plot/LFP_spectrum_on_1_session.png')

'''
# From image to matrix
im.figure.canvas.draw()
X=np.array( im.canvas.renderer.buffer_rgba() )
print('X.shape= ', X.shape(), '\n')
'''