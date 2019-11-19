import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

test_sampling_duration=1000000

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


nwb_filename = '../../Dataset/The_nwb_Raw_Dataset/indy_20160624_03.nwb'
nwb_file = h5py.File(nwb_filename, 'r')
data = nwb_file['/acquisition/timeseries/broadband/data']
conversion = data.attrs['conversion']

electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']

timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']
sampling_frequency = 1 / (timestamp[1]-timestamp[0]) # 1 / Δt 
print('sampling_frequency= ', sampling_frequency, '\n')

s=data[:test_sampling_duration, selected_channel]

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

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(17, 17), dpi=100)

print('Duration= ', timestamp[test_sampling_duration]-timestamp[0],'\n')

axes[0,0].scatter(timestamp[:test_sampling_duration], s, s=1)
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

#plt.clf()
#plt.close()
#plt.magnitude_spectrum(spike_signal, Fs=sampling_frequency)
#plt.show()

plt.clf()
plt.close()

#plot spectrogram for inspection
plt.figure(figsize=(17, 17), dpi=10)
fig, c = plt.subplots()
Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, Fs=6000, noverlap=0)
print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
print('bins= ', len(bins), ' im= ', im, '\n')
print('bins[0]=', bins[0],'\n')
print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
freqs_cnn=len(freqs)
bins_cnn=len(bins)

c.set_title('Spike Spectrogram in Channel '+str(selected_channel+1))
c.set_ylim([300, 3000])
c.set_xlabel('Time (Sample)')
c.set_ylabel('Frequency')
fig.colorbar(im)

plt.show()

plt.clf()
plt.close()

#save spectrogram for CNN
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

fig, c = plt.subplots(figsize=(freqs_cnn, bins_cnn), dpi=1, frameon=False)
Pxx, freqs, bins, im = c.specgram(spike_signal, NFFT=1024, Fs=6000, noverlap=0)
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

#plot spectrogram for inspection
plt.figure(figsize=(17, 17), dpi=10)
fig, c = plt.subplots()
Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=2*1024, Fs=600, noverlap=0)
print('type of c, Pxx, freqs, bins, im  ', type(c), type(Pxx), type(freqs), type(bins), type(im) )
print('bins= ', len(bins), ' im= ', im, '\n')
print('bins[0]=', bins[0],'\n')
print('shape of freqs ', len(freqs), ' shape of bins ', len(bins) ,  '\n')
freqs_cnn=len(freqs)
bins_cnn=len(bins)

c.set_title('LFP Spectrogram in Channel '+str(selected_channel+1))
c.set_ylim([0, 300])
c.set_xlabel('Time (Sample)')
c.set_ylabel('Frequency')
fig.colorbar(im)
plt.show()

plt.clf()
plt.close()

#save spectrogram for CNN
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

fig, c = plt.subplots(figsize=(freqs_cnn, bins_cnn), dpi=1, frameon=False)
Pxx, freqs, bins, im = c.specgram(local_field_potential_signal, NFFT=2*1024, Fs=600, noverlap=0)
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