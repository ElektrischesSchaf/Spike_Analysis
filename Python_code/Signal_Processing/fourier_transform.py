import h5py
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sampling_frequency=0 # sampling frequency
selected_channel=0


filename = '../../Dataset/The_nwb_Raw_Dataset/indy_20160624_03.nwb'
nwb_file = h5py.File(filename, 'r')
data = nwb_file['/acquisition/timeseries/broadband/data']
conversion = data.attrs['conversion']

electrode_map = nwb_file['/general/extracellular_ephys/electrode_map']

timestamp = nwb_file['/acquisition/timeseries/broadband/timestamps']
sampling_frequency=1 / (timestamp[1]-timestamp[0])

s=data[:1000000, selected_channel]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(17, 17))

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

'''
plt.clf()
plt.magnitude_spectrum(s, Fs=sampling_frequency)
plt.show()
'''

plt.clf()
plt.close()

plt.figure(1)
c = plt.subplot(211)
Pxx, freqs, bins, im = c.specgram(s, NFFT=1024, Fs=sampling_frequency, noverlap=900)
c.set_title('Spectrogram in channel '+str(selected_channel+1))
c.set_xlabel('Time')
c.set_ylabel('Frequency')
plt.show()

