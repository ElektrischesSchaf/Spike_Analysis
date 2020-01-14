# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/52179919/amplitude-and-phase-spectrum-shifting-the-phase-leaving-amplitude-untouched
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
import h5py
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.signal import hilbert, chirp

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

import matplotlib.pyplot as plt
import numpy as np
import cmath

# Generate a model signal
t0 = 1250.0
#dt = 0.152
#freq = (1./dt)/128

dt =nwb_timestamp[1,]-nwb_timestamp[0,]
print('dt=',dt,'\n')

#t = np.linspace( t0, t0+1024*dt, 1024, endpoint=False )
#signal = np.sin( t*(2*np.pi)*freq )

signal=data[:1000,0]

print('shape of signal', signal.shape)
## Fourier transform of real valued signal
signalFFT = np.fft.rfft(signal)

## Get Power Spectral Density
signalPSD = np.abs(signalFFT) ** 2
signalPSD /= len(signalFFT)**2

## Get Phase
signalPhase = np.angle(signalFFT)

## Phase Shift the signal +90 degrees
newSignalFFT = signalFFT * cmath.rect( 1., np.pi/2 )

## Reverse Fourier transform
newSignal = np.fft.irfft(newSignalFFT)

## Uncomment this line to restore the original baseline
# newSignal += signalFFT[0].real/len(signal)


# And now, the graphics -------------------

## Get frequencies corresponding to signal 
fftFreq = np.fft.rfftfreq(len(signal), dt)

plt.figure( figsize=(10, 4) )

ax1 = plt.subplot( 1, 2, 1 )
ax1.plot( nwb_timestamp[:1000], signal, label='signal')
ax1.plot( nwb_timestamp[:1000], newSignal, label='new signal')
ax1.set_ylabel( 'Signal' )
ax1.set_xlabel( 'time' )
ax1.legend()

ax2 = plt.subplot( 1, 2, 2 )
ax2.plot( fftFreq, signalPSD )
ax2.set_ylabel( 'Power' )
ax2.set_xlabel( 'frequency' )

ax2b = ax2.twinx()
ax2b.plot( fftFreq, signalPhase, alpha=0.25, color='r' )
ax2b.set_ylabel( 'Phase', color='r' )


plt.tight_layout()

#plt.show()
plt.clf()
plt.close()


plt.figure( figsize=(10, 4) )

fs=(1./dt)
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
#instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_phase = np.angle(analytic_signal)
#instantaneous_frequency = (np.diff(instantaneous_phase) /  (2.0*np.pi) * fs)
instantaneous_frequency = (np.diff(instantaneous_phase) /  (2.0*np.pi) )
fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(nwb_timestamp[:1000], signal, label='signal')
#ax0.plot(nwb_timestamp[:1000], amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
#ax1.plot(t[1:], instantaneous_frequency)
ax1.plot(nwb_timestamp[:1000], instantaneous_phase)
ax1.set_xlabel("time in seconds")
#ax1.set_ylim(0.0, 120.0)

plt.show()