import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.signal import hilbert, chirp
import seaborn as sns
# Generate a model signal
t0 = 1250.0
dt = 0.152
freq = (1./dt)/128

t = np.linspace( t0, t0+1024*dt, 1024, endpoint=False )
signal = np.sin( t*(2*np.pi)*freq )

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
ax1.plot( t, signal, label='signal')
ax1.plot( t, newSignal, label='new signal')
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
my_font_size=20

fs=(1./dt)/128
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
#instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_phase = np.angle(analytic_signal)
#instantaneous_frequency = (np.diff(instantaneous_phase) /  (2.0*np.pi) * fs)
instantaneous_frequency = (np.diff(instantaneous_phase) /  (2.0*np.pi) )
fig = plt.figure()
ax0 = fig.add_subplot(311)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
plt.xlim(t[0:][0],t[0:][-1])
plt.ylabel('Origianl Signal', fontsize=my_font_size, color='black')
ax0.legend()
ax1 = fig.add_subplot(312)
#ax1.plot(t[1:], instantaneous_frequency)
ax1.plot(t[0:], instantaneous_phase)
ax1.set_xlabel("time in seconds")
plt.xlim(t[0:][0],t[0:][-1])
#ax1.set_ylim(0.0, 120.0)
tick_pos= [0, np.pi , -np.pi]
labels = ['0', '$\pi$', '$-\pi$']
plt.yticks(tick_pos, labels)

plt.ylabel('Wrapped Phase', fontsize=my_font_size, color='black')

ax3 = fig.add_subplot(313)
sns.set()
instantaneous_phase=np.reshape(instantaneous_phase,(1,-1))
ax = sns.heatmap(instantaneous_phase, xticklabels=False, yticklabels=False, cbar=False, cmap='seismic')
plt.colorbar(ax.get_children()[0], orientation="horizontal")
plt.ylabel('Phase Heatmap', fontsize=my_font_size, color='black')


plt.show()