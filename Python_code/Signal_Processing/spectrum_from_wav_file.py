import wave
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

signal_wave = wave.open('voice.wav', 'r')
sample_frequency = 16000

data = np.fromstring(signal_wave.readframes(sample_frequency), dtype=np.int16)
sig = signal_wave.readframes(-1)

sig = np.fromstring(sig, 'Int16')

sig = sig[:]

#sig = sig[25000:32000]

plt.figure(1)
c = plt.subplot(211)
Pxx, freqs, bins, im = c.specgram(sig, NFFT=1024, Fs=16000, noverlap=900)
c.set_xlabel('Time')
c.set_ylabel('Frequency')
plt.show()