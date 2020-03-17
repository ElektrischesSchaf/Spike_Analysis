from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np

class butterworth_filter():

    def butter_lowpass(self, cutoff, fs, order=2):
        nyq_freq = 0.5 * fs
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, fs, order=2):

        b, a = self.butter_lowpass(cutoff_freq, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    # High pass
    def butter_highpass(self, cutoff, fs, order=2):
        nyq_freq = 0.5 * fs
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='highpass')
        return b, a

    def butter_highpass_filter(self, data, cutoff_freq, fs, order=2):
        b, a = self.butter_highpass(cutoff_freq, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    # Band pass
    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    #https://stackoverflow.com/questions/13728392/moving-average-or-running-mean?answertab=votes
    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)