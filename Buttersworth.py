import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy import signal
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter
df = pd.read_csv('Prueba1.csv')
AF3d = df.iloc[0: ,2]
AF3 = AF3d.to_numpy()
sf = 128
samplesize = AF3.size

    
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


def run():

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 128
    lowcut = 13.0
    highcut = 30.0

    # Plot the frequency response for a few different orders.
##    plt.figure(1)
##    plt.clf()
##    for order in [3, 6, 9]:
##        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
##        w, h = freqz(b, a, worN=2000)

##    T = 0.0078
##    nsamples= T*fs
##    t = np.linspace(0,T, nsamples, endpoint=False)
##    a = 0.02
##    f0 = 8.00
    y = butter_bandpass_filter(AF3, lowcut, highcut, fs, order=6)
    #plt.plot(y)
    plt.psd(y, label= 'AF3')
    plt.show()
##    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
##    plt.xlabel('time (seconds)')
##    plt.hlines([-a, a], 0, T, linestyles='--')
##    plt.grid(True)
##    plt.axis('tight')
##    plt.legend(loc='upper left')
run()
