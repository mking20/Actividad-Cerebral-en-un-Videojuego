import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy import signal
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter
import pylab

df = pd.read_csv('Prueba1.csv')

AF3d = df.iloc[0: ,2]
AF3 = AF3d.to_numpy()
F7d = df.iloc[0:,3]
F7=F7d.to_numpy()
F3d = df.iloc[0:,4]
F3=F3d.to_numpy()
FC5d = df.iloc[0:,5]
FC5 = FC5d.to_numpy()
T7d = df.iloc[0:,6]
T7 = T7d.to_numpy()
P7d = df.iloc[0:,7]
P7 = P7d.to_numpy()
OC1d = df.iloc[0:,8]
OC1 = OC1d.to_numpy()
OC2d = df.iloc[0:,9]
OC2 = OC2d.to_numpy()
P8d = df.iloc[0:,10]
P8=P8d.to_numpy()
T8d = df.iloc[0:,11]
T8=T8d.to_numpy()
FC6d = df.iloc[0:,12]
FC6=FC6d.to_numpy()
F4d = df.iloc[0:,13]
F4=F4d.to_numpy()
F8d = df.iloc[0:,14]
F8=F8d.to_numpy()
AF4d = df.iloc[0:,15]
AF4 = AF4d.to_numpy()

    
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
    lowcut = 4.0
    highcut = 8.0

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
    y1 = butter_bandpass_filter(AF3, lowcut, highcut, fs, order=6)
    y2 = butter_bandpass_filter(F7, lowcut, highcut, fs, order=6)
    y3 = butter_bandpass_filter(F3, lowcut, highcut, fs, order=6)
    y4 = butter_bandpass_filter(FC5, lowcut, highcut, fs, order=6)
    y5 = butter_bandpass_filter(T7, lowcut, highcut, fs, order=6)
    y6 = butter_bandpass_filter(P7, lowcut, highcut, fs, order=6)
    y7 = butter_bandpass_filter(OC1, lowcut, highcut, fs, order=6)
    y8 = butter_bandpass_filter(OC2, lowcut, highcut, fs, order=6)
    y9 = butter_bandpass_filter(P8, lowcut, highcut, fs, order=6)
    y10 = butter_bandpass_filter(T8, lowcut, highcut, fs, order=6)
    y11 = butter_bandpass_filter(FC6, lowcut, highcut, fs, order=6)
    y12 = butter_bandpass_filter(F4, lowcut, highcut, fs, order=6)
    y13 = butter_bandpass_filter(F8, lowcut, highcut, fs, order=6)
    y14 = butter_bandpass_filter(AF4, lowcut, highcut, fs, order=6)
    #plt.plot(y)
    plt.psd(y1, label= 'AF3')
    plt.psd(y2, label= 'F7')
    plt.psd(y1, label= 'F3')
    plt.psd(y4, label= 'FC5')
    plt.psd(y5, label= 'T7')
    plt.psd(y6, label= 'P7')
    plt.psd(y7, label= 'OC1')
    plt.psd(y8, label= 'OC2')
    plt.psd(y9, label= 'P8')
    plt.psd(y10, label= 'T8')
    plt.psd(y11, label= 'FC6')
    plt.psd(y12, label= 'F4')
    plt.psd(y13, label= 'F8')
    plt.psd(y14, label= 'AF4')
    pylab.legend(loc= 'upper left')
    
    plt.show()
##    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
##    plt.xlabel('time (seconds)')
##    plt.hlines([-a, a], 0, T, linestyles='--')
##    plt.grid(True)
##    plt.axis('tight')
##    plt.legend(loc='upper left')
run()
