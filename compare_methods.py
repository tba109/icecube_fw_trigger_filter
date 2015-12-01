#!/usr/bin/env python

##############################################################################
# Tyler Anderson Sat Nov 28 11:54:25 EST 2015
# 
# Compare response for two sets of coefficients
#
##############################################################################

import FixedPoint as fp
import numpy as np
import matplotlib.pyplot as plt
import first_order_iir_lpf as foi

# Make an impulse
x = np.zeros(256)
x[128] = 1

# Feed in the first version
# y1,n1,x1 = foi.first_order_iir_lpf(x,0.5,18)
y1,n1,x1 = foi.first_order_iir_lpf_a_b(x,0.1,0.1,18)
y2,n2,x2 = foi.first_order_iir_lpf_a_b(x,0.75,0.5,18)

fig, ax = plt.subplots()
ax.stem(n1,y1,linefmt='-b',markerfmt='ob')
ax.stem(n2,y2,linefmt='-g',markerfmt='og')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.show()

# Do the FFT
dt = 0.004 # sampling period in microseconds
fft_out1 = np.fft.fft(y1)
fft_freq1 = np.fft.fftfreq(x1.shape[-1],d=dt)
fft_out2 = np.fft.fft(y2)
fft_freq2 = np.fft.fftfreq(x2.shape[-1],d=dt)

# just pick out the parts I want
# fft_freq2 = np.where(np.logical_and(fft_freq>=0,fft_freq<np.amax(fft_freq)))
plt.plot(fft_freq1,np.sqrt(fft_out1.real**2 + fft_out1.imag**2),'-o')
plt.plot(fft_freq2,np.sqrt(fft_out2.real**2 + fft_out2.imag**2),'-o')
plt.yscale('log')
plt.xscale('log')
plt.xlim(0,1.01*np.amax(fft_freq1))
# plt.xlim(-1.01*np.amax(fft_freq),1.01*np.amax(fft_freq))
plt.xlabel('Frequency (MHz)')
plt.show()
