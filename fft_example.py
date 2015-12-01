#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
t = np.arange(256)
print t
sp = np.fft.fft(np.sin(t))
print sp
freq = np.fft.fftfreq(t.shape[-1])
print freq
print t.shape[-1]
plt.plot(freq, sp.real, freq, sp.imag)
plt.show()
