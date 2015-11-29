#!/usr/bin/env python

##############################################################################
# Tyler Anderson Sat Nov 28 11:54:25 EST 2015
# 
# Simulation of a simple first-order IIR filter using fixed-point binary. 
# This is to help us study implementation details in the FPGA. 
#
##############################################################################

import FixedPoint as fp
import numpy as np
import matplotlib.pyplot as plt

########################################################################################
# Simple first order low-pass IIR filter (derived from impulse invariance). 
# In discrete time, this is defined as:
# y[0] = x[0]*T/tau
# y[t] = exp[-T/tau]*y[t-T] + x[t]*T/tau
# where:
# t = time
# x = input waveform
# y = output waveform
# T = sample period
# tau = filter time constant
#  
# Inside an FPGA, we don't work with time directly, but rather sample number, n = 0,1,2..
# So, we need to convert the filter to operate on sample number. 
# Here's how I'm doing it:
# gamma = T/tau
# n = t/T
# The new filter equation is:
# y[0] = a*x[0]
# y[n] = a*x[n] + b*y[n-1]
# a = gamma 
# b = exp[-gamma]
# 
# In addition to the above, we want to study the efficiency in terms of fixed-point 
# binary representations of the calculation. I'm using the FixedPoint module from here
# for this: 
# https://pypi.python.org/pypi/Simple-Python-Fixed-Point-Module/0.5
# 
# So, in light of the above, the function accepts the following arguments: 
# x, the input waveform stored as ADC samples (integers!) in a numpy array. 
# gamma, the filter time constant. Default = 2 gives f3dB = 20MHz @ 250MSPS.
# prec is the binary precision used (i.e., the number of fractional bits).
#
# As a general note, low pass filters need time to move to the baseline. So if your
# data has a baseline offset, you either need to wait a while for it to get there, 
# or else initialize the first sample as the baseline value. 
def first_order_iir_lpf(xin, gamma=0.5, prec=6, vb=False):
    ff = fp.FXfamily(prec)
    a = fp.FXnum(gamma,ff)
    b = fp.FXnum(np.exp(-gamma),ff)

    # Initialize the output arrays
    n = np.arange(len(xin))
    x = np.zeros(len(xin))
    y = np.zeros(len(xin))
    
    # Some temporary variables
    xi = fp.FXnum(0,ff)
    yi = fp.FXnum(0,ff)

    # Run the filter
    for i in range(len(xin)):
        xi = fp.FXnum(xin[i],ff)
        
        if i == 0:
            yi = a*xi
        else:
            yi = a*xi + b*y[i-1]
                    
        if vb:
            print "i = %d, a = %.16f, b = %.16f, xi = %.16f, yi = %.16f" % (i,a,b,xi,yi)
        
        # Store fixed-point result
        n[i] = i
        x[i] = xi
        y[i] = yi

    return y,n,x
            
######################################################################################
# For running independently
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="first_order_iir_lpf",description="Function for a first-order low-pass IIR filter.")
    parser.add_argument("--gamma",type=float,help="Time constant in number of samples",default=0.5)
    parser.add_argument("--prec",type=int,help="Precision of calculation (i.e., number of bits to store right of binary point)",default=4)
    parser.add_argument("--fin", type=str,help="Read input waveform from file specified. Uses DDC2 data format.")
    parser.add_argument("--win",type=str,help="Input waveform to load. This option is ignored if the --fin option is specified. Arguments: impulse, step")
    parser.add_argument("--delay",type=int,help="Number of samples to delay with --win option. Ignored with --fin option.",default=32)
    parser.add_argument("--period",type=int,help="Total number of samples with --win option. Ignored with --fin option.",default=64)
    parser.add_argument('--version',action='version',version='%(prog)s 0.1')
    parser.add_argument('--verbose',help='Print additional debugging info',action='store_true')
    
    args = parser.parse_args()

    if args.fin: 
        # do some stuff here. 
        Null

    else:
        x = np.zeros(args.period)
        if args.win == 'step':
            for i in range(args.delay,args.period):
                x[i] = 1
        elif args.win == 'impulse':
            x[args.delay] = 1
    
    yout,nout,xout = first_order_iir_lpf(x,args.gamma,args.prec,args.verbose)
    
    plt.plot(nout,xout,'-o')
    plt.plot(nout,yout,'-o')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.show()
