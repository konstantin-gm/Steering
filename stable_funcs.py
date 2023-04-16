# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:42:11 2023

@author: mishagin.k
"""
import numpy

def allan_deviation(z, dt, tau):
    ADEV = numpy.zeros(tau.size, dtype='double')
    n = z.size
    maxi = 0
    for i in range(tau.size):
        if tau[i]*3 < n:
            maxi = i
            sigma2 = numpy.sum((z[2*tau[i]::1] - 2*z[tau[i]:-tau[i]:1] + z[0:-2*tau[i]:1])**2)
            ADEV[i] = numpy.sqrt(0.5*sigma2/(n-2*tau[i]))/tau[i]/dt
        else:
            break
    return tau[:maxi].astype(numpy.double)*dt, ADEV[:maxi]

def parabolic_deviation(z, dt, tau):
    ADEV = numpy.zeros(tau.size, dtype='double')
    n = z.size
    maxi = 0
    for i in range(tau.size):
        if tau[i]*3 < n:
            maxi = i
            M = 0
            Si = 0
            c1 = numpy.polyfit(range(0,tau[i]+1,1),z[0:tau[i]+1:1],1)
            for j in range(tau[i],n-tau[i],tau[i]):
                c2 = numpy.polyfit(range(j,j+tau[i]+1,1),z[j:j+tau[i]+1:1],1)
                Si += (c1[0] - c2[0])**2
                M += 1
                c1 = c2
            ADEV[i] = numpy.sqrt(0.5*Si/M)/dt
        else:
            break    
    return tau[:maxi].astype(numpy.double)*dt, ADEV[:maxi]