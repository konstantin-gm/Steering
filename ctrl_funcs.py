# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:42:11 2023

@author: mishagin.k
"""
import numpy
from scipy import linalg

def covar_by_noise(q, dt):
     Q = numpy.array([[q[1]**2*dt+q[2]**2*dt**3/3, q[2]**2*dt**2/2],
                      [q[2]**2*dt**2/2, q[2]**2*dt]], dtype=numpy.double)
     return Q

def LQG_coef(T, dt):
    F = numpy.array([[1, dt], [0, 1]])
    B = numpy.array([[dt], [1]])
    Wr = 1e-3*numpy.array([[0.25*T**4/dt**2]])
    Wq = 1e-3*numpy.eye(2)
    Gamma = linalg.solve_discrete_are(F, B, Wq, Wr)
    G_ = linalg.inv(B.T@Gamma@B+Wr)@B.T@Gamma@F
    G = G_[0]
    
    return G

def crit_coef(T, dt):
    gx = (1-numpy.exp(-dt/T))**2/dt
    gy = 1-numpy.exp(-2*dt/T)
    
    return numpy.array([gx, gy])

def gy1_coef(T, dt):
    gx = (1-numpy.exp(-dt/T))/dt
    gy = 1
    
    return numpy.array([gx, gy])

