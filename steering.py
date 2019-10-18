# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:50:22 2019

@author: MK
"""

import numpy
from scipy import linalg
import numba
import pylab

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

class Model:
    def __init__(self, free_noise, ref_noise, control, dt, ctrl_interval, drift):
        self.free_noise = free_noise
        self.ref_noise = ref_noise
        self.free_Q = covar_by_noise(free_noise, dt)
        self.free_L = linalg.cholesky(self.free_Q)
        self.ref_Q  = covar_by_noise(ref_noise, dt)
        self.ref_L = linalg.cholesky(self.ref_Q)
        self.G = control
        self.dt = dt
        self.ctrl_interval = ctrl_interval
        self.drift = drift
    @numba.jit
    def calculate(self, N):
        self.output = numpy.zeros((N, 1))
        self.ref = numpy.zeros((N, 1))
        self.free = numpy.zeros((N, 1))
        
        F = numpy.array([[1, dt], [0, 1]])
        B = numpy.array([dt, 1])
        H = numpy.array([1, 0])
        D = numpy.array([0.5*dt**2, dt])
        
        R = self.ref_noise[0]**2 + self.free_noise[0]**2
        P = numpy.array([[R, 0], [0, self.free_noise[1]**2]])
        
        u=0
        
        Xlock = numpy.zeros(2)
        Xfree = numpy.zeros(2)
        Xref = numpy.zeros(2)
        dX = numpy.zeros(2)
        
        for i in range(N):
            free_w = numpy.random.randn(2)
            ref_w = numpy.random.randn(2)
            
            Xlock = F@Xlock + self.free_L@free_w + B*u + D*self.drift
            Xfree = F@Xfree + self.free_L@free_w + D*self.drift
            Xref = F@Xref + self.ref_L@ref_w
            
            wpn_free = numpy.random.randn(1)*self.free_noise[0]
            wpn_ref = numpy.random.randn(1)*self.ref_noise[0]
            
            self.output[i] = Xlock[0] + wpn_free
            self.free[i] = Xfree[0] + wpn_free
            self.ref[i] = Xref[0] + wpn_ref
            
            z = (Xlock[0] + wpn_free) - (Xref[0] + wpn_ref)
            
            Ppred = F @ P @ F.T + self.free_Q + self.ref_Q
            dXpred = F@dX +B*u
            K = Ppred@H.T/(H@Ppred@H.T+R)
            dX = dXpred + K*(z - H@dXpred)
            P = (numpy.eye(2)-numpy.outer(K, H))@Ppred
            
            if i%self.ctrl_interval==0:
                u = -self.G@dX
            else:
                u = 0
                
        return self.output, self.ref, self.free



#@numba.jit('Tuple((float64[:], float64[:]))(float64[:], float64, float64[:])', cache=True)
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

#@numba.jit('Tuple((float64[:], float64[:]))(float64[:], float64, float64[:])', cache=True)
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

PHM_noise = (numpy.sqrt(9e-26), numpy.sqrt(1e-25), numpy.sqrt(2.3e-34))
#GNSS_noise = (0,numpy.sqrt(9e-24), 1e-60) #WFN reference
GNSS_noise = (numpy.sqrt(1e-18), 1e-50, 1e-60) #WPN reference

dt = 1000
T = 3.3*1e5
model = Model(PHM_noise, GNSS_noise, LQG_coef(T, dt), dt, 1, 1e-16/86400)
x, ref, free = model.calculate(200000)

tau = numpy.arange(1,10)
tau = numpy.append(tau, numpy.arange(10,100,10))
tau = numpy.append(tau, numpy.arange(100,1000,100))
tau = numpy.append(tau, numpy.arange(1000,10000,1000))
tau = numpy.append(tau, numpy.arange(10000,100000,10000))
tau = numpy.append(tau, numpy.arange(100000,1000000,100000))
tau = numpy.append(tau, numpy.arange(1000000,10000000,1000000))
taus, adev_lock = allan_deviation(x[:,0], dt, tau)
taus, adev_free = allan_deviation(free[:,0], dt, tau)
taus, adev_ref = allan_deviation(ref[:,0], dt, tau)
pylab.loglog(taus, adev_free)
pylab.loglog(taus, adev_ref)
pylab.loglog(taus, adev_lock)
taus, pdev_lock = parabolic_deviation(x[:,0], dt, tau)
taus, pdev_free = parabolic_deviation(free[:,0], dt, tau)
taus, pdev_ref = parabolic_deviation(ref[:,0], dt, tau)
pylab.loglog(taus, pdev_free)
pylab.loglog(taus, pdev_ref)
pylab.loglog(taus, pdev_lock)

f = []
f.append('tau.txt')
f.append('adev_free.txt')
f.append('adev_ref.txt')
f.append('adev_lock.txt')
f.append('pdev_free.txt')
f.append('pdev_ref.txt')
f.append('pdev_lock.txt')
d = [taus, adev_free, adev_ref, adev_lock, pdev_free, pdev_ref, pdev_lock]
for i in range(7):
    f[i] = open(f[i],'w')
    for dat in d[i]:
        f[i].write(str(dat) + '\n')
    f[i].close()