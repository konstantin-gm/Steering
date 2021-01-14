# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:50:22 2019

@author: MK
"""

import numpy as np
from scipy import linalg
import numba
import pylab
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def covar_by_noise(q, dt):
     Q = np.array([[q[1]**2*dt+q[2]**2*dt**3/3, q[2]**2*dt**2/2],
                      [q[2]**2*dt**2/2, q[2]**2*dt]], dtype=np.double)
     return Q

def LQG_coef(T, dt):
    F = np.array([[1, dt], [0, 1]])
    B = np.array([[dt], [1]])
    Wr = 1e-3*np.array([[0.25*T**4/dt**2]])
    Wq = 1e-3*np.eye(2)
    Gamma = linalg.solve_discrete_are(F, B, Wq, Wr)
    G_ = linalg.inv(B.T@Gamma@B+Wr)@B.T@Gamma@F
    G = G_[0]
    
    return G

def crit_coef(T, dt):
    gx = (1-np.exp(-dt/T))**2/dt
    gy = 1-np.exp(-2*dt/T)
    
    return np.array([gx, gy])

def gy1_coef(T, dt):
    gx = (1-np.exp(-dt/T))/dt
    gy = 1
    
    return np.array([gx, gy])

def norm_f(a, amin, amax):
    if a < amin:
        return -2
    if a > amax:
        return 2
    return (4*a - amin - amax)/(amax - amin)

def sys_of_rules1():
    universe = np.linspace(-1, 1, 3)
    error = ctrl.Antecedent(universe, 'error')
    delta = ctrl.Antecedent(universe, 'delta')
    output = ctrl.Consequent(universe, 'output')
    names = ['n', 'ze', 'p']
    error.automf(names=names)
    delta.automf(names=names)
    output.automf(names=names)    

    rule0 = ctrl.Rule(antecedent=((error['n'] & delta['ze']) |
                              (error['n'] & delta['n']) |
                              (error['ze'] & delta['n'])),
                  consequent=output['n'], label='rule neg')

    rule1 = ctrl.Rule(antecedent=((error['n'] & delta['p']) |
                              (error['ze'] & delta['ze']) |
                              (error['p'] & delta['n'])),
                  consequent=output['ze'], label='rule ze')

    rule2 = ctrl.Rule(antecedent=((error['p'] & delta['p']) |
                              (error['ze'] & delta['p']) |
                              (error['p'] & delta['ze'])),
                  consequent=output['p'], label='rule pos')

    system = ctrl.ControlSystem(rules=[rule0, rule1, rule2])
    return system


def sys_of_rules2():
    universe = np.linspace(-2, 2, 5)
    error = ctrl.Antecedent(universe, 'error')
    delta = ctrl.Antecedent(universe, 'delta')
    output = ctrl.Consequent(universe, 'output')
    names = ['nb', 'ns', 'ze', 'ps', 'pb']
    error.automf(names=names)
    delta.automf(names=names)
    output.automf(names=names)
    rule0 = ctrl.Rule(antecedent=((error['nb'] & delta['nb']) |
                              (error['ns'] & delta['nb']) |
                              (error['nb'] & delta['ns'])),
                  consequent=output['nb'], label='rule nb')

    rule1 = ctrl.Rule(antecedent=((error['nb'] & delta['ze']) |
                              (error['nb'] & delta['ps']) |
                              (error['ns'] & delta['ns']) |
                              (error['ns'] & delta['ze']) |
                              (error['ze'] & delta['ns']) |
                              (error['ze'] & delta['nb']) |
                              (error['ps'] & delta['nb'])),
                  consequent=output['ns'], label='rule ns')

    rule2 = ctrl.Rule(antecedent=((error['nb'] & delta['pb']) |
                              (error['ns'] & delta['ps']) |
                              (error['ze'] & delta['ze']) |
                              (error['ps'] & delta['ns']) |
                              (error['pb'] & delta['nb'])),
                  consequent=output['ze'], label='rule ze')

    rule3 = ctrl.Rule(antecedent=((error['ns'] & delta['pb']) |
                              (error['ze'] & delta['pb']) |
                              (error['ze'] & delta['ps']) |
                              (error['ps'] & delta['ps']) |
                              (error['ps'] & delta['ze']) |
                              (error['pb'] & delta['ze']) |
                              (error['pb'] & delta['ns'])),
                  consequent=output['ps'], label='rule ps')

    rule4 = ctrl.Rule(antecedent=((error['ps'] & delta['pb']) |
                              (error['pb'] & delta['pb']) |
                              (error['pb'] & delta['ps'])),
                  consequent=output['pb'], label='rule pb')
    system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4])
    return system

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
    #@numba.jit
    def calculate(self, N):
        self.output = np.zeros((N, 1))
        self.ref = np.zeros((N, 1))
        self.free = np.zeros((N, 1))
        
        F = np.array([[1, dt], [0, 1]])
        B = np.array([dt, 1])
        H = np.array([1, 0])
        D = np.array([0.5*dt**2, dt])
        
        R = self.ref_noise[0]**2 + self.free_noise[0]**2
        P = np.array([[R, 0], [0, self.free_noise[1]**2]])
        
        u = 0
        zi = []
        ui = []
        
        Xlock = np.zeros(2)
        Xfree = np.zeros(2)
        Xref = np.zeros(2)
        dX = np.zeros(2)
        
        xK = []
        yK = []
                
        system = sys_of_rules2()
        sim = ctrl.ControlSystemSimulation(system)
        
        for i in range(N):
            free_w = np.random.randn(2)
            ref_w = np.random.randn(2)
            
            Xlock = F@Xlock + self.free_L@free_w + B*u + D*self.drift
            Xfree = F@Xfree + self.free_L@free_w + D*self.drift
            Xref = F@Xref + self.ref_L@ref_w
            
            wpn_free = np.random.randn(1)*self.free_noise[0]
            wpn_ref = np.random.randn(1)*self.ref_noise[0]
            
            self.output[i] = Xlock[0] + wpn_free
            #self.output[i] = dX[0]
            self.free[i] = Xfree[0] + wpn_free
            self.ref[i] = Xref[0] + wpn_ref
            
            z = (Xlock[0] + wpn_free) - (Xref[0] + wpn_ref)
            
            Ppred = F @ P @ F.T + self.free_Q + self.ref_Q
            dXpred = F@dX +B*u
            K = Ppred@H.T/(H@Ppred@H.T+R)
            dX = dXpred + K*(z - H@dXpred)
            #self.output[i] = dX[0]
            P = (np.eye(2)-np.outer(K, H))@Ppred
            
            if i%self.ctrl_interval==0:
                # u = -self.G@dX
                sim.input['error'] = norm_f(dX[0], -1e-9, 1e-9)
                sim.input['delta'] = norm_f(dX[1], -5e-15, 5e-15)
                sim.compute()
                u = -1e-15*sim.output['output']
                zi.append(z)
                ui.append(u)
                xK.append(dX[0])
                yK.append(dX[1])
            else:
                u = 0                
        
        return self.output, self.ref, self.free, np.array(ui), np.array(zi), np.array(xK), np.array(yK)



#@numba.jit('Tuple((float64[:], float64[:]))(float64[:], float64, float64[:])', cache=True)
def allan_deviation(z, dt, tau):
    ADEV = np.zeros(tau.size, dtype='double')
    n = z.size
    maxi = 0
    for i in range(tau.size):
        if tau[i]*3 < n:
            maxi = i
            sigma2 = np.sum((z[2*tau[i]::1] - 2*z[tau[i]:-tau[i]:1] + z[0:-2*tau[i]:1])**2)
            ADEV[i] = np.sqrt(0.5*sigma2/(n-2*tau[i]))/tau[i]/dt
        else:
            break
    return tau[:maxi].astype(np.double)*dt, ADEV[:maxi]

#@numba.jit('Tuple((float64[:], float64[:]))(float64[:], float64, float64[:])', cache=True)
def parabolic_deviation(z, dt, tau):
    ADEV = np.zeros(tau.size, dtype='double')
    n = z.size
    maxi = 0
    for i in range(tau.size):
        if tau[i]*3 < n:
            maxi = i
            M = 0
            Si = 0
            c1 = np.polyfit(range(0,tau[i]+1,1),z[0:tau[i]+1:1],1)
            for j in range(tau[i],n-tau[i],tau[i]):
                c2 = np.polyfit(range(j,j+tau[i]+1,1),z[j:j+tau[i]+1:1],1)
                Si += (c1[0] - c2[0])**2
                M += 1
                c1 = c2
            ADEV[i] = np.sqrt(0.5*Si/M)/dt
        else:
            break    
    return tau[:maxi].astype(np.double)*dt, ADEV[:maxi]


sim = 1
adev = 1
pdev = 1
save = 0
dt =1000
if sim:
    adev = 1
    GNSS_noise = (np.sqrt(1e-18), 1e-50, 1e-60) #WPN reference
    q1 = np.sqrt(1e-25)
    q2 = np.sqrt(2.3e-34)
    free_noise = (np.sqrt(9e-26), q1, q2)
    
    #T = 86400*5.97#100000#164416
    T = 300000
    # G = crit_coef(T, dt)     
    # G = LQG_coef(0.5*T, dt)
    G = np.array([5e-6, 0.9])
    # print('[gx, gy] = ', G)
    model = Model(free_noise, GNSS_noise, G, dt, 1, 5e-16/86400)
    x, ref, free, u, z, xK, yK = model.calculate(20000)
    pylab.figure(1)
    pylab.plot(z)
    pylab.plot(x, 'y')
    pylab.plot(xK, 'r')
    # pylab.ylabel("Разность фаз (Калман), с")
    # pylab.xlabel("время")
    pylab.ylabel("Phase offset (Kalman), s")    
    pylab.xlabel("time, x1000 s")
    pylab.figure(2)
    pylab.plot(yK)    
    pylab.plot(u, 'r')
    # pylab.legend("Разность частот (Калман)", "Управление")
    # pylab.xlabel("время")
    pylab.legend(("Frequency offset (Kalman)", "Control"))    
    pylab.xlabel("time, x1000 s")
    pylab.show()
    
    if adev:
        tau = np.arange(1,10)
        tau = np.append(tau, np.arange(10,100,10))
        tau = np.append(tau, np.arange(100,1000,100))
        tau = np.append(tau, np.arange(1000,10000,1000))
        tau = np.append(tau, np.arange(10000,100000,10000))
        tau = np.append(tau, np.arange(100000,1000000,100000))
        tau = np.append(tau, np.arange(1000000,10000000,1000000))
        taus, adev_lock = allan_deviation(x[10000:-1,0], dt, tau)
        taus, adev_free = allan_deviation(free[10000:-1,0], dt, tau)
        taus, adev_ref = allan_deviation(ref[10000:-1,0], dt, tau)
        pylab.figure(3)
        pylab.loglog(taus, adev_free)
        pylab.loglog(taus, adev_ref)
        pylab.loglog(taus, adev_lock)
        pylab.ylabel("ADEV")
        pylab.xlabel("averaging time, s")
    if pdev:
        taus, pdev_lock = parabolic_deviation(x[:,0], dt, tau)
        taus, pdev_free = parabolic_deviation(free[:,0], dt, tau)
        taus, pdev_ref = parabolic_deviation(ref[:,0], dt, tau)
        pylab.figure(4)
        pylab.loglog(taus, pdev_free)
        pylab.loglog(taus, pdev_ref)
        pylab.loglog(taus, pdev_lock)
        pylab.ylabel("PDEV")
        pylab.xlabel("averaging time, s")
    if save:        
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