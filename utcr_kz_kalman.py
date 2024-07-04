import numpy
from scipy import linalg
import matplotlib.pyplot as plt

def covar_by_noise(q, dt):
    '''
    Covariance matrix

    Parameters
    ----------
    q : tuple(3) - (q0, q1, q2)
        Square roots of noise intensities of discrete process(q0 - WPN, q1 - WFN, q2 - RWFN).
    dt : double
        Interval of discrete-time model.

    Returns
    -------
    Q : numpy.ndarray([2, 2])
        Covariance matrix.

    '''
    Q = numpy.array([[q[1]**2*dt+q[2]**2*dt**3/3, q[2]**2*dt**2/2],
                      [q[2]**2*dt**2/2, q[2]**2*dt]], dtype=numpy.double)
    return Q


def crit_coef(T, dt):
    '''
    Calculation of gain coefficients (critical damping) for PI-regulator

    Parameters
    ----------
    T : double
        Time constant [s].
    dt : double
        Steering interval.

    Returns
    -------
    numpy.ndarray([1, 2])
        Gain coefficients.

    '''
    gx = (1-numpy.exp(-dt/T))**2/dt
    gy = 1-numpy.exp(-2*dt/T)
    
    return numpy.array([gx, gy])

class Model:
    def __init__(self, free_noise, ref_noise, control, dt, ctrl_interval, drift):
        '''
        Model for timescale steering simulation

        Parameters
        ----------
        free_noise : tuple(3) - (q0, q1, q2)
            Free oscillator (clock) noise parameters (q0 - WPN, q1 - WFN, q2 - RWFN).
        ref_noise : tuple(3) - (q0, q1, q2)
            Reference signal noise parameters (q0 - WPN, q1 - WFN, q2 - RWFN).
        control : numpy.ndarray([1, 2])
            Control loop gain coefficients.
        dt : int
            Modeling interval [s].
        ctrl_interval : int
            Steering interval [dt].
        delay : int
            Delay interval [dt].
        drift : double
            Frequency drift of free oscillator per day.

        Returns
        -------
        None.

        '''
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
        self.F = numpy.array([[1, dt], 
                         [0, 1]])
        self.B = numpy.array([[dt], 
                         [1]])
        self.H = numpy.array([[1, 0]])
        self.D = numpy.array([[0.5*dt**2], 
                         [dt]])        
        self.R = self.ref_noise[0]**2 + self.free_noise[0]**2
            
    def kalman(self, dX, P, z, u):
        dX = self.F @ dX + self.B*u
        P = self.F @ P @ self.F.T + self.free_Q + self.ref_Q            
        K = P @ self.H.T / (self.H @ P @ self.H.T + self.R)
        dX = dX + K*(z - self.H @ dX)
        P = (numpy.eye(2) - numpy.outer(K, self.H)) @ P
        
        return dX
        
    def calculate(self, Xlock, Xfree, Xref, N):
        self.output = numpy.zeros([N, 1])
        self.ref = numpy.zeros((N, 1))
        self.free = numpy.zeros((N, 1))
        self.dphi = numpy.zeros((N, 1))
        uarr = []
                        
        P = numpy.array([[self.R, 0], 
                         [0, self.free_noise[1]**2]])
        
        qK = []
        zlist = []
        t = numpy.linspace(0, 6, 7) * 86400.
        
        u = 0        
        acc = 0
        dX = numpy.zeros([2, 1])        
        dXpred = numpy.zeros([2, 1])
        dXctrl = numpy.zeros([2, 1])
        last_u = 0
        
        for i in range(N):
            
            free_w = numpy.random.randn(2, 1)
            ref_w = numpy.random.randn(2, 1)
            Xlock = self.F@Xlock + self.free_L@free_w + self.B*u + self.D*self.drift
            Xfree = self.F@Xfree + self.free_L@free_w + self.D*self.drift
            Xref = self.F@Xref + self.ref_L@ref_w
            
            
            wpn_free = numpy.random.randn(1)*self.free_noise[0]
            wpn_ref = numpy.random.randn(1)*self.ref_noise[0]
            
            self.output[i] = Xlock[0] + wpn_free
            self.free[i] = Xfree[0] + wpn_free
            self.ref[i] = Xref[0] + wpn_ref #+ 1e-9*numpy.cos(2*numpy.pi*i*dt/86400./2.)
            
            z = (Xlock[0] + wpn_free) - (Xref[0] + wpn_ref) 
            zlist.append(z[0])
            
            dX = self.kalman(dX, P, z, u)
            
            qK.append(dX)            
                        
#           Управление
            if len(zlist) >= 11:
                dXctrl = qK[6]                
                dXctrl[1] = (qK[0][1]+qK[1][1]+qK[2][1]+qK[3][1]+qK[4][1]+qK[5][1]+qK[6][1]+last_u*4)/7
                #dXctrl[1] = (qK[4][1]+qK[5][1]+qK[6][1])/3
                                
                for k in  range(4):
                    dXctrl = self.F @ dXctrl
                   
                u = -self.G @ dXctrl
                #acc += 1e-1*u
                #u += acc
                last_u = u
                uarr.append(u)
                qK = qK[7:]
                zlist = zlist[7:]
            else:
                u = 0                
            
            self.dphi[i] = z
            
        return self.output, self.ref, self.free, self.dphi, uarr

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

if __name__ == "__main__":
        
    dt = 86400
    steer_dt = 7*dt
    T = 10*dt
    G = crit_coef(T, steer_dt)
    print('[gx, gy] = ', G)
    
    PHM_noise = (numpy.sqrt(4.1e-26), numpy.sqrt(7.2e-26), numpy.sqrt(2.4e-36))
    UTC_noise = (numpy.sqrt(8e-20), numpy.sqrt(8.5e-25), 1e-60)     
    model = Model(PHM_noise, UTC_noise, G, dt, 7, 2e-17/86400)
    
    Xlock = numpy.zeros([2, 1])
    Xfree = numpy.zeros([2, 1])
    Xref = numpy.zeros([2, 1])
    x, ref, free, dphi, u = model.calculate(Xlock, Xfree, Xref, 20000)
    
    plt.figure(1)
    t_tr = 30
    plt.plot(ref[t_tr:])
    plt.plot(x[t_tr:])
    plt.xlabel("время, в ед." + str(dt) + " c")
    plt.ylabel("фаза, с")
    stdphi = numpy.sqrt(numpy.mean(numpy.array(dphi[t_tr:])**2))#numpy.std(dphi1[t_tr:])
    rmsu = numpy.sqrt(numpy.mean(numpy.array(u[t_tr:])**2))
    print(f'Average offset, s: {numpy.mean(dphi[t_tr:])}')
    print(f'RMS of timescale error, s: {stdphi}')
    print(f'RMS of frequency corrections: {rmsu}') 
    
    tau = numpy.arange(1,10)
    tau = numpy.append(tau, numpy.arange(10,100,10))
    tau = numpy.append(tau, numpy.arange(100,1000,100))
    tau = numpy.append(tau, numpy.arange(1000,10000,1000))
    tau = numpy.append(tau, numpy.arange(10000,100000,10000))
    tau = numpy.append(tau, numpy.arange(100000,1000000,100000))
    tau = numpy.append(tau, numpy.arange(1000000,10000000,1000000))
    t_tr = 30
    taus, adev_lock = allan_deviation(x[t_tr:-1,0], dt, tau)
    taus, adev_free = allan_deviation(free[t_tr:-1,0], dt, tau)
    taus, adev_ref = allan_deviation(ref[t_tr:-1,0], dt, tau)
    plt.figure(2)
    plt.loglog(taus, adev_free)
    plt.loglog(taus, adev_ref)
    plt.loglog(taus, adev_lock)
    plt.legend(["свободный","опора","захват"])
    plt.xlabel("интервал времени измерения, c")
    plt.ylabel("СКДО")
    plt.title("Переходный процесс исключен: " + str(numpy.round(t_tr*dt/86400)) + " сут.")