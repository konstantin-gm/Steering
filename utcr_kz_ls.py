import numpy
from scipy import linalg
import matplotlib.pyplot as plt
import allantools as allan


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
    def __init__(self, free_noise, ref_noise, control, dt, ctrl_interval, delay, drift):
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
        self.delay = delay
        self.drift = drift
        
    def calculate(self, Xlock, Xfree, Xref, N):
        '''
        Simulate timescale steering process

        Parameters
        ----------
        Xlock : numpy.ndarray([2, 1])
            Initial state of steered oscillator (clock).
        Xfree : numpy.ndarray([2, 1])
            Initial state of free oscillator.
        Xref : numpy.ndarray([2, 1])
            Initial state of reference signal.
        N : int
            Number of iterations.

        Returns
        -------
        output : numpy.ndarray([N, 1])
            Phase of steered oscillator.
        ref : numpy.ndarray([N, 1])
            Phase of reference signal.
        free : numpy.ndarray([N, 1])
            Phase of unlocked oscillator.
        dphi1 : numpy.ndarray([N, 1])
            Steering offset.
        dphi1 : numpy.ndarray([N, 1])
            Steering offset for alternative algorithm (no extrapolation).
        uarr1 : list
            Frequency corrections.
        uarr1 : list
            Frequency corrections for alternative algorithm.

        '''
        
        output = numpy.zeros([N, 1])
        ref = numpy.zeros((N, 1))
        free = numpy.zeros((N, 1))
        dphi1 = numpy.zeros((N, 1))        
        uarr1 = []
        dphi2 = numpy.zeros((N, 1))        
        uarr2 = []
        
        Xlock1 = Xlock
        Xlock2 = Xlock.copy()
        
        F = numpy.array([[1, dt], 
                         [0, 1]])
        B = numpy.array([[dt], 
                         [1]])
        D = numpy.array([[0.5*dt**2], 
                         [dt]])
        
        zlist1 = []
        zlist2 = []
        t = numpy.linspace(0, self.ctrl_interval-1, self.ctrl_interval) * 86400.
        
        u1 = 0        
        u2 = 0
        last_u1 = 0
        last_u2 = 0
        dXctrl = numpy.zeros([2, 1])
        delta = self.ctrl_interval - self.delay
        
        acc = 0
        for i in range(N):
            
            free_w = numpy.random.randn(2, 1)
            ref_w = numpy.random.randn(2, 1)
            Xlock1 = F@Xlock1 + self.free_L@free_w + B*u1 + D*self.drift
            Xlock2 = F@Xlock2 + self.free_L@free_w + B*u2 + D*self.drift
            Xfree = F@Xfree + self.free_L@free_w + D*self.drift
            Xref = F@Xref + self.ref_L@ref_w
            
            
            wpn_free = numpy.random.randn(1)*self.free_noise[0]
            wpn_ref = numpy.random.randn(1)*self.ref_noise[0]
            
            output[i] = Xlock1[0] + wpn_free
            free[i] = Xfree[0] + wpn_free
            ref[i] = Xref[0] + wpn_ref
            
            z1 = (Xlock1[0] + wpn_free) - (Xref[0] + wpn_ref) 
            zlist1.append(z1[0])
            z2 = (Xlock2[0] + wpn_free) - (Xref[0] + wpn_ref) 
            zlist2.append(z2[0])                     
            
            if len(zlist1) >= self.ctrl_interval + self.delay:
                #Control with linear extrapolation of phase
                tmp_array = numpy.array(zlist1[0:self.ctrl_interval])                
                tmp_array[0:delta] -= last_u1*numpy.linspace(delta, 1, delta)*86400
                c = numpy.polyfit(t, tmp_array, 1) #linear approximation
                dXctrl[0] = (c[1] + c[0]*(t[-1] + self.delay*86400))  #extrapolation of timescale offset to the moment of control action                                
                dXctrl[1] = c[0] #frequency offset estimation                
                u1 = -self.G @ dXctrl      
                uarr1.append(u1)
                last_u1 = u1    
                
                #Control without linear extrapolation
                tmp_array = numpy.array(zlist2[0:self.ctrl_interval])
                tmp_array[0:delta] -= last_u2*numpy.linspace(delta, 1, delta)*86400
                c = numpy.polyfit(t, tmp_array, 1) #linear approximation
                dXctrl[0] = c[1] + c[0]*t[-1] #no extrapolation
                dXctrl[1] = c[0] #frequency offset estimation                
                u2 = -self.G @ dXctrl      
                uarr2.append(u2)
                last_u2 = u2    
                zlist1 = zlist1[self.ctrl_interval:]
                zlist2 = zlist2[self.ctrl_interval:]
            else:
                u1 = 0              
                u2 = 0
            
            dphi1[i] = z1
            dphi2[i] = z2
            
        return output, ref, free, dphi1, dphi2, uarr1, uarr2


if __name__ == "__main__":
    
    dt = 86400
    steer_dt = 7*dt
    T =10*dt
    G = crit_coef(T, steer_dt)
    print('[gx, gy] = ', G)
    
    PHM_noise = (numpy.sqrt(4.1e-26), numpy.sqrt(7.2e-26), numpy.sqrt(2.4e-36))
    UTC_noise = (numpy.sqrt(8e-20), numpy.sqrt(8.5e-25), 1e-60)     
    model = Model(PHM_noise, UTC_noise, G, dt, 7, 4, 1.8e-17/86400)
    
    Xlock = numpy.zeros([2, 1])
    Xfree = numpy.zeros([2, 1])
    Xref = numpy.zeros([2, 1])
    x, ref, free, dphi1, dphi2, u1, u2 = model.calculate(Xlock, Xfree, Xref, 2000)
    
    plt.figure(1)
    t_tr = 30
    plt.plot(dphi1[t_tr:])
    plt.plot(dphi2[t_tr:])
    plt.xlabel("time, in units of " + str(dt) + " s")
    plt.ylabel("phase offset, s")    
       
    stdphi1 = numpy.sqrt(numpy.mean(numpy.array(dphi1[t_tr:])**2))#numpy.std(dphi1[t_tr:])
    rmsu1 = numpy.sqrt(numpy.mean(numpy.array(u1[t_tr:])**2))
    print(f'Average offset, s: {numpy.mean(dphi1[t_tr:])}')
    print(f'STD of timescale error, s: {stdphi1}')
    print(f'RMS of frequency corrections: {rmsu1}')    
    
    stdphi2 = numpy.sqrt(numpy.mean(numpy.array(dphi2[t_tr:])**2))#numpy.std(dphi2[t_tr:])
    rmsu2 = numpy.sqrt(numpy.mean(numpy.array(u2[t_tr:])**2))
    print('No extrapolation result')
    print(f'Average offset, s: {numpy.mean(dphi2[t_tr:])}')
    print(f'STD of timescale error, s: {stdphi2}')
    print(f'RMS of frequency corrections: {rmsu2}')
    print(stdphi2/stdphi1, rmsu2/rmsu1)
    
    tau = numpy.arange(1, 100)
    taus, pdev_lock1, err, ns = allan.pdev(dphi1[t_tr:-1, 0]/dt, rate=1, taus=tau)
    taus, pdev_lock2, err, ns = allan.pdev(dphi2[t_tr:-1, 0]/dt, rate=1, taus=tau)
    taus, pdev_clk_ref, err, ns = allan.pdev((ref[t_tr:-1, 0]-free[t_tr:-1, 0])/dt, rate=1, taus=tau)
    taus, pdev_free, err, ns = allan.pdev(free[t_tr:-1, 0]/dt, rate=1, taus=tau)
    taus, pdev_ref, err, ns = allan.pdev(ref[t_tr:-1, 0]/dt, rate=1, taus=tau)
    
    plt.figure(2)        
    plt.loglog(taus, pdev_free, 'r')
    plt.loglog(taus, pdev_clk_ref, 'g')
    plt.loglog(taus, pdev_ref, 'b')
    plt.loglog(taus, pdev_lock1, 'm')
    plt.loglog(taus, pdev_lock2, '--m')
    plt.xlabel("Интервал времени измерения, cут")
    plt.ylabel("Параболическая девиация")
    #plt.xlabel("Averagig time, days")
    #plt.ylabel("PDEV")
    arrowprops = {
        'arrowstyle': '->',
        }
    (selected_x, selected_y) = (taus[0], pdev_free[0])
    plt.annotate('1',
              xy=(selected_x, selected_y),
              xytext=(selected_x*1.5, selected_y*1.5),
              arrowprops=arrowprops)
    i = numpy.where(taus==10)
    (selected_x, selected_y) = (taus[i], pdev_clk_ref[i])
    plt.annotate('2',
             xy=(selected_x, selected_y),
             xytext=(selected_x*1.2, selected_y*1.2),
             arrowprops=arrowprops)
    i = numpy.where(taus==25)
    (selected_x, selected_y) = (taus[i], pdev_ref[i])        
    plt.annotate('3',
              xy=(selected_x, selected_y),
              xytext=(selected_x*1.3, selected_y*1.3),
              arrowprops=arrowprops)
    i = numpy.where(taus==20)
    (selected_x, selected_y) = (taus[i], pdev_lock1[i])
    plt.annotate('4',
              xy=(selected_x, selected_y),
              xytext=(selected_x*1.5, selected_y*1.5),
              arrowprops=arrowprops)
    ax = plt.gca()
    ax.set_ylim([4e-16, 6e-15])
    ax.set_yscale('log')
    plt.tick_params(axis='y', which='minor')
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.ticker as mticker
    ax.yaxis.set_minor_formatter(mticker.LogFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.savefig('pdevs_color_model_ls.svg', format='svg', dpi=1200)
    plt.savefig('pdevs_color_model_ls.png', format='png', dpi=1200)