"""
Functions for spectral analysis

*lomb_scargle
"""
import numpy as np
"""
Calculate Lomb-Scargle periodogram (Scargle, 1982)

imput parameters:
t : observed time. 1-d vector.
x(t): observed value. 1-d vector

"""
def lomb_scargle(t, x, n, nout = -1, dfreq = -1, alarm_rate = 0.01):
    
    if nout == -1:
        nout = 4*n
    if dfreq == -1:
        dt = t[n-1]-t[0]
        f_ny = 0.5*n/dt
        dfreq = f_ny/nout

    freq = np.arange(nout)*dfreq
    freq = freq.reshape([nout, 1])
    t = t.reshape([1, n])

    e = np.ones(n)
    freq_dot_t = np.dot(freq, t)
    
    sum2s = np.dot(np.sin(2.*2.*np.pi*freq_dot_t), e.reshape([n, 1]))
    sum2c = np.dot(np.cos(2.*2.*np.pi*freq_dot_t), e.reshape([n, 1]))

    tau = np.arctan(sum2s/sum2c)/(2.*np.pi*freq) #[nout, 1]

    freq_tau_mat = np.dot(freq*tau, e.reshape([1, n])) #[nout, 1] x [1, n]
    s = np.sin(2.*np.pi*(freq_dot_t - freq_tau_mat )) #[nout, n]
    c = np.cos(2.*np.pi*(freq_dot_t - freq_tau_mat ))
    
    y1 = sum(x)
    y2 = sum(x**2)
    sy = np.dot(s, x.reshape([n, 1])) # [nout, n] x [n, 1]
    cy = np.dot(c, x.reshape([n, 1]))
    s1 = np.dot(s, e.reshape([n, 1]))
    c1 = np.dot(c, e.reshape([n, 1]))
    s2 = np.dot(s**2, e.reshape([n, 1]))
    c2 = np.dot(c**2, e.reshape([n, 1]))
    
    sigma = (np.float64(y2) - np.float64(y1)**2/n)/n
    p = 0.5*(cy**2/c2 + sy**2/s2)
    amp = np.sqrt(cy**2/c2**2 + sy**2/s2**2)
    
    ssl = -np.log(1-(1-alarm_rate)**(1/np.float64(n)))*sigma
    alpha = (1-np.exp(-p/sigma))**n
    
    res = {"freq": freq.reshape([nout]), "pl": p.reshape([nout]), "ssl": ssl, "sigma": sigma, "significance_rate":alpha.reshape([nout]), "amp": amp.reshape([nout])}
    
    return res
