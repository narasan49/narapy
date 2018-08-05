"""
Functions for bayesian estimation

* GMRFM_without_BC
"""
import numpy as np
from scipy.optimize import minimize



"""
Free energy
"""
def ln_p(x, vt, lmd, n):
    res = 0.
    for i in range(0, int(n**2)):
        tmp = 1./(x[1] + x[0]/lmd[i])
        res += -0.5*(np.log(tmp) - vt[i]**2.*tmp)
    return res

"""
Estimation of hyper-parameter
"""
def estimate_hyp_param(a0, vt, lmd, n):
    a = minimize(ln_p, a0, args = (vt, lmd, n), method = 'Nelder-Mead')
    return a.x


"""
Image restoration using 
Gaussian Markov Random Field Model without Boundary Conditions (Katakami et al., 2017)

Usage
 res = GMRFM_without_BC(v, a0)

Input values
 v : imput image with a size of n x n
 a0: inutial guess of hyper-parameters.
     This should be 2-element array.

Return values
 res["a"]: estimated hyper-parameters
 res["u"]: restored image

"""
def GMRFM_without_BC(v, a0):
    if v.shape[0] != v.shape[1]:
        print("input array should be square matrix")
    
    # n x n matrix
    n = v.shape[0]
    
    i_vec = np.arange(n)
    j_vec = np.arange(n)
    i_mat, j_mat = np.meshgrid(i_vec, j_vec)
    
    #define matrices K, U
    Kmat = np.sqrt(2./n)*np.cos((i_mat+.5)*j_mat/n*np.pi)
    Kmat[0, :] = np.sqrt(1./n)
    Umat = np.kron(Kmat, Kmat)
    
    #eigenvalue lambda
    lmd = 4.*np.sin(i_mat*np.pi/2./n)**2. + 4.*np.sin(j_mat*np.pi/2./n)**2.
    lmd = lmd.reshape([n*n])
    lmd[0] = 1e-10
    
    #vt = U x v
    vt = np.dot(Umat, v.reshape([n*n]))
    
    #optimize hyper-parameters
    a = estimate_hyp_param(a0, vt, lmd, n)
    
    #image restoration
    ut = 1./(1.+a[1]/a[0]*lmd)*vt
    u = np.linalg.solve(Umat, ut).reshape([n, n])
    res = {'u': u, 'a': a}
    return res

