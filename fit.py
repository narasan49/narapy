from scipy.optimize import least_squares, curve_fit
from scipy.linalg import solve
import numpy as np

def gauss_fit(x, y, sigma=None, p0=None, bounds=(-np.inf, np.inf)):
    def func(x, a, b, c, d):
        return a*np.exp(-0.5*((x-b)/c)**2)+d

    popt, pcov = curve_fit(func, x, y, sigma=sigma, bounds=bounds, p0=p0)
    perr = np.sqrt(pcov.diagonal())
    return popt, perr

def LinearFit(x, y):
    n     = x.reshape([-1]).shape[0]
    sumx  = np.sum(x)
    sumy  = np.sum(y)
    sumxx = np.sum(x*x)
    sumyy = np.sum(y*y)
    sumxy = np.sum(x*y)

    # Amat x xvec = bvec
    Amat = [[sumxx, sumx],
            [sumx , n   ]]
    bvec =  [sumxy, sumy]

    res = solve(Amat, bvec)
    return res

def FitCC(x, y, err, maxind):
    def func_fit(x, a, b, c, d, e):
        return np.arctanh(a*np.exp(-0.5*((x-b)/c)**2))+d+e*x

    initial_guess = [y[maxind], x[maxind], 10, 0, 0]
    rng = np.arange(maxind-30, maxind+30)
    try:
        res = curve_fit(func_fit, x[rng], y[rng], sigma=err[rng], absolute_sigma=True,
                        p0=initial_guess, bounds=([-np.inf, x.min(), 0., -10., -np.inf], [np.inf, x.max(), 100, 10., np.inf]))
        param = res[0]
        perr = np.sqrt(np.diag(res[1]))
    except RuntimeError:
        param = [np.NaN for i in range(5)]
        perr  = [np.NaN for i in range(5)]
    return param[1], perr[1]
