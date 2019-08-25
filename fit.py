from scipy.optimize import least_squares, curve_fit
from scipy.linalg import solve
import numpy as np

# def gauss_fit(a0, x, y):
#     def model(a, x):
#         return a[0]*np.exp(-0.5*((x-a[2])/a[1])**2)+a[3]

#     def fun(a, x, y):
#         return model(a, x) -y

#     def jac(a, x, y):
#         J = np.empty((x.size, a.size))
#         J[:,0] = np.exp(-0.5*((x-a[2])/a[1])**2)
#         J[:,1] = a[0]*(x-a[2])**2/a[1]**3*np.exp(-0.5*((x-a[2])/a[1])**2)
#         J[:,2] = a[0]*(x-a[2])/a[1]**2*np.exp(-0.5*((x-a[2])/a[1])**2)
#         J[:,3] = 1.0
#         return J

#     res = least_squares(fun, a0, jac=jac, args=(x, y))
#     return res

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
