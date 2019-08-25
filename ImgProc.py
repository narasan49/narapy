import numpy as np
from fit import LinearFit
import scipy.ndimage as nd

"""
ミナート補正:

単位：rad [W/m^2 sr nm]
      eang, iang [degree]
(ln(uu0), ln(uI))をフィッティング
"""
def MinnaertCorrection(rad, eang, iang):
    nx = rad.shape[1]
    ny = rad.shape[0]
    
    #異常な輝度を除外、入射角・出射角80°以上を除外
    valid_data = np.where((rad > 1.0e-7) &
                          (iang< 80.0) &
                          (eang< 80.0))

    if valid_data[0].shape[0] > 1000:
        I  = rad[valid_data]
        mu0= np.cos(iang[valid_data]*np.pi/180)
        mu = np.cos(eang[valid_data]*np.pi/180)

        ln1 = np.log(mu*mu0)
        ln2 = np.log(mu*I)

        #ln1, ln2 を線形フィッティング
        res = LinearFit(ln1, ln2)

        crad = rad*np.cos(eang*np.pi/180)/(np.cos(iang*np.pi/180)*np.cos(eang*np.pi/180))**res[0]
        crad = np.where((rad > 1.0e-7) &
                        (iang< 85.0) &
                        (eang< 85.0), crad, np.NaN)
    else:
        crad=np.array([None])
        res =np.array([None, None])
    return crad, res

"""
ガウス平滑化を用いたハイパスフィルタ
"""
def GaussHighPass(img, degree):
    fil = nd.gaussian_filter(img, [degree,degree], mode="wrap")
    res = img - fil
    return res