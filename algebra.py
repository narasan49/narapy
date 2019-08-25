import numpy as np
from scipy.ndimage.interpolation import shift
import scipy.ndimage as nd

def FindPeaks(data):
    dif = shift(data, -1, cval=0) - data
    dif2 = dif*shift(dif, 1, cval=0)

    cand_dif = np.where(dif<0)
    cand_dif2 = np.where(dif2<0)
    return np.intersect1d(cand_dif, cand_dif2)

def FindPeaks2d(data, roi=np.array([]), thresh=0.0, degree=2):
    sm_data = nd.gaussian_filter(data, [degree, degree])
    if roi.ndim == data.ndim:
        """roi を設定した場合、していない場合で共通のピークを選ぶ。"""
        roied_data = sm_data*roi
        roied_xdif = shift(roied_data, [0, -1], cval=0) - roied_data
        roied_xdif2 = roied_xdif*shift(roied_xdif, [0, 1], cval=0)

        roied_ydif = shift(roied_data, [-1, 0], cval=0) - roied_data
        roied_ydif2 = roied_ydif*shift(roied_ydif, [1, 0], cval=0)

        cand_roied_xdif = np.where(roied_xdif<0, 1, 0)
        cand_roied_xdif2 = np.where(roied_xdif2<0, 1, 0)
        cand_roied_ydif = np.where(roied_ydif<0, 1, 0)
        cand_roied_ydif2 = np.where(roied_ydif2<0, 1, 0)
        cand_roied_thresh = np.where(roied_data>thresh, 1, 0)
        cand_roied = cand_roied_xdif*cand_roied_xdif2*cand_roied_ydif*cand_roied_ydif2*cand_roied_thresh
    else:
        cand_roied = 1

    xdif = shift(sm_data, [0, -1], cval=0) - sm_data
    xdif2 = xdif*shift(xdif, [0, 1], cval=0)

    ydif = shift(sm_data, [-1, 0], cval=0) - sm_data
    ydif2 = ydif*shift(ydif, [1, 0], cval=0)

    cand_xdif = np.where(xdif<0, 1, 0)
    cand_xdif2 = np.where(xdif2<0, 1, 0)
    cand_ydif = np.where(ydif<0, 1, 0)
    cand_ydif2 = np.where(ydif2<0, 1, 0)
    cand_thresh = np.where(sm_data>thresh, 1, 0)
    ind = np.where(cand_xdif*cand_xdif2*cand_ydif*cand_ydif2*cand_thresh*cand_roied==1)
    return ind
