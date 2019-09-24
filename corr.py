import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import shift
import time

"""
CrossCorr:小領域の雲追跡
img1: 追跡に用いる画像
img2: 探索領域
"""

def CorrInCorr2FFT(img1_pad,img1_pad_bi, img2_1d):
    n2 = img2_1d.shape[0]
    n1 = img1_pad.shape[0]
    img1_pad_pad = np.zeros_like(img2_1d)
    img1_pad_pad[0:n1] = img1_pad
    img1_pad_bi_pad = np.zeros_like(img2_1d)
    img1_pad_bi_pad[0:n1] = img1_pad_bi

    fft_img2_1d = np.fft.fft(img2_1d)
    fft_sq_img2_1d = np.fft.fft(img2_1d**2)
    fft_img1_pad = np.fft.fft(img1_pad_pad)
    fft_img1_pad_bi = np.fft.fft(img1_pad_bi_pad)

    Simg22    = np.fft.ifft(fft_sq_img2_1d*np.conj(fft_img1_pad_bi)).real
    Simg2     = np.fft.ifft(fft_img2_1d*np.conj(fft_img1_pad_bi)).real
    Simg1img2 = np.fft.ifft(fft_img2_1d*np.conj(fft_img1_pad)).real
    Simg2 = Simg2[:(n2-n1+1)]
    Simg22 = Simg22[:(n2-n1+1)]
    Simg1img2 = Simg1img2[:(n2-n1+1)]
    return Simg22, Simg2, Simg1img2

def CrossCorr2(img1, img2):
    nx1 = img1.shape[1]
    ny1 = img1.shape[0]
    nx2 = img2.shape[1]
    ny2 = img2.shape[0]
    n = nx1*ny1
    img1_pad = np.append(img1, np.zeros([ny1, nx2-nx1]), axis=1).reshape([-1])
    img1_pad_bi = np.append(np.ones_like(img1), np.zeros([ny1, nx2-nx1]), axis=1).reshape([-1])
    img2_1d = img2.reshape([-1])
    Simg12=np.sum(img1.reshape([-1])**2)
    Simg1=np.sum(img1.reshape([-1]))

    Simg22, Simg2, Simg1img2 = CorrInCorr2FFT(img1_pad, img1_pad_bi, img2_1d)
    cov1 = Simg12 - Simg1**2/n + 1.0e-20 #0割りを防ぐために極小値を加算
    cov2 = Simg22 - Simg2**2/n + 1.0e-20
    cov12 = Simg1img2 - Simg1*Simg2/n
    c = cov12/np.sqrt(cov1*cov2)
    c = c[:-1]
    c = c.reshape([ny2-ny1, nx2])
    c = c[:,:nx2-nx1]
    return c

def eval_func(theta, img):
    dif = dif_img_along_streak(img, theta)
    var = np.nanvar(dif)
    return var

def orientation(img, theta0=None):

    if not theta0:
        #局所解を避けるため0~piを10分割してその中から初期値を決める。
        #10個の初期値の評価関数を比較
        theta0_cand = np.linspace(0.4, 0.6, 10)*np.pi
        phi0_cand   = np.zeros([10])
        for i in range(0,10):
            phi0_cand[i] = eval_func(theta0_cand[i], img)

        theta0_ind = np.argmin(phi0_cand)
        theta0 = theta0_cand[theta0_ind]
    #評価関数の最小化
    theta = minimize(eval_func, theta0, args=(img), method='Nelder-Mead')
    res = theta.x[0]

    #0 < theta < piをとるように
    if res > np.pi:
        res -= np.pi
    if res < 0:
        res += np.pi
    return res

def dif_img_along_streak(img, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    #print(c, s)
    zipj = shift(img, [0,-1], cval=np.NaN) #z[i+1,j  ]
    zimj = shift(img, [0, 1], cval=np.NaN) #z[i-1,j  ]
    zijp = shift(img, [-1,0], cval=np.NaN) #z[i  ,j+1]
    zijm = shift(img, [ 1,0], cval=np.NaN) #z[i  ,j-1]

    res = c*0.5*(zipj-zimj)+s*0.5*(zijp-zijm) # (cos, sin) と grad(z[i,j])の内積
    res = np.where(res-res == 0, res, 0) #nan埋めを0埋めに変換

    return res

def max_indices(cc):
    maxcc = np.nanmax(cc)
    tmp = np.where(cc==maxcc)
#     print(tmp)
    xind=tmp[1]
    yind=tmp[0]
    return xind,yind, maxcc

# class Validation:

#     def MaxCorrelationArea(TmplShape, TargShape, x_dis, y_dis, lonTarg, latTarg, deg2grid):
#         """
#         MC: MaxCorrelation
#         ________________________
#         |   _______            |
#         |   |   MCpoint _______|
#         |   |      x    |      |
#         |   |      |    |      x:init
#         |   ^^^^^^^     |      |
#         |               ^^^^^^^|
#         |______________________|

#         """
#         TmplLeftBottom = (TargShape[1] - TmplShape[1], TargShape[0]/2 - TmplShape[0]/2)
#         TargLeftBottom = (TargShape[1] - TmplShape[1] + x_dis,
#                           TargShape[0]/2 - TmplShape[0]/2 + y_dis)
#         LonLeftBottom = lonTarg[int(TargLeftBottom[0])]
#         LatLeftBottom = latTarg[int(TargLeftBottom[1])]

#         LonWidth = TmplShape[1]/deg2grid
#         LatHeight= TmplShape[0]/deg2grid

#         return LonLeftBottom, LatLeftBottom, LonWidth, LatHeight

"""
誤差評価に必要な実行自由度の計算
Effective Degree of Freedom
入力値：
    imgs: テンプレート、テンプレートと相関が最大になるターゲット領域
    nsp: 画像数
"""
def EDoF(imgs, nsp):
    X = np.array([])
    Y = np.array([])
    tmpl1d = img[0].reshape([-1])
    M = tmpl1d.shape[0]

    for i in range(nsp-1):
        X = np.append(X, tmpl1d)

    for i in range(nsp-1):
        targ1d = img[i+1].reshape([-1])
        Y = np.append(Y, targ1d)

    RX = np.correlate(X, X, mode="same")
    RY = np.correlate(Y, Y, mode="same")

    Omega=0
    for i in range(0,(nsp-1)*M*2+1):
        Omega+=(1-np.abs(i-(nsp-1)*M)/((nsp-1)*M))*RX[i]*RY[i]

    Me = (nsp-1)*M/Omega

    return Me

def auto_corr(x):
    n = x.shape[0]
    ave = np.sum(x)/n
    ac  = np.correlate(x-ave, x-ave, "full")
    return ac/ac[n-1]
