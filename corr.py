import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import shift
import time

def displace_axes(dx, dy):
    #x: 要素0番目が西向き最大変位
    #y: 要素0番目が南向き最大変位
    x = np.arange(-dx,0)
    y = np.arange(-dy/2,dy/2) #y軸+-に同じだけ探索すると、2画像のy軸要素数の偶奇は同じになる。・・・差は２で割れる

    return x, y

def CC(img1, img2, cnt):
    n = img1.shape[0]
    ones = np.ones([n])
    nv = np.correlate(ones, cnt, mode="same")

    fg = np.correlate(img2, img1, mode="same")
    #fg[0]の計算に使っているimg1は
    #img1[0:n/2]
    #この分散を知りたい

    fsum = np.correlate(ones, img1, mode="same")
    f2sum= np.correlate(ones, img1*img1, mode="same")
    gsum = np.correlate(img2, ones, mode="same")
    g2sum= np.correlate(img2*img2, ones, mode="same")

    cov  = fg - fsum*gsum/nv
    fvar = f2sum-fsum**2/nv
    gvar = g2sum-gsum**2/nv
    print(cov[n//4:n//2])
    print(nv[n//4:n//2])
    print(fvar[n//4:n//2])
    print(gvar[n//4:n//2])

    ff = np.correlate(img1, img1, mode="same")
    #gg = np.correlate(img2, img2, mode="same")
    #これは自己相関
    #ff[0]は
    #f[0]*f[n/2]+f[1]*f[n/2+1]+...+f[n/2-1]*f[n-1]
#     print(ff[n//2])

    return cov/np.sqrt(fvar*gvar)

# def cross_correlation_coefficient():
#     for xi in range(0,nx2-nx1):
#         for yj in range(0,ny2-ny1):
#             for i in range(
#             cc[j,i] =
#             img2_cut = img2[0+j:ny1+j, 0+i:nx1+i]
#             c = np.corrcoef(img1.reshape([nx1*ny1]), img2_cut.reshape([nx1*ny1]))
#             cc[j,i] = c[0,1]


"""
CrossCorr:小領域の雲追跡
img1: 追跡に用いる画像
img2: 探索領域
"""
def CrossCorr(img1, img2):
    nx1 = img1.shape[1]
    ny1 = img1.shape[0]
    nx2 = img2.shape[1]
    ny2 = img2.shape[0]
    n = nx1*ny1

#     t1 = time.time()
    cc = np.zeros([ny2-ny1, nx2-nx1])
    for i in range(0,nx2-nx1):
        for j in range(0,ny2-ny1):
            img2_cut = img2[0+j:ny1+j, 0+i:nx1+i]
            c = np.corrcoef(img1.reshape([nx1*ny1]), img2_cut.reshape([nx1*ny1]))
            cc[j,i] = c[0,1]

#     print(time.time()-t1)
    return cc

"""
1より5倍くらい早くなった
"""
# @jit
def CorrInCorr2(img1_pad,img1_pad_bi, img2_1d):
    Simg22    = np.correlate(img2_1d**2, img1_pad_bi)
    Simg2     = np.correlate(img2_1d, img1_pad_bi)
    Simg1img2 = np.correlate(img2_1d, img1_pad)
    return Simg22, Simg2, Simg1img2

def CorrInCorr2FFT(img1_pad,img1_pad_bi, img2_1d):
#     t2 = time.time()
    n2 = img2_1d.shape[0]
    n1 = img1_pad.shape[0]
    img1_pad_pad = np.zeros_like(img2_1d)
#     img1_pad_pad[(n2-n1)//2:(n2-n1)//2+n1] = img1_pad
    img1_pad_pad[0:n1] = img1_pad
    img1_pad_bi_pad = np.zeros_like(img2_1d)
#     img1_pad_bi_pad[(n2-n1)//2:(n2-n1)//2+n1] = img1_pad_bi
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
#     print(time.time()-t2)
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

#     t1 = time.time()
#     Simg22, Simg2, Simg1img2 = CorrInCorr2(img1_pad, img1_pad_bi, img2_1d)

    Simg22, Simg2, Simg1img2 = CorrInCorr2FFT(img1_pad, img1_pad_bi, img2_1d)
#     print(time.time()-t1)
    cov1 = Simg12 - Simg1**2/n + 1.0e-20 #0割りを防ぐために極小値を加算
    cov2 = Simg22 - Simg2**2/n + 1.0e-20
#     print(cov2)
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
