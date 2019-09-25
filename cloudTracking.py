import numpy as np
from scipy.interpolate import interp2d, interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import copy
import algebra
import corr
import vcodataio
import time
import fit

"""
Input:
radiances[time, longitude, latitude]
times[time]
inang[longitude, latitude]
emang[longitude, latitude]
lon[longitude]
lat[latitude]
sslon

tsize:
template size in unit of degrees.

xdivision, ydivision:
Images are divided into (xdivision-1) x (ydivision-1) templates.

sx[ydivision], sy:
Search region for each templates in unit of grids.
Here, we assume the wind velocity are within 50-150 m/s (zonal) and -30-30 m/s (meridional).
This is why the longitudes of search areas are variable along latitude.

"""

class CloudTracking():
    class DataForCloudTracking():
        pass

    class CloudTrackingResults():
        pass

    def __init__(self, radiances, times, inang, emang, sslon, lon, lat, sx, sy, xdivision, ydivision, tsize=8):
        deg2grid=int(1/(lon[1]-lon[0]))

        self.sx=sx
        self.sy=sy
        self.deg2grid=deg2grid
        self.tsize=tsize*deg2grid
        self.xdivision=xdivision
        self.ydivision=ydivision
        self.data = self.DataForCloudTracking()
        self.res = self.CloudTrackingResults()

        nsp, ny, nx=radiances.shape
        self.nsp=nsp
        ssp = int(sslon*deg2grid)
        """    center longitude and latitude of each template    """
        #indices
        lon_vec_ind = np.array([int(ssp - self.tsize/2/2*xdivision + self.tsize/2*mx + self.tsize/2) for mx in range(xdivision-1)])
        lon_vec_ind = np.where(lon_vec_ind >= nx, lon_vec_ind-nx, lon_vec_ind)
        lat_vec_ind = np.array([int(self.tsize/2*my + (90*deg2grid - ydivision*self.tsize/2/2) + self.tsize/2) for my in range(ydivision-1)])
        #In degrees
        lon_vec = lon[lon_vec_ind] - 1/deg2grid/2
        lat_vec = lat[lat_vec_ind] - 1/deg2grid/2

        """   inangle and emangle for each template center   """
        inang_tmpl = inang[0][lat_vec_ind][:,lon_vec_ind]
        emang_tmpl = emang[0][lat_vec_ind][:,lon_vec_ind]

        self.data.ny=ny
        self.data.nx=nx
        self.data.radiances=radiances
        self.data.times=times
        self.data.sslon=sslon
        self.data.lon=lon
        self.data.lat=lat
        self.data.ssp=ssp
        self.data.lon_vec_ind=lon_vec_ind
        self.data.lat_vec_ind=lat_vec_ind
        self.data.lon_vec=lon_vec
        self.data.lat_vec=lat_vec
        self.data.inang_tmpl=inang_tmpl
        self.data.emang_tmpl=emang_tmpl
        self.RegionOfInterest()

    def CrossCorrelation(self, dif_streak=True):
        """
        r: cross-correlation
        xind, yind: indices of the templates
        theta = determined orientation
        """
        r    = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp-1)]
        xind = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp)]
        yind = [[[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)] for j in range(self.nsp)]
        theta=  [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]

        for mx in range(self.xdivision-1):
            for my in range(self.ydivision-1):
                if (self.data.inang_tmpl[my][mx] < 80.0 and self.data.emang_tmpl[my][mx] <80.0):
                    minlon = int(self.data.ssp - self.tsize/2/2*self.xdivision + self.tsize/2*mx)
                    minlat = int(self.tsize/2*my + (90*self.deg2grid - self.ydivision*self.tsize/2/2))
                    rad=[]
                    """Cut out templates"""
                    for ti in range(self.nsp):
                        sy_ti = int(self.sy*round(self.data.times[ti]-self.data.times[0]))
                        sx_ti = int(self.sx[my]*round(self.data.times[ti]-self.data.times[0]))
                        yindi = np.arange(minlat-sy_ti, minlat+sy_ti+self.tsize)
                        xindi = np.arange(minlon-sx_ti, minlon+self.tsize)
                        # yindi = np.linspace(minlat-sy_ti, minlat+sy_ti+self.tsize-1, self.tsize+sy_ti*2).astype(np.int64)
                        # xindi = np.linspace(minlon-sx_ti, minlon+self.tsize-1, self.tsize+sx_ti).astype(np.int64)
                        #nxを超えたら0へ回るようにする
                        xindi = np.where(xindi >= self.data.nx, xindi-self.data.nx, xindi)
                        radi = self.data.radiances[ti][yindi][:,xindi]
                        #low-pass
                        radi = nd.gaussian_filter(radi, [5,5])
                        if dif_streak:
                            if ti==0:
                                theta[my][mx] = corr.orientation(radi)
                            #differentiate
                            radi = corr.dif_img_along_streak(radi, theta[my][mx])
                        else:
                            theta[my][mx] = None

                        xind[ti][my][mx] = xindi.tolist()
                        yind[ti][my][mx] = yindi.tolist()
                        rad.append(radi)

                    for i in range(self.nsp-1):
                        r[i][my][mx] = corr.CrossCorr2(rad[0], rad[i+1]).tolist()
        self.res.theta = theta
        self.res.cc = r
        self.res.xind = xind
        self.res.yind = yind

    def TemporalSuperPosition(self):
        """
        Superpose the CCSs obtained with diferent pair.
        The region where obtained CCSs were 0 or 1 was ignored (fill blank ndarray).
        """

        selec = [[[1 if np.array(cc).ndim==2 else 0 for cc in cc_ti_row] for cc_ti_row in cc_ti] for cc_ti in self.res.cc]
        selec = np.where(np.array(selec).sum(axis=0)==2)
        selec = np.array([selec[0], selec[1]]).transpose()
        res_cc_tsp = [[np.NaN]*(self.xdivision-1) for i in range(self.ydivision-1)]

        for my, mx in selec:
            cc_tsp = np.array(self.res.cc[-1][my][mx][:])
            xsize_tsp = cc_tsp.shape[1]
            ysize_tsp = cc_tsp.shape[0]

            for ti in range(0,self.nsp-2):
                cc = np.array(self.res.cc[ti][my][mx][:])
                xsize = cc.shape[1]
                ysize = cc.shape[0]
                x = np.arange(xsize)
                y = np.arange(ysize)
                new_x = np.arange(xsize_tsp)/xsize_tsp*xsize
                new_y = np.arange(ysize_tsp)/ysize_tsp*ysize

                cc_resize = interp2d(x, y, cc)(new_x, new_y)

                #superpose
                cc_tsp += cc_resize
            cc_tsp /= self.nsp-1
            res_cc_tsp[my][mx] = cc_tsp.tolist()

        self.res.cc_tsp=res_cc_tsp

    def RegionOfInterest(self):
        roi = []
        dt = self.data.times[-1]-self.data.times[0]
        Rv = 6052+70
        for i in range(self.ydivision-1):
            sy_ti = int(self.sy*round(dt))
            sx_ti = int(self.sx[i]*round(dt))
            roi_y = np.zeros([2*sy_ti, sx_ti])
            #150 m/s => grid
            umax = int(sx_ti-((150.0*dt*3600*1e-3)/(2*np.pi*Rv*np.cos(self.data.lat_vec[i]*np.pi/180))*360*self.deg2grid)//1)
            if umax<0: umax = 0
            #50 m/s
            umin = int(sx_ti-((50.0*dt*3600*1e-3)/(2*np.pi*Rv*np.cos(self.data.lat_vec[i]*np.pi/180))*360*self.deg2grid)//1)
            #+-30 m/s
            vmax = int(sy_ti+((30.0*dt*3600*1e-3)/(2*np.pi*Rv)*360*self.deg2grid)//1)
            vmin = int(sy_ti-((30.0*dt*3600*1e-3)/(2*np.pi*Rv)*360*self.deg2grid)//1)
            roi_y[vmin:vmax][:,umax:umin]=1
            roi.append(roi_y)
        self.roi = roi

    def Relaxation(self, a=0.2, d=1.0, thresh=0.2):
        xdivision = self.xdivision
        ydivision = self.ydivision
        candidates = [[0 for i in range(xdivision-1)] for j in range(ydivision-1)]
        p = [[0 for i in range(xdivision-1)] for j in range(ydivision-1)]
        for x in range(xdivision-1):
            for y in range(ydivision-1):
                cc = np.array(self.res.cc_tsp[y][x])
                if isinstance(cc, np.ndarray):
                    if cc.ndim==2:
                        candidates[y][x] = algebra.FindPeaks2d(cc, roi=self.roi[y], thresh=thresh)
                        p[y][x] = cc[candidates[y][x]]
                    else:
                        candidates[y][x] = (np.array([]), np.array([]))
                        p[y][x] = np.array([])
        p_pre = copy.deepcopy(p)
        opt_u = np.zeros([ydivision-1, xdivision-1])
        opt_v = np.zeros([ydivision-1, xdivision-1])
        for mx in range(xdivision-1):
            for my in range(ydivision-1):
                if candidates[my][mx][0].shape[0] >= 1: #if not brank...
                    peaky, peakx = candidates[my][mx]
                    Ik = peakx.shape[0]
                    q = np.zeros([Ik])
                    for i in range(Ik):
                        #Gk: neighboring templates
                        Gkx = []
                        Gky = []
                        if mx < xdivision-1-1:
                            Gkx.append(mx+1)
                            Gky.append(my)
                        if mx > 0:
                            Gkx.append(mx-1)
                            Gky.append(my)
                        if my < ydivision-1-1:
                            Gkx.append(mx)
                            Gky.append(my+1)
                        if my > 0:
                            Gkx.append(mx)
                            Gky.append(my-1)
                        Gk = zip(Gkx, Gky)
                        for mxp, myp in Gk:
                            peakxp = candidates[myp][mxp][1]
                            peakyp = candidates[myp][mxp][0]
                            Ikp = peakxp.shape[0]
                            for ip in range(Ikp):
                                q[i] += np.exp(-a/d*((peakx[i]-peakxp[ip])**2+(peaky[i]-peakyp[ip])**2))*p_pre[myp][mxp][ip]

                    sum_pq = np.sum(p_pre[my][mx]*q)
                    for i in range(Ik):
                        p[my][mx][i] = p_pre[my][mx][i]*q[i]/sum_pq

                    maxp_ind = np.argmax(p[my][mx])
                    opt_u[my][mx] = peakx[maxp_ind]
                    opt_v[my][mx] = peaky[maxp_ind]
                else:
                    opt_u[my][mx] = np.NaN
                    opt_v[my][mx] = np.NaN

        self.res.u_grid=opt_u
        self.res.v_grid=opt_v

    def SubPixelEstimation(self, rng=20):
        """
        sub-pixel estimation of the peak position obtained by relaxation labeling
        with error estimation
        """

        usub = np.zeros([self.ydivision-1, self.xdivision-1])
        vsub = np.zeros([self.ydivision-1, self.xdivision-1])
        uerr = np.zeros([self.ydivision-1, self.xdivision-1])
        verr = np.zeros([self.ydivision-1, self.xdivision-1])
        for mx in range(self.xdivision-1):
            for my in range(self.ydivision-1):
                u_grid=self.res.u_grid[my][mx]
                v_grid=self.res.v_grid[my][mx]
                if (u_grid-u_grid==0) and (v_grid-v_grid==0):
                    uinds, vinds, ccs, imgs2 = [], [], [], []
                    img1 = self.data.radiances[0][self.res.yind[0][my][mx]][:,self.res.xind[0][my][mx]]
                    theta = self.res.theta[my][mx]
                    if theta: img1 = corr.dif_img_along_streak(img1, theta)
                    cc_n = np.array(self.res.cc[-1][my][mx])
                    for i in range(self.nsp-1):
                        cc = np.array(self.res.cc[i][my][mx])
                        uind = cc.shape[1]/cc_n.shape[1]*self.res.u_grid[my][mx]
                        vind = cc.shape[0]/cc_n.shape[0]*self.res.v_grid[my][mx]
                        uinds.append(int(uind))
                        vinds.append(int(vind))
                        ccs.append(cc)

                        img2 = self.data.radiances[i+1][self.res.yind[i+1][my][mx]][:,self.res.xind[i+1][my][mx]]
                        if theta: img2 = corr.dif_img_along_streak(img2, theta)
                        imgs2.append(img2)

                    sp_err_x = np.zeros([ccs[-1].shape[1]])
                    sp_err_y = np.zeros([ccs[-1].shape[0]])
                    uvec_new = np.linspace(0, 1, sp_err_x.shape[0])
                    vvec_new = np.linspace(0, 1, sp_err_y.shape[0])
                    """calculate degree of freedom near the peak position"""
                    # tt = time.time()
                    for uind, vind, cc, img2 in zip(uinds, vinds, ccs, imgs2):
                        ny1, nx1 = img1.shape
                        ny2, nx2 = img2.shape
                        """correlation length"""
                        omega = np.zeros([ny2-ny1, nx2-nx1])
                        x = img1.reshape([-1])
                        n = x.shape[0]
                        cx = corr.auto_corr(x)
                        tau = np.arange(-n+1,n)
                        #自由度計算範囲
                        #ピーク位置から rng 以内もしくはデータの端
                        rngu_min = int(max([uind-rng, 0]))
                        rngu_max = int(min([uind+rng, nx2-nx1]))
                        rngv_min = int(max([vind-rng, 0]))
                        rngv_max = int(min([vind+rng, ny2-ny1]))
                        rngu = np.arange(rngu_min, rngu_max)
                        rngv = np.arange(rngv_min, rngv_max)
                        omega_u = np.arange(nx2-nx1)
                        omega_v = np.arange(ny2-ny1)
                        #for ループなくしたい
                        for dx in rngu:
                            c2_u = corr.auto_corr(img2[vind:vind+ny1][:,dx:nx1+dx].reshape([-1]))
                            omega_u[dx] = np.nansum((1-abs(tau)/n)*cx*c2_u)
                        # print(omega_u2)
                        # print(omega_u)
                        for dy in rngv:
                            c2_v = corr.auto_corr(img2[dy:ny1+dy][:,uind:nx1+uind].reshape([-1]))
                            omega_v[dy] = np.nansum((1-abs(tau)/n)*cx*c2_v)
                        dof_x = n/omega_u
                        dof_y = n/omega_v
                        err_x = 1.65/(dof_x-3)
                        err_y = 1.65/(dof_y-3)
                        """superpose errors of CCSs"""
                        uvec = np.linspace(0, 1, err_x.shape[0])
                        vvec = np.linspace(0, 1, err_y.shape[0])
                        sp_err_x += interp1d(uvec, err_x**2)(uvec_new)
                        sp_err_y += interp1d(vvec, err_y**2)(vvec_new)

                    # print(time.time()-tt)
                    sp_err_x = np.sqrt(sp_err_x)
                    sp_err_y = np.sqrt(sp_err_y)
                    """sub-pixel estimation with non-linear least square fitting"""
                    cc_tsp = np.array(self.res.cc_tsp[my][mx])
                    nccy, nccx = cc_tsp.shape

                    usub[my, mx], uerr[my, mx] = fit.FitCC(np.arange(nccx), cc_tsp[int(v_grid)], sp_err_x, int(u_grid))
                    vsub[my, mx], verr[my, mx] = fit.FitCC(np.arange(nccy), cc_tsp[:,int(u_grid)], sp_err_y, int(v_grid))
                else:
                    usub[my, mx], uerr[my, mx], vsub[my, mx], verr[my, mx] = np.NaN, np.NaN, np.NaN, np.NaN
        self.res.u_sub_grid = usub
        self.res.u_err_grid = uerr
        self.res.v_sub_grid = vsub
        self.res.v_err_grid = verr

# class CTAnalysis():
#     def __init__(self, file):
#         import json
