import matplotlib.pyplot as plt
import numpy as np


class fig_3panels():
    def __init__(self, fid, size, x1, x2, x3, y):
        self.fid = fid
        self.size = size
        if len(x1.shape)==1:
            x1 = x1.reshape([1,-1])

        if len(x2.shape)==1:
            x2 = x2.reshape([1,-1])

        if len(x3.shape)==1:
            x3 = x3.reshape([1,-1])

        xs = [x1, x2, x3]
        self.xs = xs
        self.y = y

        self.set_labels()
        self.set_xranges()
        self.set_common_label("")
        self.set_xscales()
        self.set_common_range()

    def set_common_label(self, label):
        self.common_label=label

    def set_label(self, targ, label):
        self.labels[targ] = label

    def set_labels(self, l1="", l2="", l3=""):
        self.labels = [l1, l2, l3]

    def set_linestyle(self, ls1=["k-"], ls2=["k-"], ls3=["k-"]):
        xs = self.xs
        lss = [ls1, ls2, ls3]
        for i in range(len(lss)):
            while len(lss[i]) < xs[i].shape[0]:
                lss[i].append(lss[i][0])
        self.lss = lss

    def set_xranges(self, r1=[], r2=[], r3=[]):
        self.lims=[r1, r2, r3]

    def set_common_range(self, rng=[]):
        self.common_range = rng

    def set_xscale(self, targ, scale):
        self.xscale[targ] = scale

    def set_xscales(self, s1="", s2="", s3=""):
        self.xscale = [s1, s2, s3]

    def plot(self, fontsize=16):
        if not hasattr(self, "lss"):
            self.set_linestyle()

        fig = plt.figure(self.fid, (self.size, self.size))
        ax1 = fig.add_axes((0.1, 0.1, 0.25, 0.8))
        ax = []
        for j in range(len(self.xs)):
            ax.append(fig.add_axes((0.1+j*0.325, 0.1, 0.25, 0.8)))
            for i in range(self.xs[j].shape[0]):
                ax[j].plot(self.xs[j][i], self.y, self.lss[j][i])
            plt.tick_params(labelsize=fontsize)
            plt.xlabel(self.labels[j], fontsize=fontsize)
            if j==0:
                plt.ylabel(self.common_label, fontsize=fontsize)
            else:
                plt.tick_params(labelleft=False)

            if self.lims[j]: plt.xlim(self.lims[j])
            if self.xscale[j]: plt.xscale(self.xscale[j])
            if self.common_range: plt.ylim(self.common_range)

        plt.show()


def maxIndices(arr):
    y = arr.argmax()//arr.shape[1]
    x = arr.argmax()%arr.shape[1]
    return y, x
def minIndices(arr):
    y = arr.argmin()//arr.shape[1]
    x = arr.argmin()%arr.shape[1]
    return y, x
def BoxPlot(left, bottom, height, width):
    plt.plot([left, left+width], [bottom, bottom], "--k")
    plt.plot([left+width, left+width], [bottom, bottom+height], "--k")
    plt.plot([left, left+width], [bottom+height, bottom+height], "--k")
    plt.plot([left, left], [bottom, bottom+height], "--k")

def figTemplate(fid, height, lon, lat, img, deg2grid, fontsize=16, box=False, uind=None, vind=None, cc=[[0],[0]], intval=1):
    ny, nx = img.shape
    aspct = nx/ny
    fig = plt.figure(1, (height*aspct, height))
    plt.pcolormesh(lon, lat, img, cmap="gray")
    plt.tick_params(labelsize=fontsize)
    plt.xlabel("East longitude\n[degree]", fontsize=fontsize)
    plt.ylabel("North latitude\n[degree]", fontsize=fontsize)
    if box:
        if uind==None or vind==None:
            vind, uind = maxIndices(cc)
        left   = lon[-1] - 8 + (uind-cc.shape[1])/deg2grid/intval
        bottom = lat[0] + 8/2 + (vind-cc.shape[0]/2)/deg2grid/intval

        BoxPlot(left, bottom, 8, 8)
    return fig
def figCC(fid, height, cc, uind, vind, intval, deg2grid, fontsize=16, vmin=None, vmax=None):
    ny, nx = cc.shape
    # maxy, maxx = maxIndices(cc)
    u = (np.arange(nx)-nx)/deg2grid/intval
    v = (np.arange(ny)-ny/2)/deg2grid/intval
    aspct = nx/ny

    fig = plt.figure(1, (height*aspct, height))
    im = plt.pcolormesh(u, v, cc, vmin=vmin, vmax=vmax)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel("u [$\mathsf{deg\ hr^{-1}}$]", fontsize=fontsize)
    plt.ylabel("v [$\mathsf{deg\ hr^{-1}}$]", fontsize=fontsize)
    plt.plot([u[int(uind)]], [v[int(vind)]], "r+")

    cax = fig.add_axes((0.95, 0.1, 0.03, 0.8))
    cbar = plt.colorbar(im, cax, orientation="vertical")
    plt.tick_params(labelsize=fontsize)
    return fig
