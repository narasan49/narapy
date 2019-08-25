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