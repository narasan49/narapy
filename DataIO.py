import netCDF4
import numpy as np
from astropy.io import fits
import datetime

def load_l3(dataf):
    nc = netCDF4.Dataset(dataf, "r")
    rad = nc.variables["radiance"][:][0]*1.0e-9
    lon   = nc.variables["longitude"][:]
    lat   = nc.variables["latitude"][:]
    inang  = nc.variables["inangle"][:][0]
    emang  = nc.variables["emangle"][:][0]
    t    = nc.variables["time"][:][0]
    nc.close()
    return rad, lon, lat, inang, emang, t

def filename2days(filename_without_directory, t_str0):
    #filename = "uvi_20170801_000000_l3b_v20180901.nc"
    t  = datetime.datetime.strptime(filename_without_directory[4:19], '%Y%m%d_%H%M%S')
    t0 = datetime.datetime.strptime(t_str0 , '%Y%m%d_%H%M%S')
    t_flt = (t-t0).total_seconds()/86400.
    return t_flt

def load_l3f_ct(l3f, l3mf):
    nc = netCDF4.Dataset(l3f, "r")
    npz = np.load(l3mf)
    crad = npz["corrected_radiance"]
    lon  = nc.variables["longitude"][:]
    lat  = nc.variables["latitude"][:]
    t    = nc.variables["time"][:][0]
    sslon= nc.variables["S_SOLLON"][:]
    inang  = nc.variables["inangle"][:]
    emang  = nc.variables["emangle"][:]
    if sslon > 270:
        sslon -= 360
    if sslon < -90:
        sslon += 360
    nc.close()
    return crad, lon, lat, t, sslon, inang, emang
def load_l3f_ct2(l3f, l3mf):
    nc = netCDF4.Dataset(l3f, "r")
    npz = np.load(l3mf)
    crad = npz["corrected_radiance"]
    t    = nc.variables["time"][:][0]
    nc.close()
    return crad, t
