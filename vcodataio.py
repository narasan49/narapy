import numpy as np
import netCDF4

class L3data():
    def __init__(self, l3f, varnames=[]):
        sp = l3f.split("/")
        sp[-2] = sp[-2].replace("l3", "l3m")
        sp[-1] = sp[-1].replace("nc", "npz")
        l3mf = "/".join(sp)
        self.l3f = l3f
        self.l3mf = l3mf
        nc = netCDF4.Dataset(self.l3f, "r")
        lon = nc.variables["longitude"][:]
        lat = nc.variables["latitude"][:]
        self.grid2deg = lon[1]-lon[0]
        self.nx = lon.shape[0]
        self.ny = lat.shape[0]
        nc.close()
        if varnames:
            self.loadVariable(varnames)
    
    def loadVariable(self, varnames=[]):
        import netCDF4
        nc = netCDF4.Dataset(self.l3f, "r")
        # print(nc.variables.keys())
        if isinstance(varnames, list):
            for varname in varnames:
                if varname in ["inangle", "emangle", "phangle", "time", "S_SOLLON", "S_DISTAV"]:
                    var = nc.variables[varname][:][0]
                    setattr(self, varname, var)
                elif varname =="radiance":
                    var = nc.variables[varname][:][0]*1e-9
                    setattr(self, varname, var)
                elif varname=="corrected_radiance":
                    npz = np.load(self.l3mf)
                    self.cradiance = npz["corrected_radiance"]
                elif varname in ["longitude", "latitude"]:
                    var = nc.variables[varname][:]
                    setattr(self, varname, var)
                else:
                    nc.close()
                    raise NameError(varname)
        else:
            nc.close()
            raise TypeError()
        nc.close()

    def calcLST(self):
        if self.S_SOLLON > 270: self.S_SOLLON -= 360
        if self.S_SOLLON < -90: self.S_SOLLON += 360
        X_noon = round(self.S_SOLLON / self.grid2deg)
        lst = (X_noon - np.arange(self.nx))/self.nx * 24 + 12
        self.lst = np.where(lst <= 0, lst+24, lst)
    
#     def calcDate(self):
        