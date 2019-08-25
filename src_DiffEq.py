import numpy as np
import scipy.ndimage as nd
import time
from multiprocessing import Pool

# def calc_column(Nair, alt,imin, nlev, zinf):
#     colNair = 0.5*Nair[nlev-1]*(zinf-alt[nlev-1])*1e5
#     for i in range(1,nlev-imin):
#         colNair += 0.5*(Nair[nlev-i]+Nair[nlev-i-1])*(alt[nlev-i]-alt[nlev-i-1])*1e5
#     return colNair

def interp_smooth(x, x0, y0):
    y1= np.interp(x, x0, y0)
    y = nd.gaussian_filter(y1, 5, mode = "nearest")
    return y

def load_vira(file="/mnt/c/analysis/simu/photochemi/VIRA_alt0-250.txt", zmin=0, zmax=200, dz=1):
    data = np.loadtxt(file, skiprows = 1)
    alt0 = data[:,0]
    temp0= data[:,1]
    pres0= data[:,2]
    rho0 = data[:,3]
    Nco20= data[:,4]
    No0  = data[:,5]
    No0 = np.where(No0==0,1e-30, No0)
    zinf = 500
    alt  = np.arange(zmin, zmax+51, dz)

    temp = interp_smooth(alt, alt0, temp0)
    pres = interp_smooth(alt, alt0, pres0)
    rho  = interp_smooth(alt, alt0, rho0)
    Nco2 = 10**(interp_smooth(alt, alt0, np.log10(Nco20)))
    No   = 10**(interp_smooth(alt, alt0, np.log10(No0)))

    alt  = np.arange(zmin, zmax+1, dz)
    nlev= len(alt)
    temp=temp[0:nlev]
    pres=pres[0:nlev]
    rho=rho[0:nlev]
    Nco2=Nco2[0:nlev]
    No=No[0:nlev]
    return alt,No,Nco2,temp,nlev,zinf

#define cross_section
#unit: [cm**2]
def cross_sec(wl):
    #nwl = 131 
    #wl = np.linspace(119.5, 249.5, nwl) #[nm]
    nwl = len(wl)
    sigma = np.zeros([nwl])
    sigma[0:41] = 5e-19
    for i in range(41,nwl):
        sigma[i]=5*10**(-wl[i]/40*6+5)
    return sigma


#Load solar spectrum
def solar_spe():
    data2 = np.loadtxt('E-490_solar_spectrum.txt', skiprows = 1)
    wl = data2[:,0]*1000 #[nm]
    phi_e=data2[:,1]/1000 #[w/m**2/nm]
    nwl = wl.shape[0]
    r = (1.0140/0.7279)**2 #convert irradiance at Venus

    #unit conversion #[W/m**2/nm] -> [photons/cm**2/s/nm]
    # F = n * hv
    u = wl*1e-9/6.626e-34/2.99792458e8/1e4
    phi0 = phi_e*r*u
    return wl, phi0,nwl

#calc CO2 column density
#unit: [cm**-2]

def calc_column(Nair, alt, nlev, zinf, dz):
    colNair = np.zeros([nlev])
    colNair[nlev-1] = 0.5*Nair[nlev-1]*dz*1e5
    for i in range(1,nlev):
        colNair[nlev-i-1] = colNair[nlev-i] + 0.5*(Nair[nlev-i]+Nair[nlev-i-1])*(alt[nlev-i]-alt[nlev-i-1])*1e5
    return colNair

#band-production rate
def calc_production(theta, wl, Nco2, phi, sigma, alt,nlev,zinf,nwl, dz):
    colNco2 = calc_column(Nco2, alt, nlev, zinf, dz)
    band_prod_rate = np.zeros([nwl,nlev])
    for i in range(0,nwl):
        band_prod_rate[i] = Nco2*sigma[i]*phi[i]*np.exp(-1.0/np.cos(theta*np.pi/180)*sigma[i]*colNco2)

    #integrate production rate
    prod_rate = np.zeros([nlev])
    for i in np.where(wl <= 200.0)[0]:
#     for i in range(0,nwl-1-50):
        prod_rate += 0.5*(band_prod_rate[i]+band_prod_rate[i+1])*(wl[i+1]-wl[i])

    return prod_rate

#Loss rate
#recombination
def calc_loss(No, Nco2, temp):
    k_recomb = 2*7.5e-33*(300/temp)**3.25 #[cm^6/ s] -> [ (1e10 cm^2)^3/s]
    L = k_recomb*No**2*Nco2
    return L

#with  diffusion
#chaffin
def extp(x,x0,y0):
    y = (y0[1]-y0[0])/(x0[1]-x0[0])*(x-x0[0]) + y0[0]
    return y

def scale_height(mass,temp):
    # mass: [g/mol]
    # temp: [K]
    na = 6.02e23 #[/mol]
    boltzmann_const = 1.38e-23 #[J/K]=[kg m^2/s^2 K]
    grav_const = 8.87 #[m/s^2]
    res = boltzmann_const*temp/(mass*1e-3/na)/grav_const #[m]
    return res

"""
calculate value of index = i+0.5
"""
def VarP2(x):
    return np.roll(x, -1)
    
"""
calculate value of index = i-0.5
"""
def VarM2(x):
    return np.roll(x, 1)

"""
calculate value of index = i+0.5
"""
def VarP(x):
    tmp = np.roll(x, -1)
    return 0.5*(x + tmp)
    
"""
calculate value of index = i-0.5
"""
def VarM(x):
    tmp = np.roll(x, 1)
    return 0.5*(x + tmp)

"""
calculate difference of values of indices = i, i+1
"""
def DifP(x):
    tmp = np.roll(x, -1)
    return (tmp - x)


"""
calculate difference of values of indices = i-1, i
"""
def DifM(x):
    tmp = np.roll(x, 1)
    return (x - tmp)

"""
z, Hは[km]で扱う
密度は[10^10 cm^-3]

"""
def MainCal(no, nco2, temp, Dcoef, Keddy, nlev, z):
    dzp = DifP(z)*1e3 #[cm]
    dzm = DifM(z)*1e3 #[cm]
    dTp = DifP(temp)
    dTm = DifM(temp)
    dnop = DifP(no)
    dnom = DifM(no)
    
    nco2p = VarP2(nco2) #[/cm^-3]
    nco2m = VarM2(nco2)
    nop = VarP2(no)
    nom = VarM2(no)
    
    mp = (44*nco2p+16*nop)/(nco2p+nop)
    mm = (44*nco2m+16*nom)/(nco2m+nom)
    
    Dp = VarP2(Dcoef)#*1e-4
    Dm = VarM2(Dcoef)#*1e-4
    Tp = VarP2(temp)
    Tm = VarM2(temp)
    
    Hop = scale_height(16,Tp)#*1e2
    Hom = scale_height(16,Tm)#*1e2
    Hap = scale_height(mp,Tp)#*1e2
    Ham = scale_height(mm,Tm)#*1e2
    
    phi1p = (Dp+Keddy)*dnop/dzp #ほぼ定数　D~1/n
    phi2p = Dp   *(1/Hop+1/Tp*dTp/dzp)*nop
    phi3p = Keddy*(1/Hap+1/Tp*dTp/dzp)*nop
    phip = phi1p+phi2p+phi3p
    
    phi1m = (Dm+Keddy)*dnom/dzm
    phi2m = Dm   *(1/Hom+1/Tm*dTm/dzm)*nom
    phi3m = Keddy*(1/Ham+1/Tm*dTm/dzm)*nom
    phim = phi1m+phi2m+phi3m

    dphi_dz = (phip-phim)/(0.5*(dzp+dzm))#*1e-4 #[/cm^-3 s]
    
    dphi_dz[0] = 0
    dphi_dz[nlev-1] = 0
#     dphi_dz[nlev-2] = 0
    
#     print(phi1p[150], phi2p[150], phi3p[150])
#     print(phi1m[150], phi2m[150], phi3m[150])
#     print(dphi_dz[150], phip[150], phim[150], dzp[150], dzm[150])
    #print(Hop)
    return dphi_dz

"""
MP: multiprocess
"""
def mp_func(funct_var):
    return funct_var[0](funct_var[1])
    
def MainCalMP(no, nco2, temp, Dcoef, Keddy, nlev, z, p):
    #p = Pool(3)
#     dzp, dTp, dnop = p.map(DifP, [z*1e5, temp, no])
#     dzm, dTm, dnom = p.map(DifM, [z*1e5, temp, no])
#     nco2p, nop, Dp, Tp = p.map(VarP, [nco2, no, Dcoef, temp])
#     nco2m, nom, Dm, Tm = p.map(VarM, [nco2, no, Dcoef, temp])
#     dzp, dTp, dnop, \
#     dzm, dTm, dnom, \
#     nco2p, nop, Dp, Tp, \
#     nco2m, nom, Dm, Tm = p.map(mp_func, [(DifP, z*1e5),(DifP, temp), (DifP, no),
#                                          (DifM, z*1e5),(DifM, temp), (DifM, no),
#                                          (VarP, nco2), (VarP, no), (VarP, Dcoef), (VarP, temp),
#                                          (VarM, nco2), (VarM, no), (VarM, Dcoef), (VarM, temp),
#                                         ])
    #p.close()
    dzp = DifP(z)*1e5 #[cm]
    dTp = DifP(temp)
    dnop = DifP(no)
    dzm = DifM(z)*1e5 #[cm]
    dTm = DifM(temp)
    dnom = DifM(no)
    
    nco2p = VarP(nco2)#[cm^-3]
    nop = VarP(no)
    Dp = VarP(Dcoef)
    Tp = VarP(temp)
    
    nco2m = VarM(nco2)
    nom = VarM(no)
    Dm = VarM(Dcoef)
    Tm = VarM(temp)
    
    mp = (44*nco2p+16*nop)/(nco2p+nop)
    mm = (44*nco2m+16*nom)/(nco2m+nom)
    
    Hop = scale_height(16,Tp)*1e2
    Hom = scale_height(16,Tm)*1e2
    Hap = scale_height(mp,Tp)*1e2
    Ham = scale_height(mm,Tm)*1e2
    
    phi1p = (Dp+Keddy)*dnop/dzp #ほぼ定数　D~1/n
    phi2p = Dp   *(1/Hop+1/Tp*dTp/dzp)*nop
    phi3p = Keddy*(1/Hap+1/Tp*dTp/dzp)*nop
    phip = phi1p+phi2p+phi3p
    
    phi1m = (Dm+Keddy)*dnom/dzm
    phi2m = Dm   *(1/Hom+1/Tm*dTm/dzm)*nom
    phi3m = Keddy*(1/Ham+1/Tm*dTm/dzm)*nom
    phim = phi1m+phi2m+phi3m

    dphi_dz = (phip-phim)/(0.5*(dzp+dzm))
    
    dphi_dz[0] = 0
    dphi_dz[nlev-1] = 0
    
    #print(phi1p[150], phi2p[150], phi3p[150])
    #print(Hop)
#     print(dnop[150])
    
    return dphi_dz