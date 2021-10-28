import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import pi
from astropy.convolution import convolve
import scipy.io as sio
import numpy.polynomial as poly



def calc_meridional_modes(ys,N=6,cm=2.8):
    """
    Calculates the first N normalised meridional modes for the baroclinic wave speed cm at the latitudes ys
    
    Inputs:
    
    ys (np.ndarray) : Latitudes at which to return the modes
    N (int, optional) : Number of modes to calculate, defaults to 6
    cm (float, optional) : Baroclinic wavespeed in m/s, defaults to 2.8 m/s
    
    Returns:
    V_modes (np.ndarray): Modes for meridional velocity at latitudes ys
    P_modes (np.ndarray): Modes for pressure at latitudes ys(at zonal wavenumber k=0)
    
    """
    # ys is where we want the modes sampled at. We define a more resolved y to actually calculate the modes.
    y = np.linspace(-15,15,100)
    
    # Find modes according to Blaker et al. 2021
    earth_radius = 6371000
    earth_freq = 2*pi/24/3600
    ny = y.shape[0]
    beta = 2*earth_freq/earth_radius  
                         
    # y is in degrees, change to meters                      
    ymetres = y*earth_radius*np.pi/180
     
    # normalise y    
    ym = np.sqrt(2*beta/cm)*ymetres
    dym = ym[1] - ym[0]
     
    # Define the hermite polys. Maximum number of modes is 8
    if N > 8:
        print('Maximum 8 nodes will be found')
        N = 8
    
    He = np.zeros((9,ny))
    He[0,:] = 1
    He[1,:] = ym
    He[2,:] = ym**2 - 1
    He[3,:] = ym**3 - 3*ym
    He[4,:] = ym**4 - 6*ym**2 + 3
    He[5,:] = ym**5 - 10*ym**3 + 15*ym
    He[6,:] = ym**6 - 15*ym**4 + 45*ym**2 - 15
    He[7,:] = ym**7 - 21*ym**5 + 105*ym**3 - 105*ym
    He[8,:] = ym**8 - 28*ym**6 + 210*ym**4 - 420*ym**2 + 105
    
    # Weight to find hermite function
    phi = He*np.tile(np.expand_dims(np.exp(-ym**2/4),0),[9,1])

    
    # Now find pressure modes, normalise them too.
    # For the hermite functions phi_m = exp(-y^2/4)He_m, d phi_m / dy = 0.5*[m*phi_{m-1} - phi_{m+1}]
    
    P = np.zeros((8,ny))
    for m in range(0,8):
        if m == 0:
            P[0,:] = -0.5*(phi[1,:])
        else:
            P[m,:] = 0.5*(m*phi[m-1,:] - phi[m+1,:])
            
    # Now normalise:
    phi /= np.sqrt(np.tile(np.expand_dims(((phi**2).sum(axis = 1)*dym),1),[1,ny]))

    # Now normalise the P modes too
    P /= np.sqrt(np.tile(np.expand_dims(((P**2).sum(axis = 1)*dym),1),[1,ny]))
    
    # V_modes
    V_modes = phi[:N,:]
    P_modes = P[:N,:]
    
    V_modes_sampled = np.zeros((N,ys.shape[0]))
    P_modes_sampled = np.zeros((N,ys.shape[0]))
    
    # Now interpolate onto the given sample points
    for m in range(0,N):
        V_modes_sampled[m,:] = np.interp(ys, y, V_modes[m,:], left=np.nan, right=np.nan)
        P_modes_sampled[m,:] = np.interp(ys, y, P_modes[m,:], left=np.nan, right=np.nan)
    
    
    return V_modes_sampled, P_modes_sampled
        
        
def find_predicted_freqs(nmodes,k_wavenumbers=0,lon_lims=[0,360],lat_lims=[-12,12],average_lon = True):
    """
    Calculates the predicted wave frequencies for the first and second baroclinic modes as per the wave dispersion relation on a beta plane.
    
    Inputs:
    
    nmodes (int) : Number of meridional modes
    k_wavenumbers (int==0 OR np.ndarray, optional) : zonal wavenumbers at which to calculate frequencies. Should be in deg^{-1}. Either 0 or an array containing wavenumbers. Defaults to 0. 
    lon_lims (list, optional) : Longitude limits for baroclinic wavespeeds used. Should be a list of form [lon_min, lon_max], defaults to [0,360].
    lat_lims (list, optional) : Latitude limits for baroclinic wavespeeds used. Should be a list of form [lat_min, lat_max], defaults to [-12,12].
    average_lon (bool, optional) : If True, calculates frequency using a zonally averaged baroclinic wavespeed. If False, calculates at each longitude.
    
    If k_wavenumbers is not zero, and average_lon is False, average_lon will be changed to True.
    
    Returns:
    
    freqbc1 (np.ndarray) : frequencies of first baroclinic mode. If k_wavenumbers is not zero, has a dimension corresponding to wavenumber. If average_lon is False, has a dimension corresponding to longitude. Both options are not compatible
    freqbc2 (np.ndarray) : frequencies of second baroclinic mode. Dimensions as for freqbc1. 
    c1 (float or np.ndarray) : Meridionally averaged first baroclinic wave speed. Has dimensions of longitude if average_lon = False
    c2 (float or np.ndarray) : Meridionally averaged second baroclinic wave speed. Has dimensions of longitude if average_lon = False
    lon_freqs (np.ndarray) : Longitudes at which frequencies are found. None if average_lon = True
    k_wavenumbers (np.ndarray) : Zonal wavenumbers at which frequencies are found. 
    
    """
    
    # Load in baroclinic wave speed maps
    contents=sio.loadmat('../../data/baroclinic_wave_speeds/baroclinic_wave_speeds.mat')
    c1global = np.squeeze(contents['c1'])
    c2global = np.squeeze(contents['c2'])
    clat = np.squeeze(contents['lat'])
    clon = np.squeeze(contents['lon'])
    
    # ----------------------------------------------
    
    c1 = np.squeeze(np.nanmean(c1global[(clat <= lat_lims[1])&(clat >lat_lims[0]),:],axis=0))
    c1 = c1[(clon > lon_lims[0])&(clon <= lon_lims[-1])]
    c2 = np.squeeze(np.nanmean(c2global[(clat <= lat_lims[1])&(clat >lat_lims[0]),:],axis=0))
    c2 = c2[(clon > lon_lims[0])&(clon <= lon_lims[-1])]
    lon_freqs = clon[(clon > lon_lims[0])&(clon <= lon_lims[-1])]
    
    nvec = np.arange(0,nmodes,1)
    
    earth_radius = 6371000
    earth_freq = 2*pi/24/3600
    beta = 2*earth_freq/earth_radius  

    # If only looking for zonally uniform frequency, can simplify the dispersion relation:
    if np.isscalar(k_wavenumbers):
        if k_wavenumbers == 0:  
            if average_lon:
                
                c1 = np.nanmean(c1)
                c2 = np.nanmean(c2)
                
                # timescale in seconds:
                Te1 = 1/np.sqrt(beta*c1)
                Te2 = 1/np.sqrt(beta*c2)

                # timescale in days:
                Te1d = Te1/24/3600
                Te2d = Te2/24/3600
                
                freqbc1 = np.sqrt(2*nvec+1)/Te1d/2/np.pi
                freqbc2 = np.sqrt(2*nvec+1)/Te2d/2/np.pi
                
                lon_freqs = None
            else:
                # timescale in seconds:
                Te1 = 1/np.sqrt(beta*c1)
                Te2 = 1/np.sqrt(beta*c2)

                # timescale in days:
                Te1d = Te1/24/3600
                Te2d = Te2/24/3600
                
                freqbc1 = np.zeros((nvec.shape[0],lon_freqs.shape[0]))
                freqbc2 = np.zeros((nvec.shape[0],lon_freqs.shape[0]))

                for ilon in range(0,lon_freqs.shape[0]):
                    freqbc1[:,ilon] = np.sqrt(2*nvec+1)/Te1d[ilon]/2/np.pi
                    freqbc2[:,ilon] = np.sqrt(2*nvec+1)/Te2d[ilon]/2/np.pi         

            
        else:
            print('k_wavenumbers should be 0 or a numpy array')

    else: # Here, we want to find the dispersion relation for different k. 
          # Do this only for zonally averaged wave speeds.
        
        if average_lon == False:
            print('Finding the zonally averaged dispersion relation')
            average_lon = True
            
        # Just use zonally averaged wave speeds
        c1 = np.nanmean(c1)   
        c2 = np.nanmean(c2)   
        
        # timescale in seconds:
        Te1 = 1/np.sqrt(beta*c1)
        Te2 = 1/np.sqrt(beta*c2)

        # timescale in days:
        Te1d = Te1/24/3600
        Te2d = Te2/24/3600
        
        # lengthscale in metres
        Le1 = np.sqrt(c1/beta)
        Le2 = np.sqrt(c2/beta)
        
        # Wavenumber vector in metres^{-1}
        metres_in_degree_eq = 2*pi*earth_radius/360       
        kvecm = k_wavenumbers/metres_in_degree_eq

        # Now nondimensionalise k
        kvecnd1 = kvecm*Le1*2*pi
        kvecnd2 = kvecm*Le2*2*pi
        
        # Equation for nondimensional freq w: w^3 - (k^2 + (2m+1))*w -k = 0
        
        # First do BC1:
        freqbc1_nd = np.zeros((3,nvec.shape[0],kvecnd1.shape[0]))
        for i, n in enumerate(nvec):
            for ik,k in enumerate(kvecnd1):
                if n == 0: # Take out the w = -k solution, not finite
                    p = poly.Polynomial([-1, -k, 1])
                    freqbc1_nd[:,i,ik] = np.append(p.roots(),np.nan)
                else:
                    p = poly.Polynomial([-k, -(k**2 + (2*n+1)),0, 1])
                    freqbc1_nd[:,i,ik] = p.roots()

        # Re-dimensionalise frequency
        freqbc1 = freqbc1_nd/Te1d/2/np.pi

        # Then do BC2:
        freqbc2_nd = np.zeros((3,nvec.shape[0],kvecnd2.shape[0]))
        for i, n in enumerate(nvec):
            for ik,k in enumerate(kvecnd2):
                if n == 0: # Take out the w = -k solution, not finite
                    p = poly.Polynomial([-1, -k, 1])
                    freqbc2_nd[:,i,ik] = np.append(p.roots(),np.nan)
                else:
                    p = poly.Polynomial([-k, -(k**2 + (2*n+1)),0, 1])
                    freqbc2_nd[:,i,ik] = p.roots()

        # Re-dimensionalise frequency
        freqbc2 = freqbc2_nd/Te2d/2/np.pi

        return freqbc1, freqbc2, c1, c2, lon_freqs, k_wavenumbers
        
    
    
    
    
    

