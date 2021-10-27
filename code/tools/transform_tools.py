import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import pi
from astropy.convolution import convolve


def least_squares_spectrum_t_multi(data, t, min_period=2, max_period=370, NSR=35, reconstruct_min_period='default'):
    
    if len(data.shape) == 1:
        return least_squares_spectrum_t(data, t, min_period, max_period, NSR,reconstruct_min_period)
    elif len(data.shape) == 2:
        for ii in range(0,data.shape[1]):
            omvec, power, fitted_data = least_squares_spectrum_t(data[:,ii], t, min_period, max_period, NSR,reconstruct_min_period)
            if ii == 0:
                power_2D = np.expand_dims(power,1)
                fitted_data_2D = np.expand_dims(fitted_data,1)
            else:
                power_2D = np.concatenate((power_2D,np.expand_dims(power,1)),axis=1)
                fitted_data_2D = np.concatenate((fitted_data_2D,np.expand_dims(fitted_data,1)),axis=1)
        return omvec, power_2D, fitted_data_2D    
    elif len(data.shape) == 3:
        for ii in range(0,data.shape[2]):
            for jj in range(0,data.shape[1]):
                omvec, power, fitted_data = least_squares_spectrum_t(data[:,jj,ii], t, min_period, max_period, NSR,reconstruct_min_period)
                if jj == 0:
                    power_2D = np.expand_dims(power,1)
                    fitted_data_2D = np.expand_dims(fitted_data,1)
                else:
                    power_2D = np.concatenate((power_2D,np.expand_dims(power,1)),axis=1)
                    fitted_data_2D = np.concatenate((fitted_data_2D,np.expand_dims(fitted_data,1)),axis=1)
            if ii == 0:
                power_3D = np.expand_dims(power_2D,2)
                fitted_data_3D = np.expand_dims(fitted_data_2D,2)
            else:
                power_3D = np.concatenate((power_3D,np.expand_dims(power_2D,2)),axis=2)
                fitted_data_3D = np.concatenate((fitted_data_3D,np.expand_dims(fitted_data_2D,2)),axis=2)
        return omvec, power_3D, fitted_data_3D
        
        
def least_squares_spectrum_t(data, t, min_period, max_period, NSR=35, reconstruct_min_period='default'):
    
    # Data can have shape [nt, ny, nx] or [nt, nx or ny] 
    om_min = 0
    om_max = 1/min_period
    delta_om = 1/max_period
    omvec = np.arange(om_min,om_max,delta_om )

        
    # get a copy of the data
    d = np.squeeze(np.copy(data))

    if np.sum(np.isnan(d)) < d.shape[0]: # If not all nans
        
        # Now vectorize if necessary, and remove mean
        dvec = np.ravel(d)
        dvec -= np.nanmean(dvec)

        # Remove nans from time and dvec
        t_rm = t[~np.isnan(dvec)]
        dvec_rm = dvec[~np.isnan(dvec)]

        nd = dvec_rm.shape[0]
        nm = 2*omvec.shape[0]
        E = np.zeros((nd,nm))
        for m in range(0,int(nm/2)):
            E[:,2*m] = np.sin(2*pi*omvec[m]*t_rm)
            E[:,2*m+1] = np.cos(2*pi*omvec[m]*t_rm)


        R = np.eye(nm)*NSR
        Cmm = np.dot(np.transpose(E),E) + R
        Cmd = np.transpose(E)
        K = np.dot(np.linalg.inv(Cmm),Cmd)
        amps = np.dot(K,dvec_rm)
        
        # Allow reconstruction from a limited number of modes (for high pass)
        if reconstruct_min_period != 'default':
            nm_lim = np.argmin(np.abs(omvec - 1/reconstruct_min_period))+1
            fitted_d = np.dot(E[:,:2*nm_lim],amps[:2*nm_lim])
            sine_coeffs = amps[0::2]
            cos_coeffs = amps[1::2]
            sine_coeffs[nm_lim:] = 0      
            cos_coeffs[nm_lim:] = 0
        else:
            fitted_d = np.dot(E,amps)
            sine_coeffs = amps[0::2]
            cos_coeffs = amps[1::2]
        power = np.sqrt((sine_coeffs**2 + cos_coeffs**2)) # by Adam's defn of normalisation


        fitted_data = np.nan*np.ones_like(dvec)
        fitted_data[~np.isnan(dvec)] = fitted_d

    else:
        power = np.nan*omvec
        fitted_data = data

    return omvec, power, fitted_data


def least_squares_spectrum_t_y(data, t, y, min_period, max_period, y_modes, NSR=35, max_period_cutoff ='default'):
    
    # Data should have shape [nt, ny, nx] or [nt, ny]
    
    if max_period_cutoff == 'default':
        om_min = 0
    else:
        om_min = 1/max_period_cutoff
    om_max = 1/min_period
    delta_om = 1/max_period
    omvec = np.arange(om_min,om_max,delta_om )
    
    nmodes = y_modes.shape[0]
    
    # If data doesn't have an x dimension:
    if len(data.shape) == 2:
        
        
        # remove mean at each location
        d = np.copy(data) - np.nanmean(data,axis = 0)
        data_var = np.nanvar(d)
        
        iy = np.arange(0,y.shape[0],1).astype(int)
        il = np.arange(0,nmodes,1).astype(int)

        # Create meshgrid real and spectral space
        iL, OM = np.meshgrid(il, omvec)
        iY, T = np.meshgrid(iy, t)

        # create vectors from all 2D fields
        ilvec = np.ravel(iL)
        OMvec = np.ravel(OM)
        dvec = np.ravel(d)
        Tvec = np.ravel(T)
        iyvec = np.ravel(iY)


        # remove nans from all real fields
        dvec_rm = dvec[~np.isnan(dvec)]
        Tvec_rm = Tvec[~np.isnan(dvec)]
        iyvec_rm = iyvec[~np.isnan(dvec)]

        # Create modes
        modevec = np.zeros((ilvec.shape[0],iyvec_rm.shape[0]))
        for i, iiy in enumerate(iyvec_rm):
            for j, iil in enumerate(ilvec):
                modevec[j,i] = y_modes[iil,iiy]


        nm = 2*OMvec.shape[0]
        nd = dvec_rm.shape[0]

        E = np.zeros((nd,nm))
        for m in range(0,int(nm/2)):
            E[:,2*m] = np.sin(2*pi*OMvec[m]*Tvec_rm)*modevec[m,:]
            E[:,2*m+1] = np.cos(2*pi*OMvec[m]*Tvec_rm)*modevec[m,:]
       
        
        R = np.eye(nm)*NSR
        Cmm = np.dot(np.transpose(E),E) + R
        Cmd = np.transpose(E)
        K = np.dot(np.linalg.inv(Cmm),Cmd)
        amps = np.dot(K,dvec_rm)
        fitted_d = np.dot(E,amps)

        sine_coeffs = amps[0::2]
        cos_coeffs = amps[1::2]
        power = (sine_coeffs**2 + cos_coeffs**2)

        # restore to 2D fields
        sine_coeffs_2D = np.reshape(sine_coeffs,(OM.shape[0],OM.shape[1]))
        cos_coeffs_2D = np.reshape(cos_coeffs,(OM.shape[0],OM.shape[1]))
        power_2D = np.sqrt(sine_coeffs_2D**2 + cos_coeffs_2D**2)


        # Introduce nans back into fitted data
        fitted_data = np.nan*np.ones_like(dvec)
        fitted_data[~np.isnan(dvec)] = fitted_d

        fitted_data_2D = np.reshape(fitted_data,(T.shape[0],T.shape[1]))

        return omvec, power_2D, fitted_data_2D
    
    # If data does have an x dimension:
    if len(data.shape) == 3:
        nx = data.shape[2]
        for ix in tqdm(range(0,nx)):
            data_ix = np.copy(data[:,:,ix])
            
            # remove mean at each location
            d = np.copy(data_ix) - np.nanmean(data_ix,axis = 0)
            data_var = np.nanvar(d)

            iy = np.arange(0,y.shape[0],1).astype(int)
            il = np.arange(0,nmodes,1).astype(int)

            # Create meshgrid real and spectral space
            iL, OM = np.meshgrid(il, omvec)
            iY, T = np.meshgrid(iy, t)

            # create vectors from all 2D fields
            ilvec = np.ravel(iL)
            OMvec = np.ravel(OM)
            dvec = np.ravel(d)
            Tvec = np.ravel(T)
            iyvec = np.ravel(iY)


            # remove nans from all real fields
            dvec_rm = dvec[~np.isnan(dvec)]
            Tvec_rm = Tvec[~np.isnan(dvec)]
            iyvec_rm = iyvec[~np.isnan(dvec)]

            # Create modes
            modevec = np.zeros((ilvec.shape[0],iyvec_rm.shape[0]))
            for i, iiy in enumerate(iyvec_rm):
                for j, iil in enumerate(ilvec):
                    modevec[j,i] = y_modes[iil,iiy]


            nm = 2*OMvec.shape[0]
            nd = dvec_rm.shape[0]

            E = np.zeros((nd,nm))
            for m in range(0,int(nm/2)):
                E[:,2*m] = np.sin(2*pi*OMvec[m]*Tvec_rm)*modevec[m,:]
                E[:,2*m+1] = np.cos(2*pi*OMvec[m]*Tvec_rm)*modevec[m,:]
            
            R = np.eye(nm)*NSR
            Cmm = np.dot(np.transpose(E),E) + R
            Cmd = np.transpose(E)
            K = np.dot(np.linalg.inv(Cmm),Cmd)
            amps = np.dot(K,dvec_rm)
            fitted_d = np.dot(E,amps)

            sine_coeffs = amps[0::2]
            cos_coeffs = amps[1::2]
            power = (sine_coeffs**2 + cos_coeffs**2)

            # restore to 2D fields
            sine_coeffs_2D = np.reshape(sine_coeffs,(OM.shape[0],OM.shape[1]))
            cos_coeffs_2D = np.reshape(cos_coeffs,(OM.shape[0],OM.shape[1]))
            power_2D = np.sqrt(sine_coeffs_2D**2 + cos_coeffs_2D**2)

            # Introduce nans back into fitted data
            fitted_data = np.nan*np.ones_like(dvec)
            fitted_data[~np.isnan(dvec)] = fitted_d

            fitted_data_2D = np.reshape(fitted_data,(T.shape[0],T.shape[1]))

            if ix==0:
                power_3D = np.expand_dims(power_2D,2)
                fitted_data_3D = np.expand_dims(fitted_data_2D,2)
            else:
                power_3D = np.concatenate((power_3D,np.expand_dims(power_2D,2)),axis=2)
                fitted_data_3D = np.concatenate((fitted_data_3D,np.expand_dims(fitted_data_2D,2)),axis=2)

        return omvec,  power_3D, fitted_data_3D
                
    

    
    
    
    
    


