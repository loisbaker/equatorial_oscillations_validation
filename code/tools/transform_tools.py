import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import pi
from astropy.convolution import convolve
from tqdm import tqdm


def least_squares_spectrum_t_multi(data, t, min_period=2, max_period=370, NSR=35, reconstruct_min_period='default'):
    ''' 
    Finds least squares fit of temporal modes to time series data when the input field has multiple dimensions (time must be first dimension). Uses least_squares_spectrum_t.
    
    Inputs:
    
    data (np.ndarray): input time series, up to 3 dimensions
    t (np.ndarray): time vector, length data.shape[0]
    min_period (float or int, optional): minimum possible period of temporal modes. Defaults to 2.
    max_period (float or int, optional): maximum period of temporal modes (= length of time series). Defaults to 370.
    NSR (float or int, optional): Noise to signal ratio
    reconstruct_min_period (optional): Minimum period with which to reconstruct signal (effectively a low pass filter). Defaults to 'default', which gives no cutoff. 
    
    Returns:
    
    freqs (np.ndarray): frequencies at which power has been found
    power (np.ndarray): estimated power at each frequency. Dimensions are [frequency, (latitude), (longitude)]
    fitted_data (np.ndarray) : reconstructed field, same shape as input data. For a perfect fit, fitted_data = data.
    
    '''
    if len(data.shape) == 1:
        return least_squares_spectrum_t(data, t, min_period, max_period, NSR,reconstruct_min_period)
    elif len(data.shape) == 2:
        for ii in range(0,data.shape[1]):
            freqs, power, fitted_data = least_squares_spectrum_t(data[:,ii], t, min_period, max_period, NSR,reconstruct_min_period)
            if ii == 0:
                power_2D = np.expand_dims(power,1)
                fitted_data_2D = np.expand_dims(fitted_data,1)
            else:
                power_2D = np.concatenate((power_2D,np.expand_dims(power,1)),axis=1)
                fitted_data_2D = np.concatenate((fitted_data_2D,np.expand_dims(fitted_data,1)),axis=1)
        return freqs, power_2D, fitted_data_2D    
    elif len(data.shape) == 3:
        for ii in range(0,data.shape[2]):
            for jj in range(0,data.shape[1]):
                freqs, power, fitted_data = least_squares_spectrum_t(data[:,jj,ii], t, min_period, max_period, NSR,reconstruct_min_period)
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
        return freqs, power_3D, fitted_data_3D
        
        
def least_squares_spectrum_t(data, t, min_period=2, max_period=370, NSR=35, reconstruct_min_period='default'):
    ''' 
    Finds least squares fit of temporal modes to time series data.
    
    Inputs:
    
    data (np.ndarray): input time series, one dimension
    t (np.ndarray): time vector, same shape as data
    min_period (float or int, optional): minimum possible period of temporal modes. Defaults to 2.
    max_period (float or int, optional): maximum period of temporal modes (= length of time series). Defaults to 370.
    NSR (float or int, optional): Noise to signal ratio
    reconstruct_min_period (optional): Minimum period with which to reconstruct signal (effectively a low pass filter). Defaults to 'default', which gives no cutoff. 
    
    Returns:
    
    freqs (np.ndarray): frequencies at which power has been found
    power (np.ndarray): estimated power at each frequency
    fitted_data (np.ndarray) : reconstructed field, same shape as input data. For a perfect fit, fitted_data = data.
    
    '''
    # Data must have one dimension (time)
    if len(data.shape) > 1:
        raise TypeError("data must be one dimensional (time) - try using `least_squares_spectrum_t_multi'")
    else:
        
        freq_min = 0
        freq_max = 1/min_period
        delta_freq = 1/max_period
        freqs = np.arange(freq_min,freq_max,delta_freq)


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
            nm = 2*freqs.shape[0]
            E = np.zeros((nd,nm))
            for m in range(0,int(nm/2)):
                E[:,2*m] = np.sin(2*pi*freqs[m]*t_rm)
                E[:,2*m+1] = np.cos(2*pi*freqs[m]*t_rm)


            R = np.eye(nm)*NSR
            Cmm = np.dot(np.transpose(E),E) + R
            Cmd = np.transpose(E)
            K = np.dot(np.linalg.inv(Cmm),Cmd)
            amps = np.dot(K,dvec_rm)

            # Allow reconstruction from a limited number of modes (for high pass)
            if reconstruct_min_period != 'default':
                nm_lim = np.argmin(np.abs(freqs - 1/reconstruct_min_period))+1
                fitted_d = np.dot(E[:,:2*nm_lim],amps[:2*nm_lim])
                sine_coeffs = amps[0::2]
                cos_coeffs = amps[1::2]
                sine_coeffs[nm_lim:] = 0      
                cos_coeffs[nm_lim:] = 0
            else:
                fitted_d = np.dot(E,amps)
                sine_coeffs = amps[0::2]
                cos_coeffs = amps[1::2]
            power = np.sqrt((sine_coeffs**2 + cos_coeffs**2)) # same definition as power in Blaker et al. 2021


            fitted_data = np.nan*np.ones_like(dvec)
            fitted_data[~np.isnan(dvec)] = fitted_d

        else:
            power = np.nan*freqs
            fitted_data = np.copy(data)

        return freqs, power, fitted_data


def least_squares_spectrum_t_y(data, t, y, y_modes, min_period=2, max_period=370, NSR=35, max_period_cutoff ='default'):
    ''' 
    Finds least squares fit of temporal and meridional modes to 2D (time, latitude) or 3D (time, latitude, longtiude) fields.
    
    Inputs:
    
    data (np.ndarray): input data, two or three dimensions. Tiem should be first dimension, then latitude, then (optionally) longitude.
    t (np.ndarray): time vector, length data.shape[0]
    y (np.ndarray): latitude vector, length data.shape[1]
    min_period (float or int, optional): minimum possible period of temporal modes. Defaults to 2.
    max_period (float or int, optional): maximum period of temporal modes (= length of time series). Defaults to 370.
    y_modes (np.ndarray): meridional modes, first dimension is mode number, second dimension is latitude (length data.shape[1])
    NSR (float or int, optional): Noise to signal ratio
    max_period_cutoff (optional): Maximum period to use in frequency fit. Not necessarily equal to max_period, as this should be the maximum possible period.  Defaults to 'default', which gives no cutoff. 
    
    Returns:
    
    freqs (np.ndarray): frequencies at which power has been found
    power (np.ndarray): estimated power at each frequency. Dimensions are [frequency, mode number, (longitude)]
    fitted_data (np.ndarray) : reconstructed field, same shape as input data. For a perfect fit, fitted_data = data. 
    
    '''
    # Data should have shape [nt, ny, nx] or [nt, ny]
    
    if max_period_cutoff == 'default':
        freq_min = 0
    else:
        freq_min = 1/max_period_cutoff
    freq_max = 1/min_period
    delta_freq = 1/max_period
    freqs = np.arange(freq_min,freq_max,delta_freq )
    
    nmodes = y_modes.shape[0]
    
    # If data doesn't have an x dimension:
    if len(data.shape) == 2:

        # remove mean at each location
        d = np.copy(data) - np.nanmean(data,axis = 0)
        data_var = np.nanvar(d)
        
        iy = np.arange(0,y.shape[0],1).astype(int)
        il = np.arange(0,nmodes,1).astype(int)

        # Create meshgrid real and spectral space
        iL, FREQ = np.meshgrid(il, freqs)
        iY, T = np.meshgrid(iy, t)

        # create vectors from all 2D fields
        ilvec = np.ravel(iL)
        freqvec = np.ravel(FREQ)
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


        nm = 2*freqvec.shape[0]
        nd = dvec_rm.shape[0]

        E = np.zeros((nd,nm))
        for m in range(0,int(nm/2)):
            E[:,2*m] = np.sin(2*pi*freqvec[m]*Tvec_rm)*modevec[m,:]
            E[:,2*m+1] = np.cos(2*pi*freqvec[m]*Tvec_rm)*modevec[m,:]
       
        
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
        sine_coeffs_2D = np.reshape(sine_coeffs,(FREQ.shape[0],FREQ.shape[1]))
        cos_coeffs_2D = np.reshape(cos_coeffs,(FREQ.shape[0],FREQ.shape[1]))
        power_2D = np.sqrt(sine_coeffs_2D**2 + cos_coeffs_2D**2)


        # Introduce nans back into fitted data
        fitted_data = np.nan*np.ones_like(dvec)
        fitted_data[~np.isnan(dvec)] = fitted_d

        fitted_data_2D = np.reshape(fitted_data,(T.shape[0],T.shape[1]))

        return freqs, power_2D, fitted_data_2D
    
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
            iL, FREQ = np.meshgrid(il, freqs)
            iY, T = np.meshgrid(iy, t)

            # create vectors from all 2D fields
            ilvec = np.ravel(iL)
            freqvec = np.ravel(FREQ)
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


            nm = 2*freqvec.shape[0]
            nd = dvec_rm.shape[0]

            E = np.zeros((nd,nm))
            for m in range(0,int(nm/2)):
                E[:,2*m] = np.sin(2*pi*freqvec[m]*Tvec_rm)*modevec[m,:]
                E[:,2*m+1] = np.cos(2*pi*freqvec[m]*Tvec_rm)*modevec[m,:]
            
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
            sine_coeffs_2D = np.reshape(sine_coeffs,(FREQ.shape[0],FREQ.shape[1]))
            cos_coeffs_2D = np.reshape(cos_coeffs,(FREQ.shape[0],FREQ.shape[1]))
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

        return freqs,  power_3D, fitted_data_3D
                
    

    
    
    
    
    


