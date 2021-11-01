import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import pi
from astropy.convolution import convolve
from tqdm import tqdm


def smooth(field,n,axis=0):
    ''' 
    Convolves a 1-3D field along a given axis, using astropy's convolve function (deals well with gappy data).
    
    Inputs:
    
    field (np.ndarray): input field, 1-3 dimensions
    n (int): length of smoothing window, should be odd
    axis (int, optional): axis to smooth along. Defaults to 0. 
    
    Returns:
    
    field_sm (np.ndarray): smoothed field
    '''
    ndims = len(field.shape)
    if ndims == 1:
        field_sm = convolve(field,np.ones(n)/n,boundary='extend')
    elif ndims == 2:
        field_sm = np.zeros_like(field)
        if axis == 0:
            for i in range(0,field.shape[1]):
                field_sm[:,i] = convolve(field[:,i],np.ones(n)/n,boundary='extend')
        elif axis == 1:
            for i in range(0,field.shape[0]):
                field_sm[i,:] = convolve(field[i,:],np.ones(n)/n,boundary='extend')
        else:
            raise ValueError(f"axis {axis} isn't less than the number of dimensions")
    elif ndims == 3:
        field_sm = np.zeros_like(field)
        if axis == 0:
            for i in range(0,field.shape[1]):
                for j in range(0,field.shape[2]):
                    field_sm[:,i,j] = convolve(field[:,i,j],np.ones(n)/n,boundary='extend')
        elif axis == 1:
            for i in range(0,field.shape[0]):
                for j in range(0,field.shape[2]):
                    field_sm[i,:,j] = convolve(field[i,:,j],np.ones(n)/n,boundary='extend')
        elif axis == 2:
            for i in range(0,field.shape[0]):
                for j in range(0,field.shape[1]):
                    field_sm[i,j,:] = convolve(field[i,j,:],np.ones(n)/n,boundary='extend')
            
        else:
            raise ValueError(f"axis {axis} isn't less than the number of dimensions")
        
    return field_sm
        
def load_TAO(NEMO_year=False):
    """
    Loads in the TAO dataset, does some initial data processing
    
    Inputs:
    
    NEMO_year (bool, optional): returns dataset corresponding to the same year as the NEMO simulation 
                                if True, otherwise returns 26 year record. Defaults to False.
    
    Returns:
    
    t (np.ndarray) : time in days
    lat (np.ndarray) : latitude
    lon (np.ndarray) : longitude
    lon_TAO_midpoints (np.ndarray) : longitude of midpoints between moorings
    D (np.ndarray) : dynamic height
    ds (xarray.Dataset) : full dataset containing TAO data
    
    """
    
    # Load in data from moorings 
    ds = np.squeeze(xr.load_dataset('../../data/TAO/dyn_xyt_dy.cdf'))
    
    # make a dynamic height D with high values replaced by nans
    D = xr.where(np.abs(ds.DYN_13) > 1e10, np.nan, ds.DYN_13)
    ds['D'] = D
    
    # Let's cut the 3 westmost zonal locations - no southern hemisphere data
    ds = ds.isel({'lon':np.arange(3,11,1)})
    
    # Remove a dodgy value (there aren't too many)
    ds.D[4949,5,5] = np.nan
    
    if NEMO_year:     
        ds = ds.isel({'time':range(4838,4838+370)})
    
    lat = ds.lat.values
    lon = ds.lon.values

    t = np.arange(0,D.shape[0],1)
    
    D = ds.D.values
    
    # Define the midpoints between the moorings
    lon_TAO_midpoints = np.array([158.5,172.5,185,197.5,212.5,227.5,242.5,257.5,272.5])
    
    return t, lat, lon, lon_TAO_midpoints, D, ds
    
    
def load_NEMO(daily_mean=True,lons='default',lats='default',lon_lims='default',winds=False):
    """
    Loads in the NEMO dataset, does some initial data processing
    
    Inputs:
    
    daily_mean (bool, optional) : returns a daily mean dataset if true, otherwise 4 hourly. Defaults to True.
    lons (np.ndarray, optional) : longitudes at which to return data, should be between 0 and 360.
    lats (np.ndarray, optional) : latitudes at which to return data
    lon_lims (list, optional) : longitude limits for data (overridden by lats if specified). Given as a list [lon_min, lon_max].
    winds (bool, optional) : If True, return equatorial wind stress data.
    
    Returns:
   
    t (np.ndarray) : time in days
    lat (np.ndarray) : latitude
    lon (np.ndarray) : longitude
    D (np.ndarray) : dynamic height
    ds (xarray.Dataset) : full dataset containing NEMO data
    tau_x (np.ndarray) : returned if winds=True, contains zonal component of wind stress at equator
    tau_y (np.ndarray) : returned if winds=True, contains meridional component of wind stress at equator
    
    """
    # Load in NEMO dataset
    ds = np.squeeze(xr.load_dataset('../../data/NEMO/VN206HF_4h_hdy500m.nc'))
    
    if daily_mean:
        ds_sub = ds.isel({'time_counter':np.arange(0,ds.time_counter.shape[0],6)})
        D = ds_sub.sohdy.copy()
        for i in np.arange(0,370,1)*6:
            ind = int(i/6)
            D[ind,:,:] = ds.sohdy[i:i+6,:,:].mean(axis = 0)
        ds_sub['D'] = D
        ds = ds_sub
    # Change lons to between 0 and 360  
    ds['nav_lon'] = xr.where(ds.nav_lon < 0, ds.nav_lon+360,ds.nav_lon)
    
    # Set lat at equator to zero, as it's nan at the moment
    ds.nav_lat[49,:] = 0
    lon_full = ds.nav_lon[50,:].values
    lat_full = ds.nav_lat[:,1].values
    
    # Deal with missing values and convert into centimetres
    ds['D'] = xr.where(np.abs(ds.D) > 1e10, np.nan, ds.D)*100

    
    # Get the right locations. First change lons
    if lons != 'default':
        ilons = np.zeros(lons.shape[0]).astype(int)
        if np.any(lons < 0):
            print('lons input should be between 0 and 360')
            lons[lons <0] += 360
        for il in range(0,lons.shape[0]):
            ilons[il] = int(np.nanargmin(np.abs(lons[il] - lon_full)))
            
        ds = ds.isel({'x':ilons})
    elif lon_lims != 'default':     
        ds =ds.sel(x=ds.x[(lon_full > lon_lims[0])&(lon_full < lon_lims[1])])

    if lats != 'default':
        ilats = np.zeros(lats.shape[0]).astype(int)
        for il in range(0,lats.shape[0]):
            ilats[il] = int(np.nanargmin(np.abs(lats[il] - lat_full))) 
        ds = ds.isel({'y':ilats})
    
    t = np.arange(0,370,1)
    lon = ds.nav_lon[0,:].values
    lat = ds.nav_lat[:,1].values
    ds = np.squeeze(ds)
    D = ds.D.values
    
    
    if winds: # Then load in the wind stresses too. These have already been cropped to just include the equator.
        ds_tauy = np.squeeze(xr.open_dataset('../../data/NEMO/VN206HF_4h_sometauy_equator.nc'))
        ds_taux = np.squeeze(xr.open_dataset('../../data/NEMO/VN206HF_4h_sozotaux_equator.nc'))
        if daily_mean:
            ds_tauy_sub = ds_tauy.isel({'time_counter':np.arange(0,ds_tauy.time_counter.shape[0],6)})
            ds_taux_sub = ds_taux.isel({'time_counter':np.arange(0,ds_taux.time_counter.shape[0],6)})
            taux = ds_taux_sub.sozotaux.copy()
            tauy = ds_tauy_sub.sometauy.copy()
            for i in np.arange(0,370,1)*6:
                ind = int(i/6)
                taux[ind,:] = ds_taux.sozotaux[i:i+6,:].mean(axis = 0)
                tauy[ind,:] = ds_tauy.sometauy[i:i+6,:].mean(axis = 0)
            ds_taux_sub['tau_x'] = taux
            ds_tauy_sub['tau_y'] = tauy
            ds_taux = ds_taux_sub
            ds_tauy = ds_tauy_sub
       

        # Deal with missing values
        ds_taux['tau_x'] = xr.where(np.abs(ds_taux.tau_x) > 1e10, np.nan, ds_taux.tau_x)
        ds_tauy['tau_y'] = xr.where(np.abs(ds_tauy.tau_y) > 1e10, np.nan, ds_tauy.tau_y)


        # Get the right locations. First change lons
        if lons != 'default':
            ilons = np.zeros(lons.shape[0]).astype(int)
            if np.any(lons < 0):
                print('lons input should be between 0 and 360')
                lons[lons <0] += 360
            for il in range(0,lons.shape[0]):
                ilons[il] = int(np.nanargmin(np.abs(lons[il] - lon_full)))

            ds_taux = ds_taux.isel({'x':ilons})
            ds_tauy = ds_tauy.isel({'x':ilons})
        elif lon_lims != 'default':     
            ds_taux =ds_taux.sel(x=ds.x[(lon_full > lon_lims[0])&(lon_full < lon_lims[1])])
            ds_tauy =ds_tauy.sel(x=ds.x[(lon_full > lon_lims[0])&(lon_full < lon_lims[1])])
            
        tau_x = ds_taux.tau_x.values
        tau_y = ds_tauy.tau_y.values
    
        return t, lat, lon, D, ds, tau_x, tau_y
    
    else:
        return  t, lat, lon, D, ds
    