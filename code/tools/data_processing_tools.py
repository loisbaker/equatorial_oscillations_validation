import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import pi
from astropy.convolution import convolve


def smooth(field,n,axis=0,mode='same'):
    ''' convolves a 1-3D field along a given axis'''
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
        
def load_TAO(nyears_start=0, nyears_end=37.6, NEMO_year=False, ilats = 'default', ilons = 'default'):
    
    # Load in data from moorings 
    ds = np.squeeze(xr.load_dataset('../../data/TAO/dyn_xyt_dy_40year.cdf'))
    
    # make a dynamic height D with high values replaced by nans
    D = xr.where(np.abs(ds.DYN_13) > 1e10, np.nan, ds.DYN_13)
    ds['D'] = D
    
    # Remove a dodgy value (there aren't too many)
    ds.D[8309,5,5] = np.nan
    
    # Let's cut the 3 westmost zonal locations - no southern hemisphere data
    ds = ds.isel({'lon':np.arange(3,11,1)})
    ds.D[8309,5,5] = np.nan
    
    if not NEMO_year:
        ds = ds.isel({'time':range(int(nyears_start*366),int(nyears_end*366))})
    else:
        ds = ds.isel({'time':range(8198,8198+370)})
    
    if ilats != 'default':
        ds = ds.isel({'lat':ilats})
    if ilons != 'default':
        ds = ds.isel({'lon':ilons})
        
        
    lat = ds.lat.values
    lon = ds.lon.values
    
    
    
    
    t = np.arange(0,D.shape[0],1)
    
    D = ds.D.values
    return t, lat, lon, D, ds
    
    
def load_NEMO(daily_mean=True,lons='default',lats='default',lon_lims='default',winds=False):
    # Always work in lons between 0 and 360
    
    
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
    
    # Set lat at equator to zero, it's nan for some reason
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
    D = ds.D.values
    
    if winds: # Then load in the winds too
        ds_tauy = np.squeeze(xr.open_dataset('../../data/NEMO/VN206HF_4h_sometauy.nc'))
        ds_taux = np.squeeze(xr.open_dataset('../../data/NEMO/VN206HF_4h_sozotaux.nc'))
        if daily_mean:
            ds_tauy_sub = ds_tauy.isel({'time_counter':np.arange(0,ds_tauy.time_counter.shape[0],6)})
            ds_taux_sub = ds_taux.isel({'time_counter':np.arange(0,ds_taux.time_counter.shape[0],6)})
            taux = ds_taux_sub.sozotaux.copy()
            tauy = ds_tauy_sub.sometauy.copy()
            for i in np.arange(0,370,1)*6:
                ind = int(i/6)
                taux[ind,:,:] = ds_taux.sozotaux[i:i+6,:,:].mean(axis = 0)
                tauy[ind,:,:] = ds_tauy.sometauy[i:i+6,:,:].mean(axis = 0)
            ds_taux_sub['tau_x'] = taux
            ds_tauy_sub['tau_y'] = tauy
            ds_taux = ds_taux_sub
            ds_tauy = ds_tauy_sub
       

        # Deal with missing values and convert into centimetres
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
            ds =ds.sel(x=ds.x[(lon_full > lon_lims[0])&(lon_full < lon_lims[1])])

        if lats != 'default':
            ilats = np.zeros(lats.shape[0]).astype(int)
            for il in range(0,lats.shape[0]):
                ilats[il] = int(np.nanargmin(np.abs(lats[il] - lat_full))) 
            ds_taux = ds_taux.isel({'y':ilats})
            ds_tauy = ds_tauy.isel({'y':ilats})
           
      
    
        return t, lat, lon, D, ds, ds_taux, ds_tauy
    
    else:
        return  t, lat, lon, D, ds
    