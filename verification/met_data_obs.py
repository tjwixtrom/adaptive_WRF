# Script for opening data in xarray for MET tools
# from netCDF4 import Dataset, chartostring
import sys
import numpy as np
# import datetime as dt
import pandas as pd
import xarray as xr

date = sys.argv[1]
lead = sys.argv[2]
domain = sys.argv[3]

finit = pd.to_datetime(date, format='%Y%m%d%H')
valid = finit + pd.Timedelta(hours=int(lead))
init = valid - pd.Timedelta(hours=3)

var_name = 'precipitation'
long_name = '3-hour Accumulated Precipitation'
# valid = dt.datetime(2016, 1, 2, 0)
# valid = pd.Timestamp(2016, 5, 1, 21)
# f = Dataset('/lustre/scratch/twixtrom/ST4_201601_03h.nc')
if domain == "1":
   accum = "03"
else:
    accum = "01"
f = xr.open_dataset('/lustre/scratch/twixtrom/ST4_'+init.strftime('%Y%m')+'_'+accum+'h.nc')
precip = f.precipitation.sel(time=valid)
data = np.array(precip.data, dtype=np.float64).round(decimals=7)
met_data = data[::-1, :].copy()

# init = valid - pd.Timedelta(hours=3)
# valid_ref = dt.datetime.strptime(getattr(f.variables["time"], "units"), "seconds since %Y-%m-%d %H:%M:%S")
# lead, rem = divmod((valid-init).total_seconds(), 3600)
# leadtime = valid - init
# lead = str(leadtime.components.days*24 + leadtime.components.hours)


Nx = len(f.coords["x"])
Ny = len(f.coords["y"])

hemi = 'N'

# if getattr(f, "POLE_LAT") > 0:
#    hemi = 'N'
# else:
#    hemi = 'S'


if domain == '1':
    attrs = {
       'valid': valid.strftime("%Y%m%d_%H%M%S"),
       'init':   init.strftime("%Y%m%d_%H%M%S"),
       'lead':   lead,
       'accum':  accum,

       'name':      var_name,
       'long_name': long_name,
       'level':     "SURFACE",
       'units':     str(precip.units),

       'grid': {
         'type' : "Lambert Conformal",
         'name' : "WRF Domain",
         'hemisphere' : hemi,
         'scale_lat_1' : 30.0,
         'scale_lat_2' : 60.0,
         'lat_pin' : 38.0,
         'lon_pin' : -103.0,
         'x_pin' : float(Nx/2),
         'y_pin' : float(Ny/2),
         'lon_orient' : -101.0,
         'd_km' : 12.0,
         'r_km' : 6371.2,
         'nx' : Nx,
         'ny' : Ny,
       }
    }
else:
    attrs = {
       'valid': valid.strftime("%Y%m%d_%H%M%S"),
       'init':   init.strftime("%Y%m%d_%H%M%S"),
       'lead':   lead,
       'accum':  accum,

       'name':      var_name,
       'long_name': long_name,
       'level':     "SURFACE",
       'units':     str(precip.units),

       'grid': {
         'type' : "Lambert Conformal",
         'name' : "WRF Domain",
         'hemisphere' : hemi,
         'scale_lat_1' : 30.0,
         'scale_lat_2' : 60.0,
         'lat_pin' : 33.64635,
         'lon_pin' : -103.0017,
         'x_pin' : float(Nx/2),
         'y_pin' : float(Ny/2),
         'lon_orient' : -101.0,
         'd_km' : 4.0,
         'r_km' : 6371.2,
         'nx' : Nx,
         'ny' : Ny,
       }
    }

# print("Attributes: " + repr(attrs))
