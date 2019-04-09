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
member = sys.argv[4]

init = pd.to_datetime(date, format='%Y%m%d%H')
valid = init + pd.Timedelta(hours=int(lead))
init_str = str(date)
var_name = 'timestep_pcp'
long_name = 'Timestep Accumulated Precipitation'
# valid = dt.datetime(2016, 5, 1, 21)

if domain == "1":
   accum = "03"
else:
    accum = "01"

# f = Dataset('/lustre/scratch/twixtrom/adaptive_wrf_post/control_thompson/2016050112/wrfprst_d01_2016050112.nc')
f = xr.open_dataset('/lustre/scratch/twixtrom/adaptive_wrf_post/'+member+'/'+init_str+'/wrfprst_d0'+domain+'_'+init_str+'.nc')
# data = np.float64(f.variables[var_name][3, :])
precip = f.timestep_pcp.sel(time=valid)
data = np.array(precip.data, dtype=np.float64).round(decimals=7)
met_data = data[::-1, :].copy()

# init = dt.datetime.strptime(getattr(f, "START_DATE"), "%Y-%m-%d_%H:%M:%S")
# valid_ref = dt.datetime.strptime(f.time.units, "seconds since %Y-%m-%d %H:%M:%S")
# init = pd.Timestamp(f.time.isel(time=0).data)
# leadtime = valid - init
# lead = str(leadtime)

Nx = len(f.coords["x"])
Ny = len(f.coords["y"])

# if getattr(f, "POLE_LAT") > 0:
#    hemi = 'N'
# else:
#    hemi = 'S'
hemi = 'N'
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
     'scale_lat_1' : float(f.attrs["TRUELAT1"]),
     'scale_lat_2' : float(f.attrs["TRUELAT2"]),
     'lat_pin' : float(f.attrs["CEN_LAT"]),
     'lon_pin' : float(f.attrs["CEN_LON"]),
     'x_pin' : float(Nx/2),
     'y_pin' : float(Ny/2),
     'lon_orient' : float(f.attrs["STAND_LON"]),
     'd_km' : f.attrs["DX"]/1000.0,
     'r_km' : 6371.2,
     'nx' : Nx,
     'ny' : Ny,
   }
}

# print("Attributes: " + repr(attrs))
