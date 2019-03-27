#!/home/twixtrom/miniconda3/envs/wrfpost/bin/python
import sys
from PWPP import wrfpost
import numpy as np
from metpy.units import units
# import xarray as xr

infile = sys.argv[1]
outfile = sys.argv[2]
variables = ['temp',
             'dewpt',
             'uwnd',
             'vwnd',
             'wwnd',
             'avor',
             'height',
             'temp_2m',
             'dewpt_2m',
             'mslp',
             'q_2m',
             'u_10m',
             'v_10m',
             'timestep_pcp',
             'UH',
             'cape',
             'refl'
             ]

plevs = np.array([1000, 925, 850, 700, 500, 300, 250]) * units('hPa')
# chunks = {'Time': 1}
wrfpost(infile, outfile, variables, plevs=plevs, compression=True, complevel=4,
        format='NETCDF4')
