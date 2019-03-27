#!/home/twixtrom/miniconda3/envs/analogue/bin/python
#$ -V
#$ -cwd
#$ -S /home/twixtrom/miniconda3/envs/analogue/bin/python
#$ -N calc_mean_best
#$ -o sub_mean_best
#$ -e error_mean_best
#$ -q ancellcc
#$ -pe sm 32
#$ -P communitycluster

##############################################################################################
# calc_mean_best.py - Code for calculating the best member over a date range
#
#
# by Tyler Wixtrom
# Texas Tech University
# 22 January 2019
#
##############################################################################################

import numpy as np
import xarray as xr
from netCDF4 import num2date

from analogue_algorithm.calc import verify_members

verif_param = {
    'forecast_hour': 12.,
    'threshold': 10.,
    'sigma': 1,
    'start_date': '2015-07-01T12:00:00',
    'end_date': '2015-08-01T12:00:00',
    'dt': '1D',
    'fname': '/home/twixtrom/adaptive_WRF/best_members_july.txt',
    'obsfile': '/lustre/work/twixtrom/ST4_2015_03h.nc'
    }
chunks = None

mem_list = ['mem'+str(i) for i in range(1, 21)]
mp_list = ['mem'+str(i) for i in range(1, 11)]
pbl_list = ['mem1', *['mem'+str(i) for i in range(11, 21)]]

# Save parameters to ouput metadata file.
f = open(verif_param['fname'], 'w')
f.write('Verification Parameters\n')
for key in verif_param.keys():
    f.write(key+': '+str(verif_param[key])+'\n')


def open_pcp(hour, dx):
    pcpfile = '/lustre/scratch/twixtrom/dataset_variables/temp/' \
              'adp_dataset_'+dx+'_timestep_pcp_f'+str(int(hour))+'.nc'
    precip = xr.open_dataset(pcpfile, chunks=chunks)
    return precip  #.where(precip.time < np.datetime64('2016-01-01T12:00:00'), drop=True)


pcp = open_pcp(verif_param['forecast_hour'], '12km')


stage4 = xr.open_dataset(verif_param['obsfile'], chunks=chunks, decode_cf=False)
vtimes_stage4 = num2date(stage4.valid_times, stage4.valid_times.units)
stage4.coords['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4.coords['lat'] = stage4.lat
stage4.coords['lon'] = stage4.lon


tot_rmse = verify_members(pcp, stage4.total_precipitation, verif_param, mem_list)
mean_mp = np.array([tot_rmse[mem] for mem in mp_list])
mean_best_mp = mp_list[np.nanargmin(mean_mp)]
mean_pbl = np.array([tot_rmse[mem] for mem in pbl_list])
mean_best_pbl = pbl_list[np.nanargmin(mean_pbl)]

f.write('Mean Best MP Member: '+mean_best_mp+'\n')
f.write('Mean Best PBL Member: '+mean_best_pbl+'\n')
f.close()
