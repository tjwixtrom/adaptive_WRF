#!/home/twixtrom/miniconda3/envs/analogue/bin/python
##############################################################################################
# run_wrf.py - Code for calculating the best member over a date range
#
#
# by Tyler Wixtrom
# Texas Tech University
# 22 January 2019
#
##############################################################################################
# import sys
# import subprocess
# # import shutil
# import os
# import glob
import operator
import warnings

from analogue_algorithm import (check_logs, concat_files, create_wrf_namelist,
                                find_max_coverage, find_analogue,
                                find_analogue_precip_area, increment_time)

import numpy as np
from netCDF4 import num2date
import pandas as pd
import xarray as xr
# from pathlib import Path
# import numpy as np

warnings.filterwarnings("ignore")

# ndays = sys.argv[1]
ndays = 0
# Define initial period start date
start_date = pd.Timestamp(2016, 5, 2, 12)
chunks = None
mem_list = ['mem'+str(i) for i in range(1, 21)]
mp_list = ['mem'+str(i) for i in range(1, 11)]
pbl_list = ['mem1', *['mem'+str(i) for i in range(11, 21)]]

analogue_param = {
    'sigma': 1.,
    'pcp_threshold': 10.,
    'sum_threshold': 50.,
    'pcp_operator': operator.ge,
    'logfile': '/home/twixtrom/adaptive_WRF/adaptive_WRF/an_selection_log_201605.log',
    'cape_threshold': 1000.,
    'cape_operator': operator.ge,
    'height_500hPa_threshold': 5700.,
    'height_500hPa_operator': operator.le,
    'start_date': '2015-01-01T12:00:00',
    'dataset_dir': '/lustre/scratch/twixtrom/dataset_variables/',
    'mp_an_method': 'Different Points - pcpT00+hgt500f00+capeT-3',
    'pbl_an_method': 'Same Points - pcpT00+hgt500T00',
    'dt': '1D'}
# Define model configuration parameters
wrf_param = {
    'dir_control': '/lustre/scratch/twixtrom/adaptive_wrf_post/control_thompson',
    'dir_dataset': '/lustre/scratch/twixtrom/dataset_variables',
    'rootdir': '/home/twixtrom/adaptive_WRF/',
    'scriptsdir': '/home/twixtrom/adaptive_WRF/adaptive_WRF/',
    'dir_run': '/lustre/scratch/twixtrom/adaptive_wrf_run/adaptive_run/',
    'dir_compressed_gfs': '/lustre/scratch/twixtrom/gfs_compress_201605/',

    #  Domain-Specific Parameters
    'norm_cores': 36,
    'model_Nx1': 508,           # number of grid points in x-direction
    'model_Ny1': 328,           # number of grid points in y-direction
    'model_Nz': 38,             # number of grid points in vertical
    'model_ptop': 5000,         # Pressure at model top
    'model_gridspx1': 12000,    # gridspacing in x (in meters)
    'model_gridspy1': 12000,    # gridspacing in y
    'dt': 36,                   # model time step (in sec)
    'model_centlat': 38.0,          # center latitude of domain
    'model_centlon': -103.0,        # center longitude of domain
    'model_stdlat1': 30.0,          # first true latitude of domain
    'model_stdlat2': 60.0,          # second true latitude of domain
    'model_stdlon': -101.0,          # standard longitude
    'dlbc': 360,                # number of minutes in between global model BCs
    'output_interval': 180,         # Frequency of model output to file mother domain
    'output_intervalNEST': 60,      # Frequency of model output to file - Nest
    'model_num_in_output': 10000,   # Output times per file
    'fct_len': 2880,                # Minutes to forecast for
    'feedback': 0,                  # 1-way(0) or 2-way nest(1)
    'enum': 0,                      # Number of physics runs
    'siz1': 29538340980,            # File size dom 1
    'siz2': 15445197060,            # File size dom 2

    # Nested domains info
    'model_gridspx1_nest': 4000,
    'model_gridspy1_nest': 4000,
    'iparent_st_nest': 200,
    'jparent_st_nest': 80,
    'model_Nx1_nest': 322,
    'model_Ny1_nest': 271,
    'parent_id_nest': 1,
    'grid_ratio_nest': 3,

    #  Locations of important directories

    'dir_wps': '/lustre/work/twixtrom/WPSV3.5.1/',
    'dir_wrf': '/lustre/work/twixtrom/WRFV3.5.1/run/',
    'dir_sub': '/home/twixtrom/adaptive_WRF/adaptive_WRF/',
    'dir_store': '/lustre/scratch/twixtrom/adaptive_wrf_save/control_thompson/',
    'dir_scratch': '/lustre/scratch/twixtrom/',
    'dir_gfs': '/lustre/scratch/twixtrom/gfs_data/',

    # Parameters for the model (not changed very often)
    'model_mp_phys': 8,          # microphysics scheme
    'model_spec_zone': 1,    # number of grid points with tendencies
    'model_relax_zone': 4,   # number of blended grid points
    'dodfi': 0,                  # Do Dfi 3-yes 0-no
    'model_lw_phys': 1,          # model long wave scheme
    'model_sw_phys': 1,          # model short wave scheme
    'model_radt': 30,            # radiation time step (in minutes)
    'model_sfclay_phys': 1,      # surface layer physics
    'model_surf_phys': 2,        # land surface model
    'model_pbl_phys': 1,         # pbl physics
    'model_bldt': 0,             # boundary layer time steps (0 : each time steps, in min)
    'model_cu_phys': 6,          # cumulus param
    'model_cu_phys_nest': 0,     # cumulus param 3km
    'model_cudt': 5,             # cumuls time step
    'model_use_surf_flux': 1,    # 1 is yes
    'model_use_snow': 0,
    'model_use_cloud': 1,
    'model_soil_layers': 4,
    'model_w_damping': 1,
    'model_diff_opt': 1,
    'model_km_opt': 4,
    'model_dampcoef': 0.2,
    'model_tbase': 300.,
    'model_nwp_diagnostics': 1,
    'model_do_radar_ref': 1,
    'dampopt': 3,
    'zdamp': 5000.}

# Calculated terms

wrf_param['fct_len_hrs'] = wrf_param['fct_len'] / 60.
wrf_param['dlbc_hrs'] = wrf_param['dlbc'] / 60.
wrf_param['assim_bzw'] = wrf_param['model_spec_zone'] + wrf_param['model_relax_zone']
wrf_param['otime'] = wrf_param['output_interval'] / 60.
wrf_param['otime_nest'] = wrf_param['output_intervalNEST'] / 60.
wrf_param['model_BC_interval'] = wrf_param['dlbc'] * 60.


# Find date and time of model start and end
model_initial_date = increment_time(start_date, days=int(ndays))
model_end_date = increment_time(model_initial_date, hours=wrf_param['fct_len_hrs'])
datep = increment_time(model_initial_date, hours=-1)
print('Starting forecast for: ' + str(model_initial_date), flush=True)

# Determine number of input metgrid levels
# GFS changed from 27 to 32 on May 15, 2016
if model_initial_date < pd.to_datetime('2016-05-12T12:00:00'):
    wrf_param['num_metgrid_levels'] = 27
else:
    wrf_param['num_metgrid_levels'] = 32

# Analogue selection and parameterization optimization section
# Open the previous forecast from the thompson control and find the nearest analogue

# Open output log file
logfile = open(analogue_param['logfile'], 'a+')

# Define times that can be selected as possible analogue matching times,
# must be between forecast hours 6 and 24.
an_times_d02 = pd.date_range(start=increment_time(model_initial_date, hours=6),
                             end=increment_time(model_initial_date, hours=24),
                             freq='H')

previous_forecast_time = increment_time(model_initial_date, hours=-24)
previous_forecast_file_d02 = (wrf_param['dir_control'] + '/' +
                              previous_forecast_time.strftime('%Y%m%d%H') + '/wrfprst_d02_' +
                              previous_forecast_time.strftime('%Y%m%d%H') + '.nc')
try:
    forecast_d02_data = xr.open_dataset(previous_forecast_file_d02)
except FileNotFoundError:
    logfile.write(model_initial_date.strftime('%Y%m%d%H')+', None, None\n')
    raise FileNotFoundError('File not found: '+previous_forecast_file_d02)
forecast_pcp_d02 = forecast_d02_data['timestep_pcp'].sel(time=an_times_d02)
forecast_pcp_d02.attrs['sigma'] = analogue_param['sigma']
forecast_pcp_d02.attrs['threshold'] = analogue_param['pcp_threshold']
forecast_pcp_d02.attrs['operator'] = operator.ge
sum_max_d02, max_time_d02 = find_max_coverage(forecast_pcp_d02, dim=['y', 'x'])

# check if the max leadtime meets minimum criteria
if sum_max_d02 >= analogue_param['sum_threshold']:
    domain = 'd02'
    leadtime = max_time_d02 - model_initial_date
else:
    domain = 'd01'
    an_times_d01 = pd.date_range(start=increment_time(model_initial_date, hours=6),
                                 end=increment_time(model_initial_date, hours=24), freq='3H')
    previous_forecast_file_d01 = (wrf_param['dir_control'] + '/' +
                                  previous_forecast_time.strftime('%Y%m%d%H') +
                                  '/wrfprst_d01_' +
                                  previous_forecast_time.strftime('%Y%m%d%H') + '.nc')
    try:
        forecast_d01_data = xr.open_dataset(previous_forecast_file_d01)
    except FileNotFoundError:
        logfile.write(model_initial_date.strftime('%Y%m%d%H') + ', None, None\n')
        raise FileNotFoundError('File not found: ' + previous_forecast_file_d01)
    forecast_pcp_d01 = forecast_d01_data['timestep_pcp'].sel(time=an_times_d01)
    forecast_pcp_d01.attrs['sigma'] = analogue_param['sigma']
    forecast_pcp_d01.attrs['threshold'] = analogue_param['pcp_threshold']
    forecast_pcp_d01.attrs['operator'] = operator.ge
    sum_max_d01, max_time_d01 = find_max_coverage(forecast_pcp_d01, dim=['y', 'x'])
    if sum_max_d01 >= analogue_param['sum_threshold']:
        leadtime = max_time_d01 - model_initial_date
    else:
        logfile.write(model_initial_date.strftime('%Y%m%d%H')+', None, None\n')
        raise ValueError('Precipitation exceeding threshold not forecast')

an_time = model_initial_date + leadtime
print('Domain ', domain, 'and leadtime of ', leadtime.components.hours, ' hours selected for ',
      'analogue comparison at ', an_time)

leadtime_str = str(leadtime.components.hours)
logfile.write(model_initial_date.strftime('%Y%m%d%H')+', '+domain+', '+leadtime_str+'\n')

# subset previous forecast to analogue time and add analogue attributes
an_pcp = forecast_d02_data['timestep_pcp'].sel(time=an_time)
an_pcp.attrs['threshold'] = analogue_param['pcp_threshold']
an_pcp.attrs['sigma'] = analogue_param['sigma']
an_pcp.attrs['operator'] = analogue_param['pcp_operator']


def open_pcp(hour, domain, param):
    if domain == 'd01':
        dx = '12km'
    else:
        dx = '4km'
    pcpfile = param['dataset_dir']+'adp_dataset_'+dx+'_timestep_pcp_f'+hour+'.nc'
    precip = xr.open_dataset(pcpfile, chunks=chunks)
    precip.attrs['threshold'] = param['pcp_threshold']
    precip.attrs['sigma'] = param['sigma']
    precip.attrs['operator'] = param['pcp_operator']
    fcst_mean = xr.concat([precip[mem] for mem in mem_list],
                          dim='Member').mean(dim='Member')
    precip['mean'] = fcst_mean
    return precip


def open_cape(hour, domain, param):
    if domain == 'd01':
        dx = '12km'
    else:
        dx = '4km'
    file = param['dataset_dir']+'adp_dataset_'+dx+'_cape_f'+hour+'.nc'
    cape = xr.open_dataset(file, chunks=chunks)
    cape.attrs['threshold'] = param['cape_threshold']
    cape.attrs['sigma'] = param['sigma']
    cape.attrs['operator'] = param['cape_operator']
    cape_mean = xr.concat([cape[mem] for mem in mem_list],
                      dim='Member').mean(dim='Member')
    cape['mean'] = cape_mean
    return cape


def open_height(hour, domain, param):
    if domain == 'd01':
        dx = '12km'
    else:
        dx = '4km'
    file = param['dataset_dir']+'adp_dataset_'+dx+'_height_500hPa_f'+str(int(hour))+'.nc'
    height_500hPa = xr.open_dataset(file, chunks=chunks)
    height_500hPa.attrs['threshold'] = param['height_500hPa_threshold']
    height_500hPa.attrs['sigma'] = param['sigma']
    height_500hPa.attrs['operator'] = param['height_500hPa_operator']
    height_500hPa_mean = xr.concat([height_500hPa[mem] for mem in mem_list],
                                   dim='Member').mean(dim='Member')
    height_500hPa['mean'] = height_500hPa_mean
    return height_500hPa


# Open the precip, height, and cape dataset files
pcp_dataset = open_pcp(leadtime_str, domain, analogue_param)
cape_dataset = open_cape(leadtime_str, domain, analogue_param)
height_dataset = open_height(leadtime_str, domain, analogue_param)
print(height_dataset)

# Open height and cape for forecast dataset

# Open the Stage4 precip observations file
if domain == 'd01':
    obsfile = '/lustre/work/twixtrom/ST4_2015_03h.nc'
else:
    obsfile = '/lustre/work/twixtrom/ST4_2015_01h.nc'

stage4 = xr.open_dataset(obsfile, chunks=chunks, decode_cf=False)
vtimes_stage4 = num2date(stage4.valid_times, stage4.valid_times.units)
stage4.coords['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4.coords['lat'] = stage4.lat
stage4.coords['lon'] = stage4.lon

mp_an_idx = find_analogue()


# Remove any existing namelist
# try:
#     os.remove(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'namelist.input')
# except FileNotFoundError:
#     pass
#
# # Generate namelist
# namelist = wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/namelist.input'
# print('Creating namelist.input as: '+namelist, flush=True)
# create_wrf_namelist(namelist, wrf_param, model_initial_date)
#
# # Remove any existing wrfout files
# for file in glob.glob(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/wrfout*'):
#     os.remove(file)
#
# # Call mpi for real.exe
# print('Running real.exe', flush=True)
# run_real_command = ('cd '+wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H') +
#                     ' && mpirun -np '+str(wrf_param['norm_cores'])+' '+wrf_param['dir_run']+
#                     model_initial_date.strftime('%Y%m%d%H')+'/real.exe')
# real = subprocess.call(run_real_command, shell=True)
#
# # Combine log files into single log
# concat_files((wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'),
#              (wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_real_' +
#              model_initial_date.strftime('%Y%m%d%H')+'.log'))
#
# # Remove the logs
# for file in glob.glob(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'):
#     os.remove(file)
#
# # Check for successful completion
# check_logs(wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_real_' +
#            model_initial_date.strftime('%Y%m%d%H')+'.log',
#            wrf_param['dir_sub']+wrf_param['check_log'], model_initial_date)
#
# # Call mpi for wrf.exe
# print('Running wrf.exe', flush=True)
# run_wrf_command = ('cd '+wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H') +
#                    ' && mpirun -np '+str(wrf_param['norm_cores'])+' '+wrf_param['dir_run']+
#                    model_initial_date.strftime('%Y%m%d%H')+'/wrf.exe')
# wrf = subprocess.call(run_wrf_command, shell=True)
# # wrf.wait()
#
# # Combine log files into single log
# print('Moving log files', flush=True)
# concat_files((wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'),
#              (wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_wrf_' +
#              model_initial_date.strftime('%Y%m%d%H')+'.log'))
#
# # Remove the logs
# for file in glob.glob(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'):
#     os.remove(file)
#
# # Check for successful completion
# check_logs(wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_wrf_' +
#            model_initial_date.strftime('%Y%m%d%H')+'.log',
#            wrf_param['dir_sub']+wrf_param['check_log'], model_initial_date, wrf=True)
#
# # Move wrfout files to storage
# print('Moving output', flush=True)
# move_wrf_files_command = ('mv '+wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+
#                           '/wrfout_d01* '+wrf_param['dir_store']+
#                           model_initial_date.strftime('%Y%m%d%H')+'/wrfout_d01_' +
#                           model_initial_date.strftime('%Y%m%d%H')+'.nc && '
#                           'mv ' + wrf_param['dir_run'] +
#                           model_initial_date.strftime('%Y%m%d%H') +
#                           '/wrfout_d02* ' + wrf_param['dir_store'] +
#                           model_initial_date.strftime('%Y%m%d%H') + '/wrfout_d02_' +
#                           model_initial_date.strftime('%Y%m%d%H') + '.nc')
# subprocess.run(move_wrf_files_command, shell=True)
# print('Finished with forecast initialized: '+model_initial_date)
