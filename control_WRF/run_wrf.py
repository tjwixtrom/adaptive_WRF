#!/home/twixtrom/miniconda3/envs/analogue/bin/python
##############################################################################################
# run_wrf.py - Code for running WRF over a date range
#
#
# by Tyler Wixtrom
# Texas Tech University
# 22 January 2019
#
##############################################################################################

import sys
import subprocess
# import shutil
import os
import glob
import warnings
from datetime import datetime
from pathlib import Path
from analogue_algorithm import create_wrf_namelist, check_logs, increment_time, concat_files

# import numpy as np

warnings.filterwarnings("ignore")

ndays = sys.argv[1]
member = sys.argv[2]

if member == 'thompson':
    pbl_opt = 1
    mp_opt = 8
    sfc_opt = 1
elif member == 'ETA':
    pbl_opt = 2
    mp_opt = 5
    sfc_opt = 2
else:
    raise ValueError('Model Physics member not defined')

# Define initial period start date
start_date = datetime(2016, 7, 1, 12)

# Define model configuration parameters
wrf_param = {
    'rootdir': '/home/twixtrom/adaptive_WRF/',
    'scriptsdir': '/home/twixtrom/adaptive_WRF/control_WRF/',
    'dir_run': '/lustre/scratch/twixtrom/adaptive_wrf_run/control_'+member+'_run/',
    'dir_compressed_gfs': '/lustre/scratch/twixtrom/gfs_compress_201605/',
    'check_log': 'check_log_'+member+'_201501.log',

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
    'dir_sub': '/home/twixtrom/adaptive_WRF/control_WRF/',
    'dir_store': '/lustre/scratch/twixtrom/adaptive_wrf_save/control_'+member+'/',
    'dir_scratch': '/lustre/scratch/twixtrom/',
    'dir_gfs': '/lustre/scratch/twixtrom/gfs_data/',

    # Parameters for the model (not changed very often)
    'model_mp_phys': mp_opt,          # microphysics scheme
    'model_spec_zone': 1,    # number of grid points with tendencies
    'model_relax_zone': 4,   # number of blended grid points
    'dodfi': 0,                  # Do Dfi 3-yes 0-no
    'model_lw_phys': 1,          # model long wave scheme
    'model_sw_phys': 1,          # model short wave scheme
    'model_radt': 30,            # radiation time step (in minutes)
    'model_sfclay_phys': sfc_opt,      # surface layer physics
    'model_surf_phys': 2,        # land surface model
    'model_pbl_phys': pbl_opt,         # pbl physics
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
    'zdamp': 5000.
}

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
print('Starting forecast for: '+str(model_initial_date), flush=True)

save_dir = wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Determine number of input metgrid levels
# GFS changed from 27 to 32 on May 15, 2016
if model_initial_date < datetime(2016, 5, 11, 12):
    wrf_param['num_metgrid_levels'] = 27
else:
    wrf_param['num_metgrid_levels'] = 32

# Remove any existing namelist
try:
    os.remove(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'namelist.input')
except FileNotFoundError:
    pass

# Generate namelist
namelist = wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/namelist.input'
print('Creating namelist.input as: '+namelist, flush=True)
create_wrf_namelist(namelist, wrf_param, model_initial_date)

# Remove any existing wrfout files
for file in glob.glob(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/wrfout*'):
    os.remove(file)

# Call mpi for real.exe
print('Running real.exe', flush=True)
run_real_command = ('cd '+wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H') +
                    ' && mpirun -np '+str(wrf_param['norm_cores'])+' '+wrf_param['dir_run']+
                    model_initial_date.strftime('%Y%m%d%H')+'/real.exe')
real = subprocess.call(run_real_command, shell=True)

# Combine log files into single log
concat_files((wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'),
             (wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_real_' +
              model_initial_date.strftime('%Y%m%d%H')+'.log'))

# Remove the logs
for file in glob.glob(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'):
    os.remove(file)

# Check for successful completion
check_logs(wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_real_' +
           model_initial_date.strftime('%Y%m%d%H')+'.log',
           wrf_param['dir_sub']+wrf_param['check_log'], model_initial_date)

# Call mpi for wrf.exe
print('Running wrf.exe', flush=True)
run_wrf_command = ('cd '+wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H') +
                   ' && mpirun -np '+str(wrf_param['norm_cores'])+' '+wrf_param['dir_run']+
                   model_initial_date.strftime('%Y%m%d%H')+'/wrf.exe')
wrf = subprocess.call(run_wrf_command, shell=True)
# wrf.wait()

# Combine log files into single log
print('Moving log files', flush=True)
concat_files((wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'),
             (wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_wrf_' +
              model_initial_date.strftime('%Y%m%d%H')+'.log'))

# Remove the logs
for file in glob.glob(wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+'/rsl.*'):
    os.remove(file)

# Check for successful completion
check_logs(wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')+'/rslout_wrf_' +
           model_initial_date.strftime('%Y%m%d%H')+'.log',
           wrf_param['dir_sub']+wrf_param['check_log'], model_initial_date, wrf=True)

# Move wrfout files to storage
print('Moving output', flush=True)
move_wrf_files_command = ('mv '+wrf_param['dir_run']+model_initial_date.strftime('%Y%m%d%H')+
                          '/wrfout_d01* '+wrf_param['dir_store']+
                          model_initial_date.strftime('%Y%m%d%H')+'/wrfout_d01_' +
                          model_initial_date.strftime('%Y%m%d%H')+'.nc && '
                          'mv ' + wrf_param['dir_run'] +
                          model_initial_date.strftime('%Y%m%d%H') +
                          '/wrfout_d02* ' + wrf_param['dir_store'] +
                          model_initial_date.strftime('%Y%m%d%H') + '/wrfout_d02_' +
                          model_initial_date.strftime('%Y%m%d%H') + '.nc')
subprocess.run(move_wrf_files_command, shell=True)
print('Finished with forecast initialized: '+str(model_initial_date))
