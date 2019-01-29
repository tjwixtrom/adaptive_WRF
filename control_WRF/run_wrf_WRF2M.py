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

import sys
import subprocess
# import shutil
import os
import glob
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# import numpy as np

warnings.filterwarnings("ignore")

ndays = sys.argv[1]

# Define initial period start date
start_date = datetime(2016, 5, 1, 12)

# Define model configuration parameters
wrf_param = {
    'rootdir': '/home/twixtrom/adaptive_WRF/',
    'scriptsdir': '/home/twixtrom/adaptive_WRF/control_WRF/',
    'dir_run': '/lustre/scratch/twixtrom/adaptive_wrf_run/control_WRF2M_run/',
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
    'dir_store': '/lustre/scratch/twixtrom/adaptive_wrf_save/control_WRF2M/',
    'dir_scratch': '/lustre/scratch/twixtrom/',
    'dir_gfs': '/lustre/scratch/twixtrom/gfs_data/',

    # Parameters for the model (not changed very often)
    'model_mp_phys': 16,          # microphysics scheme
    'model_spec_zone': 1,    # number of grid points with tendencies
    'model_relax_zone': 4,   # number of blended grid points
    'dodfi': 0,                  # Do Dfi 3-yes 0-no
    'model_lw_phys': 1,          # model long wave scheme
    'model_sw_phys': 1,          # model short wave scheme
    'model_radt': 30,            # radiation time step (in minutes)
    'model_sfclay_phys': 1,      # surface layer physics
    'model_surf_phys': 2,        # land surface model
    'model_pbl_phys': 99,         # pbl physics
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


def increment_time(date1, days=0, hours=0):
    """
    Increment time from start by a specified number of days or hours

    Parameters:
        date1: datetime.datetime
        days: int, number of days to advance
        hours: int, number of hours to advance
    Returns: datetime.datetime, incremented time and date
    """
    return date1 + timedelta(days=days, hours=hours)


def concat_files(inname, outname):
    """
    Concatenate text files into a single output
    :param inname: directory path and name wildcard to input files
    :param outname: directory path and name of output file
    :return: None
    """
    with open(outname, 'w') as outfile:
        for fname in glob.glob(inname):
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    outfile.close()


def create_wrf_namelist(fname, parameters):
    """
    Create a WRF control namelist

    :param fname: str, output file name string
    :param parameters: dict, WRF configuration parameters
    :return: Saves WRF control namelist as fname
    """
    f = open(fname, 'w')
    f.write("""
&time_control
run_days                            = 0,
run_hours                           = 0,
run_minutes                         = {},
run_seconds                         = 0,\n""".format(parameters['fct_len']))
    # Model start time
    f.write('start_year                          = {0}, {0},\n'.format(
        model_initial_date.strftime('%Y')))
    f.write('start_month                         = {0}, {0},\n'.format(
        model_initial_date.strftime('%m')))
    f.write('start_day                           = {0}, {0},\n'.format(
        model_initial_date.strftime('%d')))
    f.write('start_hour                          = {0}, {0},\n'.format(
        model_initial_date.strftime('%H')))
    f.write('start_minute                        = {0}, {0},\n'.format(
        model_initial_date.strftime('%M')))
    f.write('start_second                        = {0}, {0},\n'.format(
        model_initial_date.strftime('%S')))

    # Model end time
    f.write('end_year                            = {0}, {0},\n'.format(
        model_end_date.strftime('%Y')))
    f.write('end_month                           = {0}, {0},\n'.format(
        model_end_date.strftime('%m')))
    f.write('end_day                             = {0}, {0},\n'.format(
        model_end_date.strftime('%d')))
    f.write('end_hour                            = {0}, {0},\n'.format(
        model_end_date.strftime('%H')))
    f.write('end_minute                          = {0}, {0},\n'.format(
        model_end_date.strftime('%M')))
    f.write('end_second                          = {0}, {0},\n'.format(
        model_end_date.strftime('%S')))

    f.write("""interval_seconds                    = {0}
input_from_file                     = .true.,.true.,
history_interval                    = {1}, {2},
""".format(int(parameters['model_BC_interval']), parameters['output_interval'],
               parameters['output_intervalNEST']))
    f.write("""frames_per_outfile                  = {0}, {0},
restart                             = .false.,
restart_interval                    = 10000,
io_form_history                     = 2
io_form_restart                     = 2
io_form_input                       = 2
io_form_boundary                    = 2
debug_level                         = 0
nwp_diagnostics                     = {1}
/

&domains
time_step                           = {2},
time_step_fract_num                 = 0,
time_step_fract_den                 = 1,
max_dom                             = 2,
""".format(parameters['model_num_in_output'], parameters['model_nwp_diagnostics'],
           parameters['dt']))
    f.write("""e_we                                = {0}, {1},
e_sn                                = {2}, {3},
e_vert                              = {4},  {4},
eta_levels			    = 1.000, 0.995, 0.990, 0.985,
                                       0.980, 0.970, 0.960, 0.950,
                                       0.940, 0.930, 0.920, 0.910,
                                       0.900, 0.880, 0.860, 0.830,
                                       0.800, 0.770, 0.740, 0.710,
                                       0.680, 0.640, 0.600, 0.560,
                                       0.520, 0.480, 0.440, 0.400,
                                       0.360, 0.320, 0.280, 0.240,
                                       0.200, 0.160, 0.120, 0.080,
                                       0.040, 0.000
""".format(parameters['model_Nx1'], parameters['model_Nx1_nest'],
           parameters['model_Ny1'], parameters['model_Ny1_nest'],
           parameters['model_Nz']))
    f.write("""p_top_requested                     = {0},
num_metgrid_levels                  = {5}
num_metgrid_soil_levels             = 4,
dx                                  = {1}, {2}
dy                                  = {3}, {4}
grid_id                             = 1,     2,
parent_id                           = 0,    1,
""".format(parameters['model_ptop'], parameters['model_gridspx1'],
           parameters['model_gridspx1_nest'], parameters['model_gridspy1'],
           parameters['model_gridspy1_nest'], parameters['num_metgrid_levels']))
    f.write("""i_parent_start                      = 1, {0}
j_parent_start                      = 1, {1}
parent_grid_ratio                   = 1, {2}
parent_time_step_ratio              = 1, {2}
feedback                            = {3},
smooth_option                       = 0
/

""".format(parameters['iparent_st_nest'], parameters['jparent_st_nest'],
           parameters['grid_ratio_nest'], parameters['feedback']))
    f.write("""&dfi_control
dfi_opt                             = {},
dfi_nfilter                         = 7,
dfi_write_filtered_input            = .false.,
dfi_write_dfi_history               = .false.,
dfi_cutoff_seconds                  = 3600,
dfi_time_dim                        = 1000,
""".format(parameters['dodfi']))
    f.write("""dfi_bckstop_year                    = {0},
dfi_bckstop_month                   = {1},
dfi_bckstop_day                     = {2},
dfi_bckstop_hour                    = {3},
dfi_bckstop_minute                  = {4},
dfi_bckstop_second                  = {5},
""".format(datep.strftime('%Y'), datep.strftime('%m'), datep.strftime('%d'),
           datep.strftime('%H'), datep.strftime('%M'), datep.strftime('%S')))
    f.write("""dfi_fwdstop_year                    = {0},
dfi_fwdstop_month                   = {1},
dfi_fwdstop_day                     = {2},
dfi_fwdstop_hour                    = {3},
dfi_fwdstop_minute                  = {4},
dfi_fwdstop_second                  = {5},
/

""".format(model_initial_date.strftime('%Y'), model_initial_date.strftime('%m'),
           model_initial_date.strftime('%d'), model_initial_date.strftime('%H'),
           model_initial_date.strftime('%M'), model_initial_date.strftime('%S')))
    f.write("""&physics
mp_physics                          = {0}, {0},
ra_lw_physics                       = {1}, {1},
ra_sw_physics                       = {2}, {2},
radt                                = {3}, {3},
sf_sfclay_physics                   = {4}, {4},
sf_surface_physics                  = {5}, {5},
""".format(parameters['model_mp_phys'], parameters['model_lw_phys'],
           parameters['model_sw_phys'], parameters['model_radt'],
           parameters['model_sfclay_phys'], parameters['model_surf_phys']))
    f.write("""bl_pbl_physics                      = {0}, {0},
bldt                                = {1}, {1},
cu_physics                          = {2}, {3},
cudt                                = {4}, {4},
isfflx                              = {5},
ifsnow                              = {6},
icloud                              = {7},
surface_input_source                = 1,
num_soil_layers                     = {8},
sf_urban_physics                    = 0,  0,
do_radar_ref                        = {9}
/

&fdda
/

""".format(parameters['model_pbl_phys'], parameters['model_bldt'],
           parameters['model_cu_phys'], parameters['model_cu_phys_nest'],
           parameters['model_cudt'], parameters['model_use_surf_flux'],
           parameters['model_use_snow'], parameters['model_use_cloud'],
           parameters['model_soil_layers'], parameters['model_do_radar_ref']))
    f.write("""&dynamics
w_damping                           = {0},
diff_opt                            = {1},
km_opt                              = {2},
diff_6th_opt                        = 0,
diff_6th_factor                     = 0.12,
base_temp                           = {3},
damp_opt                            = {4},
zdamp                               = {5}, {5},
dampcoef                            = {6}, {6},
khdif                               = 0,      0,
kvdif                               = 0,      0,
non_hydrostatic                     = .true., .true.,
moist_adv_opt                       = 1,      1,
scalar_adv_opt                      = 1,      1,
mix_isotropic                       = 0,      0,
mix_upper_bound                     = 0.1     0.1,
iso_temp                            = 200.,
/

""".format(parameters['model_w_damping'], parameters['model_diff_opt'],
           parameters['model_km_opt'], parameters['model_tbase'],
           parameters['dampopt'], parameters['zdamp'], parameters['model_dampcoef']))
    f.write("""&bdy_control
spec_bdy_width                      = {0},
spec_zone                           = {1},
relax_zone                          = {2},
specified                           = .true., .false.,
nested                              = .false., .true.,
/

&grib2
/

&namelist_quilt
nio_tasks_per_group = 0,
nio_groups = 1,
/
""".format(parameters['assim_bzw'], parameters['model_spec_zone'],
           parameters['model_relax_zone']))
    f.close()


# Find date and time of model start and end
model_initial_date = increment_time(start_date, days=int(ndays))
model_end_date = increment_time(model_initial_date, hours=wrf_param['fct_len_hrs'])
datep = increment_time(model_initial_date, hours=-1)
print('Starting forecast for: '+str(model_initial_date), flush=True)

# Create the save directory
save_dir = wrf_param['dir_store']+model_initial_date.strftime('%Y%m%d%H')
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Determine number of input metgrid levels
# GFS changed from 27 to 32 on May 15, 2016
if model_initial_date < datetime(2016, 5, 15, 12):
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
create_wrf_namelist(namelist, wrf_param)

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
print('Finished with forecast initialized: '+model_initial_date)
