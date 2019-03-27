"""
wrf.py - Functions for analogue score calculation

by Tyler Wixtrom
Texas Tech University
7 February 2019

Helper code for WRF file operations.

"""
import glob
from pandas import Timedelta


def increment_time(date1, days=0, hours=0, minutes=0, seconds=0):
    """
    Increment time from start by a specified number of days or hours

    Parameters:
        date1: pandas.datetime
        days: int, number of days to advance
        hours: int, number of hours to advance
    Returns: pandas.datetime, incremented time and date

    """
    return date1 + Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


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


def create_wrf_namelist(fname, parameters, model_initial_date):
    """
    Create a WRF control namelist

    :param fname: str, output file name string
    :param parameters: dict, WRF configuration parameters
    :param model_initial_date: pandas.Timestamp, datetime object of model initialization
    :return: Saves WRF control namelist as fname
    """
    model_end_date = increment_time(model_initial_date, hours=parameters['fct_len_hrs'])
    datep = increment_time(model_initial_date, hours=-1)
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


def check_logs(infile, logfile, date, wrf=False):
    """
    Check if success message is in real/wrf output logs

    Parameters
    infile: string file path to log summary
    logfile: path to output log
    wrf: bool, whether this is a log for wrf or real

    """
    f = open(logfile, 'a+')
    logfile = open(infile)
    last = logfile.readlines()[-1]
    find = last.find('SUCCESS COMPLETE')
    if find == -1:
        if wrf:
            f.write('WRF not complete ' + str(date)+'\n')
        else:
            f.write('REAL not complete ' + str(date)+'\n')
