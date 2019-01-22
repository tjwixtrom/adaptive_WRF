#! /bin/bash
########################################################################
#
#   WRF_parameters.bash - Parameter file
#
########################################################################

  rootdir=/home/twixtrom/adp_ens/dataset     # System root directory
  scriptsdir=/home/twixtrom/adp_ens/dataset/scripts    # Scripts directory
  wrfdir=/lustre/work/twixtrom  # WRF core root directory
  dir_run=/lustre/scratch/twixtrom/adp_dataset_run

  #  Domain-Specific Parameters
  #*************************************************

  csys=1                      # 1 for quanah, 2 for ancellcc
  norm_cores=180               # cores per run normal
  cc_cores=32                 # cores per run ancellcc

  let model_Nx1=508           # number of grid points in x-direction
  let model_Ny1=328           # number of grid points in y-direction
  let model_Nz=38             # number of grid points in vertical
  let model_ptop=5000         # Pressure at model top
  let model_gridspx1=12000    # gridspacing in x (in meters)
  let model_gridspy1=12000    # gridspacing in y
  let dt=72                   # model time step (in sec)
  model_centlat=38.0          # center latitude of domain
  model_centlon=-103.0        # center longitude of domain
  model_stdlat1=30.0          # first true latitude of domain
  model_stdlat2=60.0          # second true latitude of domain
  model_stdlon=-101.0          # standard longitude
  let dlbc=360                # number of minutes in between global model BCs
  output_interval=180         # Frequency of model output to file mother domain
  output_intervalNEST=60      # Frequency of model output to file - Nest
  model_num_in_output=10000   # Output times per file
  fct_len=2880                # Minutes to forecast for
  feedback=0                  # 1-way(0) or 2-way nest(1)
  enum=20                      # Number of physics runs
  siz1=29538340980            # File size dom 1
  siz2=15445197060            # File size dom 2

 # Nested domains info
 #*************************************************
  let model_gridspx1_nest=4000
  let model_gridspy1_nest=4000
  iparent_st_nest=200
  jparent_st_nest=80
  model_Nx1_nest=322
  model_Ny1_nest=271
  parent_id_nest=1
  grid_ratio_nest=3

  #  Locations of important directories
  #*************************************************

  dir_wps=${wrfdir}/WPSV3.5.1quanah                   # WPS location
  dir_wrf=${wrfdir}/WRFV3.5.1quanah/run               # WRF location
  dir_rip=${rootdir}/RIPBMA                      # RIP location
  dir_scripts=${scriptsdir}/scripts                 # shell scripts
  dir_sub=${rootdir}/sub                         # submit to queue scripts
  dir_log=${rootdir}/log                         # log directory
  dir_store=/lustre/scratch/twixtrom/adp_dataset_members   # storage directory
  dir_members=$WORK/adp_dataset_members
  dir_scratch=/lustre/scratch/twixtrom
  dir_gfs=/lustre/scratch/twixtrom/gfs_data
 # dir_scratchstore=/lustre/scratch/twixtrom/dataset_runs

 # Parameters for the model (not changed very often)
 #**************************************************
  model_mp_phys=8          # microphysics scheme
  let model_spec_zone=1    # number of grid points with tendencies
  let model_relax_zone=4   # number of blended grid points
  dodfi=0                  # Do Dfi 3-yes 0-no
  model_lw_phys=1          # model long wave scheme
  model_sw_phys=1          # model short wave scheme
  model_radt=30            # radiation time step (in minutes)
  model_sfclay_phys=1      # surface layer physics
  model_surf_phys=2        # land surface model
  model_pbl_phys=1         # pbl physics
  model_bldt=0             # boundary layer time steps (0 : each time steps, in min)
  model_cu_phys=6          # cumulus param
  model_cu_phys_nest=0     # cumulus param 3km
  model_cudt=5             # cumuls time step
  model_use_surf_flux=1    # 1 is yes
  model_use_snow=0
  model_use_cloud=1
  model_soil_layers=4
  model_w_damping=1
  model_diff_opt=1
  model_km_opt=4
  model_dampcoef=0.2
  model_tbase=300.
  model_nwp_diagnostics=1
  model_do_radar_ref=1
  dampopt=3
  zdamp=5000.

  #************************************
  # Calculated terms

  let fct_len_hrs='fct_len/60'
  let dlbc_hrs='dlbc/60'
  let assim_bzw='model_spec_zone+model_relax_zone'
  let otime='output_interval/60'
  let otime_nest='output_intervalNEST/60'
