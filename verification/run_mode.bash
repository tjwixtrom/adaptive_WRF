#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N run_mode
#$ -q omni
#$ -pe sm 1
#$ -P quanah
#$ -t 1-30:1
#
# by Tyler Wixtrom
# Texas Tech University
# Script for running MODE tool within a singularity container

module load singularity
datem=$1
met=/home/twixtrom/quanah.local/met-8.0.img
n=$(( ${SGE_TASK_ID} - 1 ))

date=`/home/twixtrom/adaptive_WRF/control_WRF/advance_time_python.py ${datem} ${n} 0`

for member in "control_thompson" "control_ETA" "adaptive_wrf"; do
for domain in 1 2; do
for fhour in 12 24 36; do

thresh=1
radius=4
area=0
interest=0.7
# make the output directory
savedir=/lustre/scratch/twixtrom/adaptive_verif/${member}/${date}/${domain}/${fhour}
mkdir -p ${savedir}

if [[ ${domain} -eq 1 ]]; then
    grid_res=12
else
    grid_res=4
fi

fname="met_data_fcst ${date} ${fhour} ${domain} ${member}"
obsname="met_data_obs ${date} ${fhour} ${domain}"

cat > ${savedir}/MODEConfig << END_INPUT
////////////////////////////////////////////////////////////////////////////////
//
// MODE configuration file.
//
// For additional information, see the MET_BASE/config/README file.
//
////////////////////////////////////////////////////////////////////////////////

//
// Output model name to be written
//
model = "WRF";

//
// Output description to be written
//
desc = "NA";

//
// Output observation type to be written
//
obtype = "MC_PCP";

////////////////////////////////////////////////////////////////////////////////

//
// Verification grid
//
regrid = {
   to_grid    = NONE;
   method     = NEAREST;
   width      = 1;
   vld_thresh = 0.5;
}

////////////////////////////////////////////////////////////////////////////////

//
// Approximate grid resolution (km)
//
grid_res = ${grid_res};

////////////////////////////////////////////////////////////////////////////////

//
// Run all permutations of radius and threshold
//
quilt = FALSE;

//
// Forecast and observation fields to be verified
//
fcst = {
   field = {
      name  = "${fname}";
   }

   censor_thresh      = [];
   censor_val         = [];
   conv_radius        = ${radius};
   conv_thresh        = >=${thresh};
   vld_thresh         = 0.5;
   filter_attr_name   = ["AREA"];
   filter_attr_thresh = [>=${area}];
   merge_thresh       = >=0.5;
   merge_flag         = THRESH;
}
obs = {
   field = {
      name  = "${obsname}";
   }

   censor_thresh      = [];
   censor_val         = [];
   conv_radius        = ${radius};
   conv_thresh        = >=${thresh};
   vld_thresh         = 0.5;
   filter_attr_name   = ["AREA"];
   filter_attr_thresh = [>=${area}];
   merge_thresh       = >=0.5;
   merge_flag         = THRESH;
};

//
// Handle missing data
//
mask_missing_flag = BOTH;

//
// Match objects between the forecast and observation fields
//
match_flag = MERGE_BOTH;

//
// Maximum centroid distance for objects to be compared
//
max_centroid_dist = 800.0/grid_res;

////////////////////////////////////////////////////////////////////////////////

//
// Verification masking regions
//
mask = {
   grid      = "";
   grid_flag = NONE; // Apply to NONE, FCST, OBS, or BOTH
   poly      = "";
   poly_flag = NONE; // Apply to NONE, FCST, OBS, or BOTH
}

////////////////////////////////////////////////////////////////////////////////

//
// Fuzzy engine weights
//
weight = {
   centroid_dist    = 2.0;
   boundary_dist    = 4.0;
   convex_hull_dist = 0.0;
   angle_diff       = 1.0;
   aspect_diff      = 0.0;
   area_ratio       = 1.0;
   int_area_ratio   = 2.0;
   curvature_ratio  = 0.0;
   complexity_ratio = 0.0;
   inten_perc_ratio = 0.0;
   inten_perc_value = 50;
}

////////////////////////////////////////////////////////////////////////////////

//
// Fuzzy engine interest functions
//
interest_function = {

   centroid_dist = (
      (            0.0, 1.0 )
      (  60.0/grid_res, 1.0 )
      ( 600.0/grid_res, 0.0 )
   );

   boundary_dist = (
      (            0.0, 1.0 )
      ( 400.0/grid_res, 0.0 )
   );

   convex_hull_dist = (
      (            0.0, 1.0 )
      ( 400.0/grid_res, 0.0 )
   );

   angle_diff = (
      (  0.0, 1.0 )
      ( 30.0, 1.0 )
      ( 90.0, 0.0 )
   );

   aspect_diff = (
      (  0.00, 1.0 )
      (  0.10, 1.0 )
      (  0.75, 0.0 )
   );

   corner   = 0.8;
   ratio_if = (
      (    0.0, 0.0 )
      ( corner, 1.0 )
      (    1.0, 1.0 )
   );

   area_ratio = ratio_if;

   int_area_ratio = (
      ( 0.00, 0.00 )
      ( 0.10, 0.50 )
      ( 0.25, 1.00 )
      ( 1.00, 1.00 )
   );

   curvature_ratio = ratio_if;

   complexity_ratio = ratio_if;

   inten_perc_ratio = ratio_if;
}

////////////////////////////////////////////////////////////////////////////////

//
// Total interest threshold for determining matches
//
total_interest_thresh = ${interest};

//
// Interest threshold for printing output pair information
//
print_interest_thresh = 0.0;

////////////////////////////////////////////////////////////////////////////////

//
// Plotting information
//
met_data_dir = "MET_BASE";

fcst_raw_plot = {
   color_table      = "MET_BASE/colortables/met_default.ctable";
   plot_min         = 0.0;
   plot_max         = 0.0;
   colorbar_spacing = 1;
}

obs_raw_plot = {
   color_table      = "MET_BASE/colortables/met_default.ctable";
   plot_min         = 0.0;
   plot_max         = 0.0;
   colorbar_spacing = 1;
}

object_plot = {
   color_table      = "MET_BASE/colortables/mode_obj.ctable";
}

//
// Boolean for plotting on the region of valid data within the domain
//
plot_valid_flag = FALSE;

//
// Plot polyline edges using great circle arcs instead of straight lines
//
plot_gcarc_flag = FALSE;

////////////////////////////////////////////////////////////////////////////////

//
// NetCDF matched pairs, PostScript, and contingency table output files
//
ps_plot_flag    = TRUE;
nc_pairs_flag   = {
   latlon       = TRUE;
   raw          = TRUE;
   object_raw   = TRUE;
   object_id    = TRUE;
   cluster_id   = TRUE;
   polylines    = TRUE;
}
ct_stats_flag   = TRUE;

////////////////////////////////////////////////////////////////////////////////

shift_right = 0;   //  grid squares

////////////////////////////////////////////////////////////////////////////////

output_prefix  = "";
version        = "V8.0";

////////////////////////////////////////////////////////////////////////////////



END_INPUT

singularity exec ${met} /home/twixtrom/adaptive_WRF/verification/mode.bash ${savedir}
done
done
done