#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N run_adaptive
#$ -P quanah
#$ -q omni
#$ -pe sm 36
#$ -l h_rt=10:00:00
#$ -t 1-30:1

#############################################################
#
#  Driver script for running adaptive WRF forecasts in
#  array jobs.
#
#############################################################

module load intel
module load impi
module load netcdf-serial

python_exec=/home/twixtrom/miniconda3/envs/analogue/bin/python
runscript=/home/twixtrom/adaptive_WRF/adaptive_WRF/run_adaptive_wrf_retro.py
n=$(( $SGE_TASK_ID - 1 ))
#if [[ ${ndays} -eq 0 ]] ; then
#    rm /home/twixtrom/adaptive_WRF/adaptive_WRF/an_selection_log_201605.log
#fi
#dates=( "2016-01-04T12" "2016-01-14T12" "2016-01-29T12" "2016-05-11T12" "2016-05-21T12" "2016-07-11T12" "2016-07-26T12")
${python_exec} ${runscript} ${n}
