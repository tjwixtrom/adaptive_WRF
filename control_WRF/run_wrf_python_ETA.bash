#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N run_ETA
#$ -P quanah
#$ -q omni
#$ -pe sm 36
#$ -l h_rt=10:00:00
#$ -t 26-26:1

#############################################################
#
#  Driver script for running adaptive WRF forecasts in
#  array jobs.
#
#############################################################

module load intel
module load impi
module load netcdf-serial

runscript=/home/twixtrom/adaptive_WRF/control_WRF/run_wrf_ETA.py
ndays=$(( $SGE_TASK_ID - 1 ))

${runscript} ${ndays}
