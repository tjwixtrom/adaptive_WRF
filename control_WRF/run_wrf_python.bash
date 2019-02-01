#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N run_thompson
#$ -P quanah
#$ -q omni
#$ -pe sm 36
#$ -l h_rt=10:00:00
#$ -t 1-14:1

#############################################################
#
#  Driver script for running adaptive WRF forecasts in
#  array jobs.
#
#############################################################

module load intel
module load impi
module load netcdf-serial

runscript=/home/twixtrom/adaptive_WRF/control_WRF/run_wrf.py
ndays=$(( $SGE_TASK_ID - 1 ))

${runscript} ${ndays}
