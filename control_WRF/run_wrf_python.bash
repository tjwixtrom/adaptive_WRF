#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N run_control
#$ -P quanah
#$ -q omni
#$ -pe sm 36
#$ -l h_rt=10:00:00
#$ -t 32-62:1

#############################################################
#
#  Driver script for running adaptive WRF forecasts in
#  array jobs.
#
#############################################################

module load intel
module load impi
module load netcdf-serial
pythonexec=/home/twixtrom/miniconda3/envs/analogue/bin/python
runscript=/home/twixtrom/adaptive_WRF/control_WRF/run_wrf.py

if [ ${SGE_TASK_ID} -le 31 ] ; then
    member='thompson'
    ndays=$(( ${SGE_TASK_ID} - 1 ))
else
    member='ETA'
    ndays=$(( $SGE_TASK_ID - 32 ))
fi
${pythonexec} ${runscript} ${ndays} ${member}
