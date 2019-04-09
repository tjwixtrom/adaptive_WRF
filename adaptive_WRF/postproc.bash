#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N post_adaptive
#$ -q omni
#$ -pe sm 36
#$ -P quanah
#$ -t 1-7:1

# Array job script for running UPP/unipost for each member in an ensemble set
#
# by Tyler Wixtrom
# Texas Tech University
# 27 March 2018
#date1=2016070212
n=$(( $SGE_TASK_ID - 1 ))
#if [ $SGE_TASK_ID -le 31 ] ; then
#    date1=2016010212
#    ndays=$(( $SGE_TASK_ID - 1 ))
#else
#    date1=2016070112
#    ndays=$(( $SGE_TASK_ID - 32 ))
dates=( 2016010412 2016011412 2016012912 2016051112 2016052112 2016071112 2016072612 )
runscript=/home/twixtrom/adaptive_WRF/adaptive_WRF/pwpp.py
datem=${dates[${n}]}
#datem=`/home/twixtrom/adaptive_WRF/control_WRF/advance_time_python.py ${date1} ${ndays} 0`

mkdir -p /lustre/scratch/twixtrom/adaptive_wrf_post/adaptive_wrf/${datem}

# Process for domain 1
echo 'Processing Domain 1'
infile1=/lustre/scratch/twixtrom/adaptive_wrf_save/adaptive_wrf/${datem}/wrfout_d01_${datem}.nc
outfile1=/lustre/scratch/twixtrom/adaptive_wrf_post/adaptive_wrf/${datem}/wrfprst_d01_${datem}.nc

${runscript} ${infile1} ${outfile1}

# Process for domain 2
echo 'Processing Domain 2'
infile2=/lustre/scratch/twixtrom/adaptive_wrf_save/adaptive_wrf/${datem}/wrfout_d02_${datem}.nc
outfile2=/lustre/scratch/twixtrom/adaptive_wrf_post/adaptive_wrf/${datem}/wrfprst_d02_${datem}.nc

${runscript} ${infile2} ${outfile2}

echo 'Complete Date ' $datem
