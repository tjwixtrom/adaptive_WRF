#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N post_control
#$ -q omni
#$ -pe sm 36
#$ -P quanah
#$ -t 1-62:1

# Array job script for running UPP/unipost for each member in an ensemble set
#
# by Tyler Wixtrom
# Texas Tech University
# 27 March 2018

date1=2016070112
if [ ${SGE_TASK_ID} -le 31 ] ; then
    name='control_ETA'
    ndays=$(( $SGE_TASK_ID - 1 ))
else
    name='control_thompson'
    ndays=$(( $SGE_TASK_ID - 32 ))
fi

runscript=/home/twixtrom/adaptive_WRF/control_WRF/pwpp.py

datem=`/home/twixtrom/adaptive_WRF/control_WRF/advance_time_python.py ${date1} ${ndays} 0`

mkdir -p /lustre/scratch/twixtrom/adaptive_wrf_post/${name}/${datem}

# Process for domain 1
echo 'Processing Domain 1'
infile1=/lustre/scratch/twixtrom/adaptive_wrf_save/${name}/${datem}/wrfout_d01_${datem}.nc
outfile1=/lustre/scratch/twixtrom/adaptive_wrf_post/${name}/${datem}/wrfprst_d01_${datem}.nc

${runscript} ${infile1} ${outfile1}

# Process for domain 2
echo 'Processing Domain 2'
infile2=/lustre/scratch/twixtrom/adaptive_wrf_save/${name}/${datem}/wrfout_d02_${datem}.nc
outfile2=/lustre/scratch/twixtrom/adaptive_wrf_post/${name}/${datem}/wrfprst_d02_${datem}.nc

${runscript} ${infile2} ${outfile2}

echo 'Complete Date ' $datem
