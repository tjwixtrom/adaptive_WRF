#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N run_thompson
#$ -P quanah
#$ -q omni
#$ -pe sm 36
#$ -l h_rt=10:00:00
#$ -t 1-2:1

#############################################################
#
#  Script to generate control forecasts for adaptive
#  parameterization project validation
#
#############################################################
module load intel
module load impi
module load netcdf-serial

param=/home/twixtrom/adaptive_WRF/control_WRF/thompson_ysu_parameters.bash
source $param

ndays=$(( $SGE_TASK_ID - 1 ))
date_start=2016050112
datem=`${scriptsdir}/advance_time_python.py ${date_start} ${ndays} 0`


echo 'Starting time '$datem
echo 'at '`date`

datef=`${scriptsdir}/advance_time.scr $datem $fct_len_hrs`

rhour=`echo $datem | cut -b9-10`
ryear=`echo $datem | cut -b1-4`
rmonth=`echo $datem | cut -b5-6`
rday=`echo $datem | cut -b7-8`



############ submit real.exe,wrf.exe for all configurations
${dir_scripts}/make_namelist_wrf.scr $param $datem

# \cp ${dir_wrf}/namelist.input ${dir_run}/${datem}/mem${ie}
#
# if [ -e ${dir_run}/${datem}/wrf_done ]; then
# \rm ${dir_run}/${datem}/wrf_done
# fi

cd ${dir_run}/${datem}/
rm ${dir_run}/${datem}/wrfout*

# mpirun -np $norm_cores -machinefile ${dir_sub}/machinefile.${JOB_ID} ${dir_run}/${datem}/real.exe
mpirun -np $norm_cores ${dir_run}/${datem}/real.exe

cat ${dir_run}/${datem}/rsl* > ${dir_store}/${datem}/rslout_real_${datem}.log
\rm ${dir_run}/${datem}/rsl.*

# mpirun -np $norm_cores -machinefile ${dir_sub}/machinefile.${JOB_ID} ${dir_run}/${datem}/wrf.exe
mpirun -np $norm_cores ${dir_run}/${datem}/wrf.exe

#touch ${dir_run}/${datem}/mem${emem}/wrf_done

mkdir ${dir_store}/${datem}/

cat ${dir_run}/${datem}/rsl* > ${dir_store}/${datem}/rslout_wrf_${datem}
\rm ${dir_run}/${datem}/rsl.*

\mv ${dir_run}/${datem}/wrfout_d01* \
${dir_store}/${datem}/wrfout_d01_${datem}.nc

\mv ${dir_run}/${datem}/wrfout_d02* \
${dir_store}/${datem}/wrfout_d02_${datem}.nc

# \mv ${dir_sub}/run_wrf_member${emem}_${datem}.scr \
# ${dir_store}/${datem}/mem${emem}/run_wrf_member${emem}_${datem}
#
# \mv ${dir_sub}/submit_wrf_member${emem}_${datem} \
# ${dir_store}/${datem}/mem${ie}/submit_wrf_member${emem}_${datem}
#
# \mv ${dir_sub}/error_wrf_member${emem}_${datem} \
# ${dir_store}/${datem}/mem${emem}/error_wrf_member${emem}_${datem}
# cd $dir_sub
# ${dir_scripts}/sub_member_adp.scr \
# /home/twixtrom/adp_ens/dataset/scripts/mem_param/WRF_parameters_mem${ie}.bash \
# $ie $dir_sub 72 $datem
#
# chmod 755 ${dir_sub}/run_wrf_member${ie}_${datem}.scr
# cd $dir_sub
# qsub ${dir_sub}/run_wrf_member${ie}_${datem}.scr


echo 'Done with time '$datem
echo 'at '`date`


# \rm ${dir_sub}/machinefile.*
#
# \mv ${scriptsdir}/error_out ${dir_store}/${datem}/error_${datem}.out
#
# \mv ${scriptsdir}/submit_out ${dir_store}/${datem}/submit_${datem}.out
#
# \mv ${scriptsdir}/machinefile* ${dir_store}/${datem}/machinfile_${datem}.out
