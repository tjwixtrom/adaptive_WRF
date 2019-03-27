#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N run_metgrid.bash
#$ -o sub_metgrid
#$ -e error_metgrid
#$ -q omni
#$ -pe sm 36
#$ -P quanah
#$ -l h_rt=10:00:00

#############################################################
#
#  Script to generate metgrid files for adaptive ensemble
#  project. Use same metgrid files for all control and
#  adaptive model iterations.
#
#############################################################

datem=2016070112
dateend=2016073112
module load intel
module load netcdf-serial

param=/home/twixtrom/adaptive_WRF/control_WRF/thompson_ysu_parameters.bash
source $param

while [ $datem -le $dateend ]; do
    echo 'Starting time '$datem
    echo 'at '`date`

    if [ -e ${dir_wps}/FILE* ]; then
    	  \rm ${dir_wps}/FILE*
    fi

    if [ -e ${dir_wps}/met_em_d0* ]; then
    	  \rm ${dir_wps}/met_em_d0*
    fi

    datef=`${scriptsdir}/advance_time_python.py $datem 0 $fct_len_hrs`

    rhour=`echo $datem | cut -b9-10`
    ryear=`echo $datem | cut -b1-4`
    rmonth=`echo $datem | cut -b5-6`
    rday=`echo $datem | cut -b7-8`

    ########### Get GFS data, run WPS ######################
    rm -r ${dir_wps}/data
    mkdir -p ${dir_wps}/data
    cd $dir_wps

    ###### Copy saved data from scratch to WPS/data
    # cp ${dir_compressed_gfs}/${datem}.tar.gz ${dir_wps}/data
    tar -zxvf ${dir_compressed_gfs}/${datem}.tar.gz -C ${dir_wps}/data
    for hour in `seq -w 0 6 48`; do
        cp -v ${dir_wps}/data/${datem}/gfs_${datem}_f${hour}.grb2 ${dir_wps}/data
    done
    rm -r ${dir_wps}/data/${datem}

    ##### If GFS data not there, don't do runs for that date

    if [ ! -e ${dir_wps}/data/gfs_${datem}_f00.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f06.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f12.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f18.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f24.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f30.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f36.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f42.grb2 ] || \
    [ ! -e ${dir_wps}/data/gfs_${datem}_f48.grb2 ] || \ ; then
        echo 'No GFS data for this date, quitting '$datem

    else

        mkdir -p ${dir_store}/${datem}

        ${dir_scripts}/make_namelist_wps.scr $param $datem

        cd $dir_wps
        ${dir_wps}/link_grib.csh ${dir_wps}/data/*.grb2
        if [ -e ${dir_wps}/ungrib.log ]; then
          \rm ${dir_wps}/ungrib.log
        fi

        ${dir_wps}/ungrib.exe

        \rm ${dir_wps}/GRIBFILE*
        \mv ${dir_wps}/ungrib.log ${dir_store}/${datem}/ungrib.log_${datem}

        if [ -e ${dir_wps}/metgrid.log ]; then
          \rm ${dir_wps}/metgrid.log
        fi

        ${dir_wps}/metgrid.exe
        \rm ${dir_wps}/FILE*
        \mv ${dir_wps}/metgrid.log ${dir_store}/${datem}/metgrid.log_${datem}

        mkdir -p ${dir_run}/${datem}/
        cp -r -v ${dir_wrf}/* ${dir_run}/${datem}/.
        cp -v ${dir_wps}/met_em.d0* ${dir_run}/${datem}/

        \cp ${dir_wps}/namelist.wps ${dir_store}/${datem}/namelist.wps.${datem}
        \rm ${dir_wps}/met_em.d0*
    fi

    datem=`${scriptsdir}/advance_time_python.py $datem 0 24`

done
