#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N analogue
#$ -q ancellcc
#$ -pe sm 32
#$ -P communitycluster
#$ -t 1-14:1

##############################################################################################
# Array job script for testing multiple thresholds/methods on both the inner and outer
# dataset domains.
#
# by Tyler Wixtrom
# Texas Tech University
# 18 July 2018
##############################################################################################
#thresholds=( 1 5 )
#stdevs=( 2 5 )
#idxs=( {1..1}{0..1}{0..1} )
domain=1
ID=$(( $SGE_TASK_ID - 1 ))
# ID=0
#domain=`echo ${idxs[${ID}]} | cut -b 1`
#thresh_idx=`echo ${idxs[${ID}]} | cut -b 2`
#stdevs_idx=`echo ${idxs[${ID}]} | cut -b 3`
#
#thresh=${thresholds[${thresh_idx}]}
#stdev=${stdevs[${stdevs_idx}]}
an_fhour=48
methods=(
        "rmse_pcpT00"
        "rmse_pcpT00+dewptT00"
        "rmse_pcpT00+dewptT00+mslpT00"
        "rmse_pcpT00+dewptf00"
        "rmse_pcpT00+dewptf00+mslpf00"
        "rmse_pcpT00+capeT-3"
        "rmse_pcpT00+dewptT-3"
        "pcp_area_rmse_pcpT00+dewptT-3"
        "pcp_area_rmse_pcpT00+dewptT-3+mslpT-3"
        "pcp_area_rmse_pcpT00+dewptf00+mslpf00"
        "pcp_area_rmse_pcpT00+temp_2mT00"
        "pcp_area_rmse_pcpT00+height_500hPaT00"
        "pcp_area_rmse_pcpT00+hgt500f00+capeT-3"
        "rmse_pcpT00+hgt500f00+capeT-3"
        )
method=${methods[${ID}]}
save_dir=/lustre/work/twixtrom/analogue_analysis_rename/f${an_fhour}/domain${domain}/${method}/
mkdir -p ${save_dir}
python_exec=/home/twixtrom/miniconda3/envs/research/bin/python
runscript=/home/twixtrom/analogue_algorithm/calc_analogue.py
${python_exec} ${runscript} ${domain} ${method} ${save_dir}
