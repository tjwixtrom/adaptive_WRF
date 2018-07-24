#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N analogue
#$ -q ancellcc
#$ -pe sm 32
#$ -P communitycluster
#$ -t 1-2:1

# Array job script for testing multiple thresholds/methods on both the inner and outer
# dataset domains.
#
# by Tyler Wixtrom
# Texas Tech University
# 18 July 2018
thresholds=( 0.1 0.5 1 5 )
stdevs=( 0.5 1 2 5 10 )
idxs=( {1..2}{0..3}{0..4} )

ID=$(( $SGE_TASK_ID - 1 ))
domain=`echo ${idxs[${ID}]} | cut -b 1`
thresh_idx=`echo ${idxs[${ID}]} | cut -b 2`
stdevs_idx=`echo ${idxs[${ID}]} | cut -b 3`

thresh=${thresholds[${thresh_idx}]}
stdev=${stdevs[${stdevs_idx}]}

method="rmse"
save_dir=/home/twixtrom/analogue_analysis/${method}/${thresh}/std${stdev}/
mkdir -p ${save_dir}
python_exec=/home/twixtrom/miniconda3/envs/analogue/bin/python
runscript=/home/twixtrom/analogue_algorithm/test_analogue.py
${python_exec} ${runscript} ${domain} ${thresh} ${stdev} ${save_dir}
