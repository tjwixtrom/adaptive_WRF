#!/home/twixtrom/miniconda3/envs/analogue/bin/python
##############################################################################################
# test_analogue.py - Code for testing analogue methods
#
# Code to calculate analogues for each test forecast
# and save the ranking to a numpy file as a list
# Also outputs various plots to analyze the analogue performance
# Tests code found in calc.py
#
# Analogues are calculated based on a subset of the total dataset
# Performance is then determined by ranking the analogue selected
# best member and comparing in timeseries to the mean best over the
# analogue dataset.
#
# by Tyler Wixtrom
# Texas Tech University
# 20 July 2018
#
##############################################################################################

from datetime import datetime, timedelta
from netCDF4 import num2date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from scipy.ndimage import gaussian_filter
import warnings
import xarray as xr

from analogue_algorithm.calc import rmse, find_analogue_rmse, verify_members
from analogue_algorithm.plots import plot_panels
warnings.filterwarnings("ignore")
plt.switch_backend('agg')

domain = sys.argv[1]
thresh = sys.argv[2]
stdev = sys.argv[3]
save_dir = sys.argv[4]
method = 'rmse'
verif_std = 2
verif_thresh = 0.5

# save_dir = '/home/twixtrom/analogue_analysis/'+method+'/'+thresh+'/' + \
#            'std'+str(stdev)+'/'
fhour = 48

if domain == '1':
    datafile = '/lustre/work/twixtrom/adp_dataset_12km_48h_pcp.nc'
    obsfile = '/lustre/work/twixtrom/ST4_2015_03h.nc'
    outname_mp = save_dir+'mp_an_ranks_12km_48h_'+method+'_'+thresh+'_std'+stdev+'.npy'
    outname_pbl = save_dir+'pbl_an_ranks_12km_48h_'+method+'_'+thresh+'_std'+stdev+'.npy'

elif domain == '2':
    datafile = '/lustre/work/twixtrom/adp_dataset_4km_48h_pcp.nc'
    obsfile = '/lustre/work/twixtrom/ST4_2015_01h.nc'
    outname_mp = save_dir+'mp_an_ranks_4km_48h_'+method+'_'+thresh+'_std'+stdev+'.npy'
    outname_pbl = save_dir+'pbl_an_ranks_4km_48h_'+method+'_'+thresh+'_std'+stdev+'.npy'

else:
    raise ValueError('Domain not defined')

threshold = float(thresh)
data_start_date = datetime(2015, 1, 1, 12)
an_start_date = datetime(2015, 10, 1, 12)
an_end_date = datetime(2015, 12, 29, 12)

mem_list = ['mem'+str(i) for i in range(1, 21)]
mp_list = ['mem'+str(i) for i in range(1, 11)]
pbl_list = ['mem'+str(i) for i in range(11, 21)]

print('Opening Dataset')
pcp = xr.open_dataset(datafile, chunks={'time': 1}, decode_cf=False)
vtimes_pcp = num2date(pcp.time, pcp.time.units)
pcp.coords['time'] = np.array([np.datetime64(date) for date in vtimes_pcp])
pcp['time'] = np.array([np.datetime64(date) for date in vtimes_pcp])
pcp.coords['lat'] = pcp.lat
pcp.coords['lon'] = pcp.lon


stage4 = xr.open_dataset(obsfile, chunks={'time': 1}, decode_cf=False)
vtimes_stage4 = num2date(stage4.valid_times, stage4.valid_times.units)
stage4.coords['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4.coords['lat'] = stage4.lat
stage4.coords['lon'] = stage4.lon


fcst_mean = xr.concat([pcp[mem] for mem in mem_list], dim='Member').mean(dim='Member')
pcp['mean'] = fcst_mean


an_best_mp = []
an_best_pbl = []
date = an_start_date
while date <= an_end_date:
    print('Starting date '+str(date))
    an_idx, fcst_smooth = find_analogue_rmse(date, pcp, threshold, float(stdev))
    if np.isnan(an_idx):
        an_best_mp.append(('nan', np.nan, date, np.nan))
        an_best_pbl.append(('nan', np.nan, date, np.nan))
    else:
        print('Analogue date selected '+str(vtimes_pcp[an_idx]))
        # Get the analogue's verification
        analogue_time = vtimes_pcp[an_idx] + timedelta(hours=fhour)
        st4_an = stage4.total_precipitation.sel(
            time=analogue_time,
            drop=True
            ).where(fcst_smooth >= threshold)

        # Find best member for analouge date by RMSE
        # for MP members
        an_rmse_mp = []
        for mem in mp_list:
            mem_data = pcp[mem][an_idx, ].where(fcst_smooth >= threshold)
            rmse_mem = rmse(mem_data, st4_an)
            an_rmse_mp.append(rmse_mem)
        an_mp = np.array(an_rmse_mp)

        # for PBL members
        an_rmse_pbl = []
        for mem in pbl_list:
            mem_data = pcp[mem][an_idx, ].where(fcst_smooth >= threshold)
            rmse_mem = rmse(mem_data, st4_an)
            an_rmse_pbl.append(rmse_mem)
        an_pbl = np.array(an_rmse_pbl)

        # Get best member for both MP and PBL
        # If precip area is outside StageIV grid, nan slice is returned and value error from
        # nanargmin. In this case, there is no best analouge
        try:
            best_mp = mp_list[np.nanargmin(an_mp)]
        except ValueError:
            best_mp = 'nan'

        try:
            best_pbl = pbl_list[np.nanargmin(an_pbl)]
        except ValueError:
            best_pbl = 'nan'

        # Find actual best verifying members
        st4_fcst_date = date + timedelta(hours=fhour)
        st4_verif = stage4.total_precipitation.sel(
            time=st4_fcst_date,
            drop=True
            ).where(fcst_smooth >= threshold)

        # For MP members
        verif_mp = []
        for mem in mp_list:
            mem_data = pcp[mem].sel(
                time=date,
                drop=True
                ).where(fcst_smooth >= threshold)
            rmse_mem = rmse(mem_data, st4_verif)
            verif_mp.append(rmse_mem)
        verif_members_mp = np.array(verif_mp)

        # Sort members by performance
        sorted_mp = np.array(mp_list)[np.argsort(verif_members_mp)]

        # get rank of predicted best
        if best_mp == 'nan':
            an_best_mp.append(('nan', np.nan, date, np.nan))
        else:
            rank_mp = np.where(sorted_mp == best_mp)[0][0]
            an_best_mp.append((best_mp, rank_mp, date, vtimes_pcp[an_idx]))

        # For PBL members
        verif_pbl = []
        for mem in pbl_list:
            mem_data = pcp[mem].sel(
                time=date,
                drop=True
                ).where(fcst_smooth >= threshold)
            rmse_mem = rmse(mem_data, st4_verif)
            verif_pbl.append(rmse_mem)
        verif_members_pbl = np.array(verif_pbl)

        # Sort members by performance
        sorted_pbl = np.array(pbl_list)[np.argsort(verif_members_pbl)]

        # get rank of predicted best
        if best_pbl == 'nan':
            an_best_pbl.append(('nan', np.nan, date, np.nan))
        else:
            rank_pbl = np.where(sorted_pbl == best_pbl)[0][0]
            an_best_pbl.append((best_pbl, rank_pbl, date, vtimes_pcp[an_idx]))
    date += timedelta(days=1)

print('Writing output to file')
np.save(outname_mp, an_best_mp)
np.save(outname_pbl, an_best_pbl)

print('Analyzing Output')
mp_ranks = []
for case in an_best_mp:
    mp_ranks.append(case[1])

pbl_ranks = []
for case in an_best_pbl:
    pbl_ranks.append(case[1])

an_pbl = [case[0] for case in an_best_pbl]
an_mp = [case[0] for case in an_best_mp]

print('Generating Histograms')
xlab = 'Member'
ylab = 'Selected Frequency'
ax, ax2 = plot_panels((16, 16), (2, 2), 2,
                      grid=True,
                      x_labels=[xlab],
                      y_labels=[ylab],
                      ylim=(0, 30))
ax.hist(an_mp, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        range=(0, 10), facecolor='green', align='left', rwidth=0.5)
ax.set_title('Histogram of Observed Analogue MP Scheme Selection')

ax2.hist(an_pbl, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         range=(0, 10), facecolor='orange', align='left', rwidth=0.3)
ax2.set_title('Histogram of Observed Analogue PBL Scheme Selection')
plt.savefig(save_dir+method+'_'+thresh+'_std'+str(stdev)+'_' +
            str(fhour)+'_d0'+str(domain)+'_mem_selection.png')

xlab = 'Ranking'
ylab = 'Observed Frequency'
ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ax, ax2 = plot_panels((16, 16), (2, 2), 2,
                      grid=True,
                      x_labels=[xlab],
                      y_labels=[ylab],
                      ylim=(0, 20))
ax.hist(np.array(mp_ranks) + 1, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        range=(0, 10), facecolor='green', align='left', rwidth=0.4)
ax.set_title('Histogram of Observed Analogue MP Member Ranking')
ax.set_xticks(ticks)
ax2.hist(np.array(pbl_ranks) + 1, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         range=(0, 10), facecolor='orange', align='left', rwidth=0.5)
ax2.set_title('Histogram of Observed Analogue PBL Member Ranking')
ax2.set_xticks(ticks)
plt.savefig(save_dir+method+'_'+thresh+'_std'+str(stdev)+'_'+str(fhour)+'_d0' +
            str(domain)+'_ranks.png')

# Find average best member
print('Finding Best Member')
tot_rmse = verify_members(pcp, stage4.total_precipitation, fhour, verif_thresh,
                          verif_std, data_start_date, an_start_date)

mean_best_mp = mp_list[np.array([tot_rmse[mem]] for mem in mp_list).argmin()]
mean_best_pbl = pbl_list[np.array([tot_rmse[mem]] for mem in pbl_list).argmin()]

# Compare analogue physics with average best physics
print('Generating Timeseries')
rmse_an_pbl = []
rmse_best_pbl = []
an_pbl_mem = [case[0] for case in an_best_pbl]
an_dates = [case[-2] for case in an_best_pbl]
for i in range(len(an_dates)):
    # find index for current data, use all data up to that point
    if an_pbl_mem[i] == 'nan':
        rmse_an_pbl.append(np.nan)
        rmse_best_pbl.append(np.nan)
    else:
        date = an_dates[i]
        st4_date = date + timedelta(hours=fhour)

        # Smooth and mask observed precip
        st4_smooth = gaussian_filter(stage4[
                                            'total_precipitation'
                                            ].sel(
                                                  time=st4_date,
                                                  drop=True
                                                  ), float(stdev))
        st4_verif = stage4[
                            'total_precipitation'
                            ].sel(
                                  time=st4_date,
                                  drop=True
                                  ).where(st4_smooth >= threshold)

        # get analogue precip and MYNN precip
        an_pcp = pcp[an_pbl_mem[i]].sel(
                                        time=date,
                                        drop=True
                                        ).where(st4_smooth >= threshold)
        best_pcp = pcp[mean_best_pbl].sel(
                                          time=date,
                                          drop=True
                                          ).where(st4_smooth >= threshold)
        rmse_an_pbl.append(rmse(an_pcp, st4_verif))
        rmse_best_pbl.append(rmse(best_pcp, st4_verif))


# For MP Members
rmse_an_mp = []
rmse_best_mp = []
an_mp_mem = [case[0] for case in an_best_mp]
for i in range(len(an_dates)):
    # find index for current data, use all data up to that point
    if an_mp_mem[i] == 'nan':
        rmse_an_mp.append(np.nan)
        rmse_best_mp.append(np.nan)
    else:
        date = an_dates[i]
        st4_date = date + timedelta(hours=fhour)

        # Smooth and mask observed precip
        st4_smooth = gaussian_filter(stage4[
                                            'total_precipitation'
                                            ].sel(
                                                  time=st4_date,
                                                  drop=True
                                                  ), float(stdev))
        st4_verif = stage4[
                            'total_precipitation'
                            ].sel(
                                  time=st4_date,
                                  drop=True
                                  ).where(st4_smooth >= threshold)

        # get analogue precip and MYNN precip
        an_pcp = pcp[an_mp_mem[i]].sel(
                                       time=date,
                                       drop=True
                                       ).where(st4_smooth >= threshold)
        best_pcp = pcp[mean_best_mp].sel(
                                         time=date,
                                         drop=True
                                         ).where(st4_smooth >= threshold)
        rmse_an_mp.append(rmse(an_pcp, st4_verif))
        rmse_best_mp.append(rmse(best_pcp, st4_verif))


print('Making Plots')
fdates = [case[-2] for case in an_best_pbl]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
plt.plot(fdates, np.array(rmse_an_mp),
         label='Analogue',
         color='tab:red',
         linewidth=0.7)
plt.plot(fdates, np.array(rmse_best_mp),
         label='Best MP - '+mean_best_mp,
         color='tab:green',
         linewidth=0.7)
plt.title('Analogue and Average Best Hybrid RMSE for MP Members')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
plt.legend(shadow=True, fontsize='large', loc=0)
plt.grid()
plt.savefig(save_dir+method+'_'+thresh+'_std'+str(stdev)+'_' +
            str(fhour)+'_d0'+str(domain)+'_an_vs_best_mp.png')

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
plt.plot(fdates, np.array(rmse_an_pbl),
         label='Analogue',
         color='tab:red',
         linewidth=0.7)
plt.plot(fdates, np.array(rmse_best_pbl),
         label='Best PBL - '+mean_best_pbl,
         color='tab:green',
         linewidth=0.7)
plt.title('Analogue and Average Best Hybrid RMSE for PBL Members')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
plt.legend(shadow=True, fontsize='large', loc=0)
plt.grid()
plt.savefig(save_dir+method+'_'+thresh+'_std'+str(stdev)+'_'+str(fhour) +
            '_d0'+str(domain)+'_an_vs_best_pbl.png')
print('Analogue Analysis Completed')
