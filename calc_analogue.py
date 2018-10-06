#!/home/twixtrom/miniconda3/envs/analogue/bin/python
##############################################################################################
# calc_analogue.py - Code for testing analogue methods
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
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import operator
from netCDF4 import num2date
from scipy.ndimage import gaussian_filter

from analogue_algorithm.calc import rmse, find_analogue, find_analogue_precip_area, \
    verify_members
from analogue_algorithm.plots import plot_panels

warnings.filterwarnings("ignore")
plt.switch_backend('agg')

# chunks = {'time': 10}
chunks = None
param = {
    'domain': sys.argv[1],
    'sigma': 2.,
    'directory': sys.argv[3],
    'method': sys.argv[2],
    'forecast_hour': float(sys.argv[4]),
    'pcp_threshold': float(sys.argv[5]),
    'pcp_operator': operator.ge,
    'dewpt_threshold': float(sys.argv[7]),
    'dewpt_operator': operator.ge,
    'mslp_threshold': float(sys.argv[9]),
    'mslp_operator': operator.le,
    'cape_threshold': float(sys.argv[6]),
    'cape_operator': operator.ge,
    'temp_2m_threshold': None,
    'temp_2m_operator': None,
    'height_500hPa_threshold': float(sys.argv[8]),
    'height_500hPa_operator': operator.le,
    'start_date': '2015-01-01T12:00:00',
    'an_start_date': '2015-10-01T12:00:00',
    'an_end_date': '2015-12-29T12:00:00',
    'dt': '1D',
    'comments': 'Method testing for mslp threshold sensitivity'
    }

verif_param = {
    'forecast_hour': float(sys.argv[4]),
    'threshold': 10.,
    'sigma': 2,
    'start_date': '2015-01-01T12:00:00',
    'end_date': '2015-09-30T12:00:00',
    'dt': '1D'
    }

# Save parameters to ouput metadata file.
f = open(param['directory']+param['method']+'_analogue_param.txt', 'w')
f.write('Analogue Selection Parameters\n')
for key in param.keys():
    f.write(key+': '+str(param[key])+'\n')
f.write('\n')
f.write('\n')
f.write('Verification Parameters\n')
for key in verif_param.keys():
    f.write(key+': '+str(verif_param[key])+'\n')
f.close()

if param['domain'] == '1':
    dx = '12km'
    obsfile = '/lustre/scratch/twixtrom/ST4_2015_03h.nc'
    outname_mp = param['directory']+'mp_an_ranks_12km_48h_'+param['method']+'.npy'
    outname_pbl = param['directory']+'pbl_an_ranks_12km_48h_'+param['method']+'.npy'
elif param['domain'] == '2':
    dx = '4km'
    obsfile = '/lustre/scratch/twixtrom/ST4_2015_01h.nc'
    outname_mp = param['directory']+'mp_an_ranks_4km_48h_'+param['method']+'.npy'
    outname_pbl = param['directory']+'pbl_an_ranks_4km_48h_'+param['method']+'.npy'

else:
    raise ValueError('Domain not defined')

mem_list = ['mem'+str(i) for i in range(1, 21)]
mp_list = ['mem'+str(i) for i in range(1, 11)]
pbl_list = ['mem1', *['mem'+str(i) for i in range(11, 21)]]

print('Opening Dataset', flush=True)


def open_pcp(hour, domain, dx):
    pcpfile = '/lustre/scratch/twixtrom/adp_dataset_'+dx+'_timestep_pcp_f'+str(int(hour))+'.nc'
    precip = xr.open_dataset(pcpfile, chunks=chunks)
    precip.attrs['threshold'] = param['pcp_threshold']
    precip.attrs['sigma'] = param['sigma']
    precip.attrs['operator'] = param['pcp_operator']
    fcst_mean = xr.concat([precip[mem] for mem in mem_list],
                          dim='Member').mean(dim='Member')
    precip['mean'] = fcst_mean
    return precip.where(precip.time < np.datetime64('2016-01-01T12:00:00'), drop=True)


def open_dewpt(hour, domain, dx):
    dewptfile = '/lustre/scratch/twixtrom/adp_dataset_'+dx+'_dewpt_2m_f'+str(int(hour))+'.nc'
    dew = xr.open_dataset(dewptfile, chunks=chunks)
    dew.attrs['threshold'] = param['dewpt_threshold']
    dew.attrs['sigma'] = param['sigma']
    dew.attrs['operator'] = param['dewpt_operator']
    dew_mean = xr.concat([dew[mem] for mem in mem_list],
                         dim='Member').mean(dim='Member')
    dew['mean'] = dew_mean
    return dew


def open_mslp(hour, domain, dx):
    mslpfile = '/lustre/scratch/twixtrom/adp_dataset_'+dx+'_mslp_f'+str(int(hour))+'.nc'
    slp = xr.open_dataset(mslpfile, chunks=chunks)
    slp.attrs['threshold'] = param['mslp_threshold']
    slp.attrs['sigma'] = param['sigma']
    slp.attrs['operator'] = param['mslp_operator']
    slp_mean = xr.concat([slp[mem] for mem in mem_list],
                         dim='Member').mean(dim='Member')
    slp['mean'] = slp_mean
    return slp


def open_cape(hour, domain, dx):
    file = '/lustre/scratch/twixtrom/adp_dataset_'+dx+'_cape_f'+str(int(hour))+'.nc'
    cape = xr.open_dataset(file, chunks=chunks)
    cape.attrs['threshold'] = param['cape_threshold']
    cape.attrs['sigma'] = param['sigma']
    cape.attrs['operator'] = param['cape_operator']
    cape_mean = xr.concat([cape[mem] for mem in mem_list],
                          dim='Member').mean(dim='Member')
    cape['mean'] = cape_mean
    return cape


def open_temp(hour, domain, dx):
    file = '/lustre/scratch/twixtrom/adp_dataset_'+dx+'_temp_2m_f'+str(int(hour))+'.nc'
    temp_2m = xr.open_dataset(file, chunks=chunks)
    temp_2m.attrs['threshold'] = param['temp_2m_threshold']
    temp_2m.attrs['sigma'] = param['sigma']
    temp_2m.attrs['operator'] = param['temp_2m_operator']
    temp_2m_mean = xr.concat([temp_2m[mem] for mem in mem_list],
                             dim='Member').mean(dim='Member')
    temp_2m['mean'] = temp_2m_mean
    return temp_2m


def open_height(hour, domain, dx):
    file = '/lustre/scratch/twixtrom/adp_dataset_'+dx+'_height_500hPa_f'+str(int(hour))+'.nc'
    height_500hPa = xr.open_dataset(file, chunks=chunks)
    height_500hPa.attrs['threshold'] = param['height_500hPa_threshold']
    height_500hPa.attrs['sigma'] = param['sigma']
    height_500hPa.attrs['operator'] = param['height_500hPa_operator']
    height_500hPa_mean = xr.concat([height_500hPa[mem] for mem in mem_list],
                                   dim='Member').mean(dim='Member')
    height_500hPa['mean'] = height_500hPa_mean
    return height_500hPa


dom = float(param['domain'])
if param['method'] == 'rmse_pcpT00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
elif param['method'] == 'rmse_pcpT00+dewptT00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt(param['forecast_hour'], dom, dx)
elif param['method'] == 'rmse_pcpT00+dewptT00+mslpT00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt(param['forecast_hour'], dom, dx)
    mslp = open_mslp(param['forecast_hour'], dom, dx)
elif param['method'] == 'rmse_pcpT00+dewptf00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt(0, dom, dx)
elif param['method'] == 'rmse_pcpT00+dewptf00+mslpf00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt(0, dom, dx)
    mslp = open_mslp(0, dom, dx)
elif param['method'] == 'rmse_pcpT00+capeT-3':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    cape = open_cape((param['forecast_hour'] - 3), dom, dx)
elif param['method'] == 'rmse_pcpT00+dewptT-3':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt((param['forecast_hour'] - 3), dom, dx)
elif param['method'] == 'pcp_area_rmse_pcpT00+dewptT-3':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt((param['forecast_hour'] - 3), dom, dx)
elif param['method'] == 'pcp_area_rmse_pcpT00+dewptT-3+mslpT-3':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt((param['forecast_hour'] - 3), dom, dx)
    mslp = open_mslp((param['forecast_hour'] - 3), dom, dx)
elif param['method'] == 'pcp_area_rmse_pcpT00+dewptf00+mslpf00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    dewpt = open_dewpt(0, dom, dx)
    mslp = open_mslp(0, dom, dx)
elif param['method'] == 'pcp_area_rmse_pcpT00+temp_2mT00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    temp = open_temp(param['forecast_hour'], dom, dx)
elif param['method'] == 'pcp_area_rmse_pcpT00+height_500hPaT00':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    height = open_height(param['forecast_hour'], dom, dx)
elif param['method'] == 'pcp_area_rmse_pcpT00+hgt500f00+capeT-3':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    height = open_height(0, dom, dx)
    cape = open_cape((param['forecast_hour'] - 3), dom, dx)
elif param['method'] == 'rmse_pcpT00+hgt500f00+capeT-3':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    height = open_height(0, dom, dx)
    cape = open_cape((param['forecast_hour'] - 3), dom, dx)
elif param['method'] == 'rmse_pcpT00+capeT-1':
    pcp = open_pcp(param['forecast_hour'], dom, dx)
    cape = open_cape((param['forecast_hour'] - 1), dom, dx)
else:
    raise ValueError('Method not defined')

stage4 = xr.open_dataset(obsfile, chunks=chunks, decode_cf=False)
vtimes_stage4 = num2date(stage4.valid_times, stage4.valid_times.units)
stage4.coords['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4['time'] = np.array([np.datetime64(date) for date in vtimes_stage4])
stage4.coords['lat'] = stage4.lat
stage4.coords['lon'] = stage4.lon

# Create empty analogue output lists and time range to iterate over
an_best_mp = []
an_best_pbl = []
dates = pd.date_range(start=param['an_start_date'],
                      end=param['an_end_date'],
                      freq=param['dt'])

for date in dates:
    print('Starting date '+str(date), flush=True)
    if param['method'] == 'rmse_pcpT00':
        an_idx = find_analogue(date, pcp)
    elif param['method'] == 'rmse_pcpT00+dewptT00':
        an_idx = find_analogue(date, pcp, dewpt)
    elif param['method'] == 'rmse_pcpT00+dewptT00+mslpT00':
        an_idx = find_analogue(date, pcp, dewpt, mslp)
    elif param['method'] == 'rmse_pcpT00+dewptf00':
        an_idx = find_analogue(date, pcp, dewpt)
    elif param['method'] == 'rmse_pcpT00+dewptf00+mslpf00':
        an_idx = find_analogue(date, pcp, dewpt, mslp)
    elif param['method'] == 'rmse_pcpT00+capeT-3':
        an_idx = find_analogue(date, pcp, cape)
    elif param['method'] == 'rmse_pcpT00+capeT-1':
        an_idx = find_analogue(date, pcp, cape)
    elif param['method'] == 'rmse_pcpT00+dewptT-3':
        an_idx = find_analogue(date, pcp, dewpt)
    elif param['method'] == 'pcp_area_rmse_pcpT00+dewptT-3':
        an_idx = find_analogue_precip_area(date, pcp, dewpt)
    elif param['method'] == 'pcp_area_rmse_pcpT00+dewptT-3+mslpT-3':
        an_idx = find_analogue_precip_area(date, pcp, dewpt, mslp)
    elif param['method'] == 'pcp_area_rmse_pcpT00+dewptf00+mslpf00':
        an_idx = find_analogue_precip_area(date, pcp, dewpt, mslp)
    elif param['method'] == 'pcp_area_rmse_pcpT00+temp_2mT00':
        an_idx = find_analogue_precip_area(date, pcp, temp)
    elif param['method'] == 'pcp_area_rmse_pcpT00+height_500hPaT00':
        an_idx = find_analogue_precip_area(date, pcp, height)
    elif param['method'] == 'pcp_area_rmse_pcpT00+hgt500f00+capeT-3':
        an_idx = find_analogue_precip_area(date, pcp, height, cape)
    elif param['method'] == 'rmse_pcpT00+hgt500f00+capeT-3':
        an_idx = find_analogue(date, pcp, height, cape)
    # elif param['method'] == 'rmse_pcpf36+hgt500f00+capef33':
    #     an_idx = find_analogue(date, pcp, height, cape)
    # elif param['method'] == 'rmse_pcpf36+capef33':
    #     an_idx = find_analogue(date, pcp, cape)
    # elif param['method'] == 'rmse_pcpf36+dewptf36':
    #     an_idx = find_analogue(date, pcp, dewpt)
    # elif param['method'] == 'rmse_pcpf36+dewptf00+mslpf00':
    #     an_idx = find_analogue(date, pcp, dewpt, mslp)
    # elif param['method'] == 'rmse_pcpf36+dewptf36+mslpf36':
    #     an_idx = find_analogue(date, pcp, dewpt, mslp)
    # elif param['method'] == 'pcp_area_rmse_pcpf36+dewptf33':
    #     an_idx = find_analogue_precip_area(date, pcp, dewpt)
    else:
        raise ValueError('Method not defined')

    if np.isnan(an_idx):
        an_best_mp.append(('nan', np.nan, date, np.nan))
        an_best_pbl.append(('nan', np.nan, date, np.nan))
    else:
        print('Analogue date selected '+str(pcp.time[an_idx]), flush=True)
        # Get the analogue's verification
        fcst_smooth = gaussian_filter(pcp['mean'].sel(time=date), pcp.attrs['sigma'])
        analogue_time = pcp.time.isel(time=an_idx) + pd.Timedelta(hours=param['forecast_hour'])
        obs = stage4.total_precipitation.sel(
            time=analogue_time,
            drop=True
            )
        obs_smooth = gaussian_filter(obs, param['sigma'])
        st4_an = obs.where((obs_smooth >= param['pcp_threshold']) |
                           (fcst_smooth >= param['pcp_threshold']))

        # Find best member for analouge date by RMSE
        # for MP members
        an_rmse_mp = []
        for mem in mp_list:
            mem_data = pcp[mem][an_idx, ].where((fcst_smooth >= param['pcp_threshold']) |
                                                (obs_smooth >= param['pcp_threshold']))
            rmse_mem = rmse(mem_data, st4_an)
            an_rmse_mp.append(rmse_mem)
        an_mp = np.array(an_rmse_mp)

        # for PBL members
        an_rmse_pbl = []
        for mem in pbl_list:
            mem_data = pcp[mem][an_idx, ].where((fcst_smooth >= param['pcp_threshold']) |
                                                (obs_smooth >= param['pcp_threshold']))
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
        st4_fcst_date = date + pd.Timedelta(hours=param['forecast_hour'])
        st4 = stage4.total_precipitation.sel(time=st4_fcst_date, drop=True)
        st4_smooth = gaussian_filter(st4, verif_param['sigma'])
        st4_verif = stage4.total_precipitation.sel(
            time=st4_fcst_date,
            drop=True
            )
        # For MP members
        verif_mp = []
        for mem in mp_list:
            mem_data = pcp[mem].sel(
                time=date,
                drop=True
                )
            mem_smooth = gaussian_filter(mem_data, verif_param['sigma'])
            error = rmse(mem_data.where((st4_smooth >= verif_param['threshold']) |
                                        (mem_smooth >= verif_param['threshold'])),
                         st4_verif.where((st4_smooth >= verif_param['threshold']) |
                                         (mem_smooth >= verif_param['threshold'])))
            verif_mp.append(error)
        verif_members_mp = np.array(verif_mp)

        # Sort members by performance
        sorted_mp = np.array(mp_list)[np.argsort(verif_members_mp)]

        # get rank of predicted best
        if best_mp == 'nan':
            an_best_mp.append(('nan', np.nan, date, np.nan))
        else:
            rank_mp = np.where(sorted_mp == best_mp)[0][0]
            an_best_mp.append((best_mp, rank_mp, date, pcp.time[an_idx].values))

        # For PBL members
        verif_pbl = []
        for mem in pbl_list:
            mem_data = pcp[mem].sel(
                time=date,
                drop=True
                )
            mem_smooth = gaussian_filter(mem_data, verif_param['sigma'])
            error = rmse(mem_data.where(((st4_smooth >= verif_param['threshold']) |
                                         (mem_smooth >= verif_param['threshold']))),
                         st4_verif.where(((st4_smooth >= verif_param['threshold']) |
                                          (mem_smooth >= verif_param['threshold']))))
            verif_pbl.append(error)
        verif_members_pbl = np.array(verif_pbl)

        # Sort members by performance
        sorted_pbl = np.array(pbl_list)[np.argsort(verif_members_pbl)]

        # get rank of predicted best
        if best_pbl == 'nan':
            an_best_pbl.append(('nan', np.nan, date, np.nan))
        else:
            rank_pbl = np.where(sorted_pbl == best_pbl)[0][0]
            an_best_pbl.append((best_pbl, rank_pbl, date, pcp.time[an_idx].values))

print('Writing output to file', flush=True)
np.save(outname_mp, an_best_mp)
np.save(outname_pbl, an_best_pbl)

print('Analyzing Output', flush=True)
mp_ranks = []
for case in an_best_mp:
    mp_ranks.append(case[1])

pbl_ranks = []
for case in an_best_pbl:
    pbl_ranks.append(case[1])

an_pbl = [case[0] for case in an_best_pbl]
an_mp = [case[0] for case in an_best_mp]

print('Generating Histograms', flush=True)
xlab = 'Member'
ylab = 'Selected Frequency'
ax, ax2 = plot_panels((16, 16), (2, 2), 2,
                      grid=True,
                      x_labels=[xlab],
                      y_labels=[ylab],
                      ylim=(0, 30))
ax.hist(an_mp, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        range=(0, 10), facecolor='green', align='left', rwidth=0.5)
ax.set_title('Histogram of Observed Analogue MP Scheme Selection for ' +
             param['method']+' Method')

ax2.hist(an_pbl, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         range=(0, 10), facecolor='orange', align='left', rwidth=0.3)
ax2.set_title('Histogram of Observed Analogue PBL Scheme Selection'+param['method']+' Method')
# plt.savefig(param['directory']+param['method']+'_'+str(param['threshold'])+'_std' +
#             str(param['sigma'])+'_'+str(param['forecast_hour'])+'_d0'+str(param['domain']) +
#             '_mem_selection.png')
plt.savefig(param['directory']+param['method']+'_d0'+str(int(dom))+'_mem_selection.png')

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
ax.set_title('Histogram of Observed Analogue MP Member Ranking '+param['method']+' Method')
ax.set_xticks(ticks)
ax2.hist(np.array(pbl_ranks) + 1, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         range=(0, 10), facecolor='orange', align='left', rwidth=0.5)
ax2.set_title('Histogram of Observed Analogue PBL Member Ranking'+param['method']+' Method')
ax2.set_xticks(ticks)
# plt.savefig(param['directory']+param['method']+'_'+str(param['threshold'])+'_std' +
#             str(param['sigma'])+'_'+str(param['forecast_hour'])+'_d0'+str(param['domain']) +
#             '_ranks.png')
plt.savefig(param['directory']+param['method']+'_d0'+str(int(dom))+'_ranks.png')

# Find average best member
tot_rmse = verify_members(pcp, stage4.total_precipitation, verif_param, mem_list)
mean_mp = np.array([tot_rmse[mem] for mem in mp_list])
mean_best_mp = mp_list[np.nanargmin(mean_mp)]
mean_pbl = np.array([tot_rmse[mem] for mem in pbl_list])
mean_best_pbl = pbl_list[np.nanargmin(mean_pbl)]
# mean_best_mp = 'mem1'
# mean_best_pbl = 'mem1'

# Compare analogue physics with average best physics
print('Generating Timeseries', flush=True)
rmse_an_pbl = []
rmse_best_pbl = []
an_pbl_mem = [case[0] for case in an_best_pbl]
an_dates = [case[-2] for case in an_best_pbl]
for i in range(len(an_dates)):
    if an_pbl_mem[i] == 'nan':
        rmse_an_pbl.append(np.nan)
        rmse_best_pbl.append(np.nan)
    else:
        date = an_dates[i]
        st4_date = date + pd.Timedelta(hours=param['forecast_hour'])

        # Calculate change in RMSE with point-based verification
        st4 = stage4['total_precipitation'].sel(time=st4_date, drop=True)
        st4_smooth = gaussian_filter(st4, verif_param['sigma'])
        an_pcp = pcp[an_pbl_mem[i]].sel(time=date, drop=True)
        an_smooth = gaussian_filter(an_pcp, verif_param['sigma'])
        best_pcp = pcp[mean_best_pbl].sel(time=date, drop=True)
        best_smooth = gaussian_filter(best_pcp, verif_param['sigma'])
        error_an = rmse(an_pcp.where(((st4_smooth >= verif_param['threshold']) |
                                      (an_smooth >= verif_param['threshold']))),
                        st4.where(((st4_smooth >= verif_param['threshold']) |
                                   (an_smooth >= verif_param['threshold']))))
        rmse_an_pbl.append(error_an)
        error_best = rmse(best_pcp.where(((st4_smooth >= verif_param['threshold']) |
                                          (best_smooth >= verif_param['threshold']))),
                          st4.where(((st4_smooth >= verif_param['threshold']) |
                                     (best_smooth >= verif_param['threshold']))))
        rmse_best_pbl.append(error_best)


# For MP Members
rmse_an_mp = []
rmse_best_mp = []
an_mp_mem = [case[0] for case in an_best_mp]
for i in range(len(an_dates)):
    if an_mp_mem[i] == 'nan':
        rmse_an_mp.append(np.nan)
        rmse_best_mp.append(np.nan)
    else:
        date = an_dates[i]
        st4_date = date + pd.Timedelta(hours=param['forecast_hour'])

        # Calculate change in RMSE with point-based verification
        st4 = stage4['total_precipitation'].sel(time=st4_date, drop=True)
        st4_smooth = gaussian_filter(st4, verif_param['sigma'])
        an_pcp = pcp[an_mp_mem[i]].sel(time=date, drop=True)
        an_smooth = gaussian_filter(an_pcp, verif_param['sigma'])
        best_pcp = pcp[mean_best_mp].sel(time=date, drop=True)
        best_smooth = gaussian_filter(best_pcp, verif_param['sigma'])
        error_an = rmse(an_pcp.where(((st4_smooth >= verif_param['threshold']) |
                                      (an_smooth >= verif_param['threshold']))),
                        st4.where(((st4_smooth >= verif_param['threshold']) |
                                   (an_smooth >= verif_param['threshold']))))
        rmse_an_mp.append(error_an)
        error_best = rmse(best_pcp.where(((st4_smooth >= verif_param['threshold']) |
                                          (best_smooth >= verif_param['threshold']))),
                          st4.where(((st4_smooth >= verif_param['threshold']) |
                                     (best_smooth >= verif_param['threshold']))))
        rmse_best_mp.append(error_best)

diff_rmse_mp = np.array(rmse_an_mp) - np.array(rmse_best_mp)
diff_rmse_pbl = np.array(rmse_an_pbl) - np.array(rmse_best_pbl)

np.save(param['directory']+param['method']+'_rmse_an_mp.npy', rmse_an_mp)
np.save(param['directory']+param['method']+'_rmse_an_pbl.npy', rmse_an_pbl)
np.save(param['directory']+param['method']+'_rmse_best_mp.npy', rmse_best_mp)
np.save(param['directory']+param['method']+'_rmse_best_pbl.npy', rmse_best_pbl)

print('Making Plots', flush=True)
fdates = [case[-2] for case in an_best_pbl]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
plt.plot(fdates, diff_rmse_mp,
         label='Analogue - Best ('+mean_best_mp+')',
         color='tab:green',
         linewidth=0.7)
# plt.plot(fdates, np.array(rmse_best_mp),
#          label='Best MP - '+mean_best_mp,
#          color='tab:green',
#          linewidth=0.7)
plt.title('Analogue and Average Best Hybrid RMSE for MP Members Difference'
          '(Analogue - Best) '+param['method']+' Method')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.axhline(linewidth=4, color='tab:red')
plt.legend(shadow=True, fontsize='large', loc=0)
plt.grid()
# plt.savefig(param['directory']+param['method']+'_'+str(param['threshold'])+'_std' +
#             str(param['sigma'])+'_'+str(param['forecast_hour'])+'_d0'+str(param['domain']) +
#             '_an_vs_best_mp.png')
plt.savefig(param['directory']+param['method']+'_d0'+str(int(dom))+'_an_vs_best_mp.png')

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
plt.plot(fdates, diff_rmse_pbl,
         label='Analogue - Best ('+mean_best_pbl+')',
         color='tab:green',
         linewidth=0.7)
# plt.plot(fdates, np.array(rmse_best_pbl),
#          label='Best PBL - '+mean_best_pbl,
#          color='tab:green',
#          linewidth=0.7)
plt.title('Analogue and Average Best Hybrid RMSE for PBL Members Difference '
          '(Analogue - Best) '+param['method']+' Method')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.axhline(linewidth=4, color='tab:red')
plt.legend(shadow=True, fontsize='large', loc=0)
plt.grid()
# plt.savefig(param['directory']+param['method']+'_'+str(param['threshold'])+'_std' +
#             str(param['sigma'])+'_'+str(param['forecast_hour'])+'_d0'+str(param['domain']) +
#             '_an_vs_best_pbl.png')
plt.savefig(param['directory']+param['method']+'_d0'+str(int(dom))+'_an_vs_best_pbl.png')
print('Analogue Analysis Completed', flush=True)
