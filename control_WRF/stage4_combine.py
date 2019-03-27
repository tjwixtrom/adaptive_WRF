#!/home/twixtrom/miniconda3/envs/analogue/bin/python
#$ -r y
#$ -V
#$ -cwd
#$ -S /home/twixtrom/miniconda3/envs/analogue/bin/python
#$ -N stage4_combine
#$ -o submit_out
#$ -e error_out
#$ -P communitycluster
#$ -q ancellcc

# Script for combining StageIV precipitation files and regridding to project grid.
#
# by Tyler Wixtrom
# Texas Tech University
# 18 June 2018

import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import pygrib
from pyresample import geometry, image


date1 = datetime(2016, 7, 1, 15)
ref_date = datetime(2016, 1, 1, 0)
enddate = datetime(2016, 8, 2, 12)
outname_03h = '/lustre/scratch/twixtrom/ST4_201607_03h.nc'
outname_01h = '/lustre/scratch/twixtrom/ST4_201607_01h.nc'
dtype = 'f4'

print('Getting Obs Grid')
# Open a stageIV file and get out the grid
grib = pygrib.open('/lustre/scratch/twixtrom/stage4/ST4.2016010112.01h')
apcp = grib.read(1)[0]
obs_lat, obs_lon = apcp.latlons()
obs_grid = geometry.GridDefinition(lons=obs_lon, lats=obs_lat)

print('Getting forecast grids')
# Get the 12-km forecast grid
fcst_12km = Dataset('/lustre/scratch/twixtrom/adaptive_wrf_post/control_thompson/2016010112/wrfprst_d01_2016010112.nc')
fcst_lat_12km = fcst_12km.variables['lat'][0,]
fcst_lon_12km = fcst_12km.variables['lon'][0,]
fcst_grid_12km = geometry.GridDefinition(lons=fcst_lon_12km, lats=fcst_lat_12km)

# Get the 4-km forecast grid
fcst_4km = Dataset('/lustre/scratch/twixtrom/adaptive_wrf_post/control_thompson/2016010112/wrfprst_d02_2016010112.nc')
fcst_lat_4km = fcst_4km.variables['lat'][0,]
fcst_lon_4km = fcst_4km.variables['lon'][0,]
fcst_grid_4km = geometry.GridDefinition(lons=fcst_lon_4km, lats=fcst_lat_4km)


print('Creating output files')
# Create the output files and dimensions
outfile1 = Dataset(outname_01h, 'w')
outfile2 = Dataset(outname_03h, 'w')

outfile2.description = '3-hourly Total Observed Precipitation from NCEP StageIV Precipitation \
                       Analysis regridded to WRF-ARW grid.'
outfile2.creator = 'Tyler Wixtrom'

outfile1.description = '1-hourly Total Observed Precipitation from NCEP StageIV Precipitation \
                       Analysis regridded to WRF-ARW grid.'
outfile1.creator = 'Tyler Wixtrom'

outfile1.createDimension('time', None)
outfile1.createDimension('y', fcst_4km.dimensions['y'].size)
outfile1.createDimension('x', fcst_4km.dimensions['x'].size)

outfile2.createDimension('time', None)
outfile2.createDimension('y', fcst_12km.dimensions['y'].size)
outfile2.createDimension('x', fcst_12km.dimensions['x'].size)

# write lat and lon arrays to file
# 4-km grid
latitude_4km = outfile1.createVariable('latitude', dtype, ('y', 'x'),
                                      zlib=True, complevel=4)
latitude_4km.units = fcst_4km.variables['lat'].units
latitude_4km.description = fcst_4km.variables['lat'].description
latitude_4km[:] = fcst_lat_4km

longitude_4km = outfile1.createVariable('longitude', dtype, ('y', 'x'),
                                      zlib=True, complevel=4)
longitude_4km.units = fcst_4km.variables['lon'].units
longitude_4km.description = fcst_4km.variables['lon'].description
longitude_4km[:] = fcst_lon_4km

# 12-km grid
latitude_12km = outfile2.createVariable('latitude', dtype, ('y', 'x'),
                                      zlib=True, complevel=4)
latitude_12km.units = fcst_12km.variables['lat'].units
latitude_12km.description = fcst_12km.variables['lat'].description
latitude_12km[:] = fcst_lat_12km

longitude_12km = outfile2.createVariable('longitude', dtype, ('y', 'x'),
                                      zlib=True, complevel=4)
longitude_12km.units = fcst_12km.variables['lon'].units
longitude_12km.description = fcst_12km.variables['lon'].description
longitude_12km[:] = fcst_lon_12km

# Create the total precipitation variable
total_precipitation = outfile2.createVariable('precipitation', dtype,
                                              ('time', 'y', 'x'),
                                              zlib=True, complevel=4)
total_precipitation.units = 'millimeter'

time_unit = 'seconds since '+str(ref_date)
# Create the time variable for the 3-hr precipitation
time_03h = outfile2.createVariable('time', dtype, ('time',),
                                              zlib=True, complevel=4)
time_03h.units = time_unit

# Create the hourly precipitation variable
hourly_precipitation = outfile1.createVariable('precipitation', dtype,
                                               ('time', 'y', 'x'),
                                              zlib=True, complevel=4)
hourly_precipitation.units = 'millimeter'

# Create the time variable for the 1-hr precipitation
time_01h = outfile1.createVariable('time', dtype, ('time',),
                                              zlib=True, complevel=4)
time_01h.units = time_unit


# Get the observed precipitation for each day, regrid to forecast grid and save
j = 0
k = 0
while date1 <= enddate:
    print('Processing Time: '+str(date1))
    # open the thre 1-hour accumulation files
    time1 = date1 - timedelta(hours=2)
    file1 = '/lustre/scratch/twixtrom/stage4/ST4.'+time1.strftime('%Y%m%d%H')+'.01h'

    time2 = date1 - timedelta(hours=1)
    file2 = '/lustre/scratch/twixtrom/stage4/ST4.'+time2.strftime('%Y%m%d%H')+'.01h'

    time3 = date1
    file3 = '/lustre/scratch/twixtrom/stage4/ST4.'+time3.strftime('%Y%m%d%H')+'.01h'

    # open the files and extract data
    print('Opening file: '+file1)
    try:
        grib1 = pygrib.open(file1)
        apcp1 = grib1.read(1)[0]
        pcp1 = apcp1.values.data
        pcp1[apcp1.values.mask] = 0.
    except OSError:
        print('File not found: '+file1)
        print('Setting nan field')
        pcp1 = np.full_like(obs_lat, np.nan)

    print('Opening file: '+file2)
    try:
        grib2 = pygrib.open(file2)
        apcp2 = grib2.read(1)[0]
        pcp2 = apcp2.values.data
        pcp2[apcp2.values.mask] = 0.
    except OSError:
        print('File not found: '+file2)
        print('Setting nan field')
        pcp2 = np.full_like(obs_lat, np.nan)

    print('Opening file: '+file3)
    try:
        grib3 = pygrib.open(file3)
        apcp3 = grib3.read(1)[0]
        pcp3 = apcp3.values.data
        pcp3[apcp3.values.mask] = 0.
    except OSError:
        print('File not found: '+file3)
        print('Setting nan field')
        pcp3 = np.full_like(obs_lat, np.nan)


    # sum for 3-hour total
    tot_pcp = pcp1 + pcp2 + pcp3

    # regrid 3-hour total to forecast grid
    print('Regridding 3-hr total precipitation')
    resample_quick = image.ImageContainerNearest(tot_pcp, obs_grid, radius_of_influence=12000)
    resample_obs = resample_quick.resample(fcst_grid_12km)
    new_tot_pcp = resample_obs.image_data

    # write the 3-hour total to file
    print('Writing to file: '+outname_03h)
    total_precipitation[k, :, :] = new_tot_pcp

    # write the time for the 3-hour precipitation
    time_03h[k,] = date2num(date1, time_unit)


    pcp_obj = [pcp1, pcp2, pcp3]
    time_obj = [time1, time2, time3]

    # regrib and write the hourly totals to file
    print('Regridding 1-hr precipitation')
    for i in range(3):
        # regrid 1-hour total to forecast grid
        resample_quick = image.ImageContainerNearest(pcp_obj[i], obs_grid,
                                                     radius_of_influence=4000)
        resample_obs = resample_quick.resample(fcst_grid_4km)
        new_pcp = resample_obs.image_data

        # write the 1-hour total to file
        hourly_precipitation[j, :, :] = new_pcp

        # write the time for the 3-hour precipitation
        time_01h[j,] = date2num(time_obj[i], time_unit)
        j+= 1

    k += 1
    date1 += timedelta(hours=3)
    print('Writing to file: '+outname_01h)
outfile1.close()
outfile2.close()
