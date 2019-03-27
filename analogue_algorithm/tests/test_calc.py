##############################################################################################
# test_calc.py - Functions for testing analogue score calculation
#
# by Tyler Wixtrom
# Texas Tech University
# 23 July 2018
#
# Code for testing the caclulation of analogue score for inclusion in adaptive ensemble
# forecasts.
#
##############################################################################################
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_equal
from analogue_algorithm.calc import rmse, find_analogue, verify_members, find_max_coverage, \
    find_analogue_precip_area
import xarray as xr
import pandas as pd
import operator
import pytest


def test_rmse_no_diff():
    """Testing rmse function"""
    data = np.ones((10, 10)) * np.linspace(0, 100, 100).reshape(10, 10)
    obs = data.copy()
    rmse_calc = rmse(data, obs)
    assert_array_almost_equal(rmse_calc, 0.00, 2)


def test_rmse():
    """Test RMSE for different arrays"""
    data = np.ones((2, 2)) * np.linspace(0, 5, 4).reshape(2, 2)
    obs = np.ones((2, 2)) * np.linspace(5, 10, 4).reshape(2, 2)
    rmse_calc = rmse(data, obs)
    assert_array_almost_equal(rmse_calc, 5.00000, 5)


def test_rmse_axis():
    """Test RMSE with axis argument"""
    data = np.ones((2, 10, 10)) * np.linspace(0, 100, 200).reshape(2, 10, 10)
    obs = np.ones((2, 10, 10)) * np.linspace(50, 150, 200).reshape(2, 10, 10)
    rmse_calc = rmse(data, obs, axis=(1, 2))
    assert_array_almost_equal(rmse_calc, np.array([50.000, 50.000]), 3)


def test_find_analogue():
    """tests the find_analogue function"""
    date_array = pd.date_range(start='2015-12-28T12:00:00',
                               end='2016-01-01T12:00:00',
                               freq='1D')
    data = np.ones((5, 10, 10)) * np.linspace(0, 50, 500).reshape(5, 10, 10)
    lat = np.linspace(40, 45, 10)
    lon = np.linspace(-105, -100, 10)
    mlon, mlat = np.meshgrid(lon, lat)
    dataset = xr.Dataset({'mean': (['time', 'latitude', 'longitude'], data)},
                         coords={'time': date_array,
                                 'lat': (['latitude', 'longitude'], mlat),
                                 'lon': (['latitude', 'longitude'], mlon)})
    dataset.attrs['threshold'] = 10
    dataset.attrs['sigma'] = 5
    dataset.attrs['operator'] = operator.gt
    fdata = dataset['mean'].sel(time=date_array[-1])
    print(date_array)
    an_idx = find_analogue([fdata], [dataset])
    assert_almost_equal(an_idx, 4, 4)


def test_verif_members():
    param = {
        'forecast_hour': 0,
        'threshold': 10,
        'sigma': 2,
        'start_date': '2015-12-27T12:00:00',
        'end_date': '2015-12-29T12:00:00',
        'dt': '1D'
        }
    date_array = pd.date_range(start=param['start_date'],
                               end=param['end_date'],
                               freq=param['dt'])
    data = np.ones((3, 5, 5)) * np.linspace(0, 30, 75).reshape(3, 5, 5)
    data[1, ] = 0.
    mem1_data = data * 10.
    mem3_data = data * 5.
    lat = np.linspace(40, 45, 5)
    lon = np.linspace(-105, -100, 5)
    mlon, mlat = np.meshgrid(lon, lat)
    mem_list = ['mem1', 'mem2', 'mem3', 'mem4']
    dataset = xr.Dataset({'mem1': (['time', 'latitude', 'longitude'], mem1_data),
                          'mem2': (['time', 'latitude', 'longitude'], data),
                          'mem3': (['time', 'latitude', 'longitude'], mem3_data),
                          'mem4': (['time', 'latitude', 'longitude'], data * 15.),
                          'mean': (['time', 'latitude', 'longitude'], data * 0.5)},
                         coords={'time': date_array,
                                 'lat': (['latitude', 'longitude'], mlat),
                                 'lon': (['latitude', 'longitude'], mlon)})
    obs = xr.Dataset({'total_precipitation': (['time', 'latitude', 'longitude'], data)},
                     coords={'time': date_array,
                             'lat': (['latitude', 'longitude'], mlat),
                             'lon': (['latitude', 'longitude'], mlon)})
    with pytest.warns(RuntimeWarning):
        tot_rmse = verify_members(dataset, obs.total_precipitation, param, mem_list)
    members = np.array([tot_rmse[mem] for mem in mem_list])
    best_mem = np.nanargmin(members)
    assert_almost_equal(best_mem, 1, 4)


def test_find_analogue_multi_vars():
    """tests the find_analogue function with multiple inputs"""
    date_array = pd.date_range(start='2015-12-28T12:00:00',
                               end='2016-01-01T12:00:00',
                               freq='1D')
    data = np.ones((5, 10, 10)) * np.linspace(0, 50, 500).reshape(5, 10, 10)
    lat = np.linspace(40, 45, 10)
    lon = np.linspace(-105, -100, 10)
    mlon, mlat = np.meshgrid(lon, lat)
    dataset = xr.Dataset({'mean': (['time', 'latitude', 'longitude'], data)},
                         coords={'time': date_array,
                                 'lat': (['latitude', 'longitude'], mlat),
                                 'lon': (['latitude', 'longitude'], mlon)})
    dataset.attrs['threshold'] = 10
    dataset.attrs['sigma'] = 5
    dataset.attrs['operator'] = operator.gt

    dataset2 = xr.Dataset({'mem1': (['time', 'latitude', 'longitude'], data * 0.5),
                           'mean': (['time', 'latitude', 'longitude'], data * 2)},
                          coords={'time': date_array,
                                  'lat': (['latitude', 'longitude'], mlat),
                                  'lon': (['latitude', 'longitude'], mlon)})
    dataset2.attrs['threshold'] = 8
    dataset2.attrs['sigma'] = 5
    dataset2.attrs['operator'] = operator.gt
    fdata1 = dataset['mean'].sel(time=date_array[-1])
    fdata2 = dataset2['mean'].sel(time=date_array[-1])
    an_idx = find_analogue([fdata1, fdata2], [dataset, dataset2])
    assert_almost_equal(an_idx, 4, 4)


def test_find_max_coverage():
    """Tests the find_analogue function"""
    date_array = pd.date_range(start='2015-12-28T12:00:00',
                               end='2016-01-01T12:00:00',
                               freq='1D')
    data = np.ones((5, 10, 10)) * np.linspace(0, 50, 500).reshape(5, 10, 10)
    lat = np.linspace(40, 45, 10)
    lon = np.linspace(-105, -100, 10)
    mlon, mlat = np.meshgrid(lon, lat)
    dataset = xr.DataArray(data=data,
                           coords={'time': date_array,
                                   'lat': (['latitude', 'longitude'], mlat),
                                   'lon': (['latitude', 'longitude'], mlon)},
                           dims=['time', 'latitude', 'longitude'])

    dataset.attrs['threshold'] = 10
    dataset.attrs['sigma'] = 1
    dataset.attrs['operator'] = operator.ge
    sum_max, max_time = find_max_coverage(dataset, dim=['latitude', 'longitude'])
    assert_almost_equal(sum_max, 4504.00801, 4)
    assert_equal(max_time, pd.to_datetime('2016-01-01T12:00:00'))


def test_find_analogue_precip_area():
    """tests the find_analogue function with multiple inputs"""
    date_array = pd.date_range(start='2015-12-28T12:00:00',
                               end='2016-01-01T12:00:00',
                               freq='1D')
    data = np.ones((5, 10, 10)) * np.linspace(0, 50, 500).reshape(5, 10, 10)
    lat = np.linspace(40, 45, 10)
    lon = np.linspace(-105, -100, 10)
    mlon, mlat = np.meshgrid(lon, lat)
    dataset = xr.Dataset({'mean': (['time', 'latitude', 'longitude'], data)},
                         coords={'time': date_array,
                                 'lat': (['latitude', 'longitude'], mlat),
                                 'lon': (['latitude', 'longitude'], mlon)})
    dataset.attrs['threshold'] = 10
    dataset.attrs['sigma'] = 5
    dataset.attrs['operator'] = operator.gt

    dataset2 = xr.Dataset({'mem1': (['time', 'latitude', 'longitude'], data * 0.5),
                           'mean': (['time', 'latitude', 'longitude'], data * 2)},
                          coords={'time': date_array,
                                  'lat': (['latitude', 'longitude'], mlat),
                                  'lon': (['latitude', 'longitude'], mlon)})
    dataset2.attrs['threshold'] = 8
    dataset2.attrs['sigma'] = 5
    dataset2.attrs['operator'] = operator.gt
    fdata1 = dataset['mean'].sel(time=date_array[-1])
    fdata2 = dataset2['mean'].sel(time=date_array[-1])
    an_idx = find_analogue_precip_area([fdata1, fdata2], [dataset, dataset2])
    assert_almost_equal(an_idx, 4, 4)
