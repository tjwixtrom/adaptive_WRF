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
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from analogue_algorithm.calc import rmse, find_analogue, verify_members
from datetime import datetime
import xarray as xr
import pandas as pd


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
    # date = datetime(2016, 1, 1, 12)
    # date_array = np.array([datetime(2015, 12, 28, 12), datetime(2015, 12, 29, 12),
    #                        datetime(2015, 12, 30, 12), datetime(2015, 12, 31, 12),
    #                        datetime(2016, 1, 1, 12)])
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
    an_idx, fcst_smooth = find_analogue(date_array[-1], dataset)
    assert_almost_equal(an_idx, 3, 4)


def test_verif_members():
    param = {
        'forecast_hour': 0,
        'threshold': 10,
        'sigma': 2,
        'start_date': '2015-12-28T12:00:00',
        'end_date': '2015-12-29T12:00:00',
        'dt': '1D'
    }
    date_array = pd.date_range(start=param['start_date'],
                               end=param['end_date'],
                               freq=param['dt'])
    data = np.ones((2, 5, 5)) * np.linspace(5, 30, 50).reshape(2, 5, 5)
    lat = np.linspace(40, 45, 5)
    lon = np.linspace(-105, -100, 5)
    mlon, mlat = np.meshgrid(lon, lat)
    dataset = xr.Dataset({'mem1': (['time', 'latitude', 'longitude'], data),
                          'mem2': (['time', 'latitude', 'longitude'], data * 2.),
                          'mean': (['time', 'latitude', 'longitude'], data * 0.5)},
                         coords={'time': date_array,
                                 'lat': (['latitude', 'longitude'], mlat),
                                 'lon': (['latitude', 'longitude'], mlon)})
    obs = xr.Dataset({'total_precipitation': (['time', 'latitude', 'longitude'], data)},
                     coords={'time': date_array,
                             'lat': (['latitude', 'longitude'], mlat),
                             'lon': (['latitude', 'longitude'], mlon)})
    tot_rmse = verify_members(dataset, obs.total_precipitation, param)
    best_mem = np.array([tot_rmse[mem]] for mem in dataset.data_vars.keys()).argmin()
    assert_almost_equal(best_mem, 0, 4)
