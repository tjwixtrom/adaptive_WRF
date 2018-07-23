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
from calc.calc import rmse, find_analogue_rmse
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


def test_find_analogue_rmse():
    """tests the find_analogue_rmse function"""
    date = datetime(2016, 1, 1, 12)
    data = np.ones((5, 10, 10)) * np.linspace(0, 50, 500).reshape(5, 10, 10)
    lat = np.linspace(40, 45, 10)
    lon = np.linspace(-105, -100, 10)
    mlon, mlat = np.meshgrid(lon, lat)
    dataset = xr.Dataset({'mean': (['time', 'latitude', 'longitude'], data)},
                                   coords={'time': pd.date_range(start='2015-12-28T12:00:00',
                                                                 end='2016-1-1T12:00:00',
                                                                 periods=5),
                                           'reference_time': pd.Timestamp('2015-01-01'),
                                           'latitude': (['latitude', 'longitude'], mlat),
                                           'longitude': (['latitude', 'longitude'], mlon)})
    an_idx, fcst_smooth = find_analogue_rmse(date, dataset, 10, 5)
    assert_almost_equal(an_idx, 1, 4)
