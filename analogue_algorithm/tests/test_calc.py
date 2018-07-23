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
from numpy.testing import assert_array_almost_equal
from analogue_algorithm.calc_analogue import rmse


def test_rmse_no_diff():
    """Testing rmse function"""
    data = np.ones((10, 10)) * np.linspace(0, 100, 100).reshape(10, 10)
    obs = data.copy()
    rmse_calc = rmse(data, obs)
    assert_array_almost_equal(rmse_calc, 0.00, 2)


def test_rmse():
    data = np.ones((2, 2)) * np.linspace(0, 5, 4).reshape(2, 2)
    obs = np.ones((2, 2)) * np.linspace(5, 10, 4).reshape(2, 2)
    rmse_calc = rmse(data, obs)
    assert_array_almost_equal(rmse_calc, 0.00, 2)
