##############################################################################################
# calc_analogue.py - Functions for analogue score calculation
#
# by Tyler Wixtrom
# Texas Tech University
# 20 July 2018
#
# Code for caclulation of analogue score for inclusion in adaptive ensemble forecasts.
#
##############################################################################################
import numpy as np


def rmse(predictions, targets, axis=None, nan=False):
    """
    Root Mean Square Error (RMSE)

    Calculate RMSE on grid or timeseries

    Parameters:
        predictions: array, forecast variable array.
        targets: array, observation variable array.
        axis: tuple, optional. Axes over which to perform calculation, If none, RMSE
            calculated for entire array shape.
        nan: bool, optional. Determines if nan values in inputs should result in
            nan output. Default is to ignore nan points.
    Out:
        rmse: Root Mean Square Error of predictions
    """
    if nan:
        rmse_data = np.sqrt(np.mean(((predictions - targets) ** 2), axis=axis))
    else:
        rmse_data = np.sqrt(np.nanmean(((predictions - targets) ** 2), axis=axis))
    return rmse_data
