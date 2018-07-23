##############################################################################################
# calc.py - Functions for analogue score calculation
#
# by Tyler Wixtrom
# Texas Tech University
# 20 July 2018
#
# Code for caclulation of analogue score for inclusion in adaptive ensemble forecasts.
#
##############################################################################################
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.ndimage import gaussian_filter


def rmse(predictions, targets, axis=None, nan=False):
    """
    Root Mean Square Error (RMSE)

    Calculate RMSE on grid or timeseries

    :param predictions: array, forecast variable array.
    :param targets: array, observation variable array.
    :param axis: tuple, optional. Axes over which to perform calculation, If none, RMSE
        calculated for entire array shape.
    :param nan: bool, optional. Determines if nan values in inputs should result in
        nan output. Default is to ignore nan points.
    :return: Root Mean Square Error of predictions
    """
    if nan:
        rmse_data = np.sqrt(np.mean(((predictions - targets) ** 2), axis=axis))
    else:
        rmse_data = np.sqrt(np.nanmean(((predictions - targets) ** 2), axis=axis))
    return rmse_data


def find_analogue_rmse(forecast_date, dataset, threshold, sigma):
    """
    Finds the index value of the closest analogue within the input dataset

    :param forecast_date: datetime object or numpy.datetime64 object for forecast date.
        Forecast date should be included within dataset.
    :param dataset: xarray dataset of analgoue forecasts. Ensemble mean should be
        named 'mean' in the dataset.
    :param threshold: float. Threshold for masking smoothed forecast field.
    :param sigma: float. Standard deviation of gaussian filter to use for smoothing.
    :return: index of closest analogue
    """
    fcst_mean = dataset['mean'].sel(time=forecast_date, drop=True)

    # Smooth and mask forecast mean
    fcst_smooth = gaussian_filter(fcst_mean, sigma)
    fcst_masked = fcst_mean.where(fcst_smooth >= threshold)

    # mask the mean, subset for up to current date, find closest analogues by mean RMSE
    dataset_mean = dataset['mean'].where(dataset.time < forecast_date, drop=True)
    dataset_mean_masked = dataset_mean.where(fcst_smooth >= threshold)

    # Actually find the index of the closest analogue
    try:
        with ProgressBar():
            an_idx = np.nanargmin(rmse(dataset_mean_masked, fcst_masked, axis=(1, 2)))
    except ValueError:
        an_idx = np.nan
    return an_idx, fcst_smooth
