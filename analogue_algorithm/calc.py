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
from datetime import timedelta
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


def verify_members(dataset, observations, forecast_hour, threshold, sigma,
                   start_date, end_date):
    """
    Calculates the sum of RMSE for each dataset member over the specified time range

    :param dataset: input forecast xarray dataset
    :param observations: observations xarray dataset
    :param forecast_hour: valid forecast hour of forecast dataset
    :param threshold: threshold for verification masking
    :param sigma: standard deviation of guassian filter
    :param start_date: start date of verification period
    :param end_date: end date of verification period
    :return: Sum of RMSE for each member over verification period
    """
    tot_rmse = {}
    mem_list = dataset.data_vars.keys()
    print(mem_list)
    if 'mean' in mem_list:
        mem_list.remove('mean')
    for mem in mem_list:
        tot_rmse[mem] = []

    date = start_date
    while date < end_date:
        obs_date = date + timedelta(hours=forecast_hour)
        obs_data = observations.sel(time=obs_date)
        obs_smooth = gaussian_filter(obs_data, sigma)
        for mem in mem_list:
            mem_rmse = tot_rmse[mem]
            mem_data = dataset[mem].sel(time=date).where(obs_smooth >= threshold)
            tot_rmse[mem] = mem_rmse + rmse(mem_data.values,
                                            obs_data.where(obs_smooth >= threshold).values)
        date += timedelta(days=1)
    return tot_rmse
