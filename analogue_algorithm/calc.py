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
# from datetime import timedelta

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
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


def find_analogue(forecast_date, dataset):
    """
    Finds the index value of the closest analogue within the input dataset

    :param forecast_date: datetime object or numpy.datetime64 object for forecast date.
        Forecast date should be included within dataset.
    :param parameters: dict. Dictionary with project parameters. Should include 'threshold'
        and 'sigma' values.
    :param args: xarray datasets containing analgoue forecasts. Ensemble mean should be
        named 'mean' in each dataset. One dataset for each variable. Dataset dimensions
        should include initialized time, forecast hour, latitude, and longitude.
    :return: index of closest analogue
    """
    sigma = dataset.sigma
    threshold = dataset.threshold

    fcst_mean = dataset['mean'].sel(time=forecast_date, drop=True)

    # Smooth and mask forecast mean
    fcst_smooth = gaussian_filter(fcst_mean, sigma)
    fcst_masked = fcst_mean.where(fcst_smooth >= threshold)

    # mask the mean, subset for up to current date, find closest analogues by mean RMSE
    dataset_mean = dataset['mean'].where(dataset.time < np.datetime64(forecast_date),
                                         drop=True)
    dataset_mean_masked = dataset_mean.where(fcst_smooth >= threshold)

    # Actually find the index of the closest analogue
    try:
        with ProgressBar():
            an_idx = np.nanargmin(rmse(dataset_mean_masked, fcst_masked, axis=(1, 2)))
    except ValueError:
        an_idx = np.nan
    return an_idx, fcst_smooth


def verify_members(dataset, observations, parameters):
    """
    Calculates the sum of RMSE for each dataset member over the specified time range

    :param dataset: xarray dataset. Input forecast dataset
    :param observations: xarray dataset. Observations dataset
    :param parameters: dict. Dictionary of parameter values as below
              forecast_hour: float. Valid forecast hour of forecast dataset
              threshold: float. Threshold for verification masking
              sigma: float. Standard deviation of guassian filter
              start_date: str. Start date of verification period
              end_date: str. End date of verification period
    :return: Sum of RMSE for each member over verification period
    """
    tot_rmse = {}
    mem_list = [mem for mem in dataset.data_vars.keys()]
    if 'mean' in mem_list:
        mem_list.remove('mean')
    for mem in mem_list:
        tot_rmse[mem] = []

    dates = pd.date_range(start=parameters['start_date'],
                          end=parameters['end_date'],
                          freq=parameters['dt'])
    for date in dates:
        obs_date = date + pd.Timedelta(hours=parameters['forecast_hour'])
        obs_data = observations.sel(time=obs_date)
        obs_smooth = gaussian_filter(obs_data, parameters['sigma'])
        for mem in mem_list:
            mem_rmse = tot_rmse[mem]
            mem_data = dataset[mem].sel(time=date).where(obs_smooth >= parameters['threshold'])
            tot_rmse[mem] = mem_rmse + rmse(mem_data.values,
                                            obs_data.where(
                                                obs_smooth >= parameters['threshold']).values
                                            )
    return tot_rmse
