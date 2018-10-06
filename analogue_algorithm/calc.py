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
import xarray as xr

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


def find_analogue(forecast_date, *args):
    """
        Finds the index value of the closest analogue within the input dataset

        :param forecast_date: datetime object or numpy.datetime64 object for forecast date.
            Forecast date should be included within dataset.
        :param args: xarray datasets containing analgoue forecasts. Ensemble mean should be
            named 'mean' in each dataset. One dataset for each variable. Dataset dimensions
            should include initialized time, forecast hour, latitude, and longitude. Global
            attributes for sigma and threshold should also be defined.
        :return: index of closest analogue
        """
    score = np.zeros(args[0].time.where(args[0].time < np.datetime64(forecast_date),
                                        drop=True).shape)
    for arg in args:
        if 'mean' not in arg.data_vars.keys():
            fcst_mean = xr.concat([arg[mem] for mem in arg.data_vars.keys()],
                                  dim='Member').mean(dim='Member')
            arg['mean'] = fcst_mean
        sigma = arg.sigma
        threshold = arg.threshold

        fcst_mean = arg['mean'].sel(time=forecast_date, drop=True)

        # Smooth and mask forecast mean
        fcst_smooth = gaussian_filter(fcst_mean, sigma)
        operator = arg.operator
        fcst_masked = fcst_mean.where(operator(fcst_smooth, threshold))

        # mask the mean, subset for up to current date, find closest analogues by mean RMSE
        dataset_mean = arg['mean'].where(arg.time < np.datetime64(forecast_date),
                                         drop=True)
        dataset_mean_masked = dataset_mean.where(operator(fcst_smooth, threshold))

        # Actually find the index of the closest analogue
        score += rmse(dataset_mean_masked, fcst_masked, axis=(1, 2))
    try:
        an_idx = np.nanargmin(score)
    except ValueError:
        an_idx = np.nan
    return an_idx


def verify_members(dataset, observations, parameters, mem_list):
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
    :param mem_list: List of string member names
    :return: Sum of RMSE for each member over verification period

    Note: Calculation is performed for points with precipitation observed in either
          forecast or observed dataset.
    """
    tot_rmse = {}
    for mem in mem_list:
        tot_rmse[mem] = 0.

    dates = pd.date_range(start=parameters['start_date'],
                          end=parameters['end_date'],
                          freq=parameters['dt'])
    for date in dates:
        obs_date = date + pd.Timedelta(hours=parameters['forecast_hour'])
        obs_data = observations.sel(time=obs_date)
        obs_smooth = gaussian_filter(obs_data, parameters['sigma'])
        for mem in mem_list:
            mem_rmse = tot_rmse[mem]
            fcst_smooth = gaussian_filter(dataset[mem].sel(time=date),
                                          parameters['sigma'])
            mem_data = dataset[mem].sel(
                                        time=date
                                        ).where(
                                        ((obs_smooth >= parameters['threshold']) |
                                         (fcst_smooth >= parameters['threshold']))
                                        )
            obs_data_points = obs_data.where(
                           ((obs_smooth >= parameters['threshold']) |
                            (fcst_smooth >= parameters['threshold'])))
            error = rmse(mem_data.values, obs_data_points.values)
            if np.isnan(error):
                error = 0.
            tot_rmse[mem] = mem_rmse + error
    return tot_rmse


def verify_members_grid(dataset, observations, parameters, mem_list):
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
    :param mem_list: List of string member names
    :return: Sum of RMSE for each member over verification period

    Note: Calculation is performed for entire grid
    """
    tot_rmse = {}
    fcst_dates = pd.date_range(start=parameters['start_date'],
                               end=parameters['end_date'],
                               freq=parameters['dt'])

    pcp_sum = dataset.sel(time=fcst_dates).sum(dim=['time'])
    obs_dates = fcst_dates + pd.Timedelta(float(parameters['forecast_hour']),
                                          parameters['dt'])
    obs_sum = observations.sel(time=obs_dates).sum(dim=['time'])
    for mem in mem_list:
        tot_rmse[mem] = rmse(pcp_sum, obs_sum)
    return tot_rmse


def find_analogue_precip_area(forecast_date, precipitation, *args):
    """
        Finds the index value of the closest analogue within the input dataset

        :param forecast_date: datetime object or numpy.datetime64 object for forecast date.
            Forecast date should be included within dataset.
        :param precipitation: xarray dataset for precipitation variable
        :param args: xarray datasets containing analgoue forecasts. Ensemble mean should be
            named 'mean' in each dataset. One dataset for each variable. Dataset dimensions
            should include initialized time, forecast hour, latitude, and longitude. Global
            attributes for sigma and threshold should also be defined.
        :return: index of closest analogue
        """
    if 'mean' not in precipitation.data_vars.keys():
        fcst_precip_mean = xr.concat([precipitation[mem]
                                      for mem in precipitation.data_vars.keys()],
                                     dim='Member').mean(dim='Member')
        precipitation['mean'] = fcst_precip_mean
    sigma = precipitation.sigma
    threshold = precipitation.threshold

    fcst_mean_precipitation = precipitation['mean'].sel(time=forecast_date, drop=True)

    # Smooth and mask forecast mean
    fcst_smooth_precipitation = gaussian_filter(fcst_mean_precipitation, sigma)
    operator = precipitation.operator
    fcst_masked_precipitation = fcst_mean_precipitation.where(
        operator(
            fcst_smooth_precipitation,
            threshold))

    # mask the mean, subset for up to current date, find closest analogues by mean RMSE
    dataset_mean_precipitation = precipitation['mean'].where(
        precipitation.time < np.datetime64(forecast_date), drop=True)
    dataset_mean_masked_precipitation = dataset_mean_precipitation.where(
                                                                operator(
                                                                    fcst_smooth_precipitation,
                                                                    threshold))

    # Actually find the index of the closest analogue
    score = rmse(dataset_mean_masked_precipitation, fcst_masked_precipitation, axis=(1, 2))

    for arg in args:
        if 'mean' not in arg.data_vars.keys():
            fcst_mean = xr.concat([arg[mem] for mem in arg.data_vars.keys()],
                                  dim='Member').mean(dim='Member')
            arg['mean'] = fcst_mean

        fcst_mean = arg['mean'].sel(time=forecast_date, drop=True)

        # Mask based on smoothed precipitation field
        fcst_masked = fcst_mean.where(operator(fcst_smooth_precipitation, threshold))

        # mask the mean, subset for up to current date, find closest analogues by mean RMSE
        dataset_mean = arg['mean'].where(arg.time < np.datetime64(forecast_date),
                                         drop=True)
        dataset_mean_masked = dataset_mean.where(operator(fcst_smooth_precipitation,
                                                          threshold))

        # Actually find the index of the closest analogue
        score += rmse(dataset_mean_masked, fcst_masked, axis=(1, 2))
    try:
        an_idx = np.nanargmin(score)
    except ValueError:
        an_idx = np.nan
    return an_idx
