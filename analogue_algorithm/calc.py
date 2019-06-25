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
import dask
from dask.diagnostics import ProgressBar
from scipy.ndimage import gaussian_filter


@dask.delayed
def rmse_dask(predictions, targets, axis=None, nan=False):
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
        rmse_data = np.sqrt(np.nanmean(
            ((predictions - targets) ** 2), axis=axis))
    return rmse_data


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
        rmse_data = np.sqrt(np.nanmean(
            ((predictions - targets) ** 2), axis=axis))
    return rmse_data

# def rmse_xarray(predictions, targets, axis=None, nan=False):
#     return xr.apply_ufunc(rmse, predictions, targets, kwargs={'axis': axis, 'nan': nan},
#                           dask='parallelized', output_dtypes=[float],
#                           output_core_dims=[()], vectorize=True)


def find_analogue(forecast, dataset, mean=False):
    """
        Finds the index value of the closest analogue within the input dataset

        :param forecast: ordered list of forecast variable arrays for a single forecast time
        :param dataset:  list of xarray datasets containing analgoue forecast variables
            corresponding to the forecast variable list. Ensemble mean should be
            named 'mean' in each dataset. One dataset for each variable. Dataset dimensions
            should include initialized time, forecast hour, latitude, and longitude. Global
            attributes for sigma and threshold should also be defined.
        :param mean: bool, if True, calculate ensemble mean and add to dataset list
        :return: index of closest analogue
        """
    score = np.zeros(dataset[0].time.shape[0])
    argscore = []
    for forecast_var, data_var in zip(forecast, dataset):
        # if mean:
        #     data_mean = xr.concat([data_var[mem] for mem in data_var.data_vars.keys()],
        #                           dim='Member').mean(dim='Member')
        #     data_var['mean'] = data_mean
        sigma = data_var.sigma
        threshold = data_var.threshold

        # Smooth and mask forecast mean
        fcst_smooth = xr.apply_ufunc(
            gaussian_filter, forecast_var, sigma, dask='allowed')
        operator = data_var.operator
        fcst_masked = forecast_var.where(
            operator(fcst_smooth, threshold), drop=True)

        # mask the mean, subset for up to current date, find closest analogues by mean RMSE
        dataset_mean = data_var['mean']
        # Use Thompson+YSU for comparison instead of mean
        # dataset_mean = data_var['mem1']
        dataset_mean_masked = dataset_mean.where(
            operator(fcst_smooth, threshold), drop=True)
        # Actually find the index of the closest analogue
        argscore.append(rmse_dask(dataset_mean_masked,
                                  fcst_masked, axis=(-2, -1)))

    with ProgressBar():
        for arg in dask.compute(*argscore):
            if np.isnan(arg).all():
                break
            else:
                score += arg

    try:
        an_idx = np.nanargmin(score)
    except ValueError:
        an_idx = np.nan
    # Return the analogue score too
    try:
        an_score = score[an_idx]
    except ValueError:
        an_score = np.nan
    return an_idx, an_score


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


def find_analogue_precip_area(forecast, dataset):
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
    sigma = dataset[0].sigma
    threshold = dataset[0].threshold
    score = np.zeros(dataset[0].time.shape[0])
    # Smooth and mask forecast mean
    fcst_smooth = xr.apply_ufunc(
        gaussian_filter, forecast[0], sigma, dask='allowed')
    operator = dataset[0].operator
    argscore = []
    for fcst, data in zip(forecast, dataset):
        # Mask based on smoothed precipitation field
        fcst_masked = fcst.where(operator(fcst_smooth, threshold), drop=True)

        # mask the mean, subset for up to current date, find closest analogues by mean RMSE
        dataset_mean = data['mean'].where(
            operator(fcst_smooth, threshold), drop=True)
        # Use Thompson+YSU instead of mean for analogue selection
        # dataset_mean = data['mem1'].where(operator(fcst_smooth, threshold), drop=True)

        # Actually find the index of the closest analogue
        argscore.append(rmse_dask(dataset_mean, fcst_masked, axis=(-2, -1)))

    with ProgressBar():
        for arg in dask.compute(*argscore):
            if np.isnan(arg).all():
                break
            else:
                score += arg

    try:
        an_idx = np.nanargmin(score)
    except ValueError:
        an_idx = np.nan
    # Return the analogue score too
    try:
        an_score = score[an_idx]
    except ValueError:
        an_score = np.nan
    return an_idx #, an_score


def find_max_coverage(data, dim=None):
    """
    Calculates the time of maximum coverage.

    :param data: xarray.dataArray, input dataArray, should have attributes for smoothing
                 and threshold defined.
    :param dim: dimensions to find maximum over
    """
    data_smooth = gaussian_filter(data, data.attrs['sigma'])
    data_masked = data.where(data.operator(data_smooth, data.threshold))
    data_sum = data_masked.sum(dim=dim)
    sum_max = data_sum.max().data
    max_time_idx = data_sum.argmax().data
    max_time = data.time.isel(time=max_time_idx).item()
    return sum_max, pd.Timestamp(max_time)
