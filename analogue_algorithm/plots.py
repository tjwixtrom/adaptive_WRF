##############################################################################################
# plots.py - Functions for analogue analysis plots
#
# by Tyler Wixtrom
# Texas Tech University
# 20 July 2018
#
# Code for analysis plotting of adaptive ensemble forecasts.
#
##############################################################################################

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_panels(figsize, layout, num_axes, grid=True, axis_titles=None, x_labels=None,
                y_labels=None, ylim=None, xlim=None, proj=None):
    """Generate matplotlib axis object for a given figsize and layout
    figsize: tuple
        tuple of the figure size.
    layout: tuple
        layout that should be passed to figure.add_subplot.
    num_axes: int
        number of axes.
    grid: bool
        add a grid background if true (default).
    axis_titles: list
        list of string titles for each axis.
    x_labels, y_labels: list
        list of x and y labels for each axis.
        If only one is given all axes will be labeled the same.
    x_lim, ylim: tuple
        tuple of x-axis or y-axis limits.

    Returns:
        list of axis objects corresponding to each axis in the input layout.

    """
    fig = plt.figure(figsize=figsize)
    ret = []
    for i in range(num_axes):
        ax = fig.add_subplot(layout[0], layout[1], i+1, projection=proj)
        if grid:
            ax.grid()
        if axis_titles is not None:
            ax.set_title(axis_titles[i])
        if x_labels is not None:
            if len(x_labels) > 1:
                ax.set_xlabel(x_labels[i])
            else:
                ax.set_xlabel(x_labels[0])
        if y_labels is not None:
            if len(y_labels) > 1:
                ax.set_ylabel(y_labels[i])
            else:
                ax.set_ylabel(y_labels[0])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        ret.append(ax)
    return ret