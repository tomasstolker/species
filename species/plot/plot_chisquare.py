"""
Module with functions for making plots.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colorbar import Colorbar

from .. read import read_result


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def plot_chisquare(tag,
                   fix,
                   output):
    """
    :param output:
    :type output: str

    :return: None
    """

    sys.stdout.write('Plotting chi-square map: '+output+'...')
    sys.stdout.flush()

    result = read_result.ReadResult('chi-square', tag)
    points, chisquare = result.get_chisquare(fix)

    valueiter = iter(points.values())

    y_item = next(valueiter)
    x_item = next(valueiter)

    x_grid, y_grid = np.meshgrid(x_item, y_item)

    fig = plt.figure(1, figsize=(4.5, 4))
    gridsp = mpl.gridspec.GridSpec(1, 3, width_ratios=[4., 0.2, 0.3])
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[0, 0])
    ax2 = plt.subplot(gridsp[0, 2])

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in',
                    width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=True)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in',
                    width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=True)

    contours = ax1.contour(x_grid, y_grid, chisquare, 10, colors='white')
    ax1.clabel(contours, inline=True, fontsize=8)

    extent = [np.amin(x_grid), np.amax(x_grid), np.amin(y_grid), np.amax(y_grid)]
    fig = ax1.imshow(chisquare, extent=extent, origin='lower', aspect='auto', cmap='magma')

    cbar = Colorbar(ax=ax2, mappable=fig, orientation='vertical', ticklocation='right')
    cbar.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='white')
    cbar.ax.set_ylabel('Reduced chi-square', rotation=270, labelpad=18, fontsize=12)

    keyiter = iter(points.keys())

    ax1.set_ylabel(next(keyiter), fontsize=14, ha='center', va='top')
    ax1.set_xlabel(next(keyiter), fontsize=14, ha='center', va='bottom')

    ax1.get_xaxis().set_label_coords(0.5, -0.10)
    ax1.get_yaxis().set_label_coords(-0.18, 0.5)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()
