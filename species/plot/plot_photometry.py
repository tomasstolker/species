"""
Module with functions for making plots.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colorbar import Colorbar

from .. read import read_object


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def sptype_discrete(sptype, shape):
    """
    :param sptype:
    :type sptype:

    :return::
    :rtype: numpy.ndarray
    """

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        sp = item[0:2]

        if sp in (np.string_('M0'), np.string_('M1'), np.string_('M2'), np.string_('M3'), np.string_('M4')):
            spt_disc[i] = 0

        elif sp in (np.string_('M5'), np.string_('M6'), np.string_('M7'), np.string_('M8'), np.string_('M9')):
            spt_disc[i] = 1

        elif sp in (np.string_('L0'), np.string_('L1'), np.string_('L2'), np.string_('L3'), np.string_('L4')):
            spt_disc[i] = 2

        elif sp in (np.string_('L5'), np.string_('L6'), np.string_('L7'), np.string_('L8'), np.string_('L9')):
            spt_disc[i] = 3

        elif sp in (np.string_('T0'), np.string_('T1'), np.string_('T2'), np.string_('T3'), np.string_('T4')):
            spt_disc[i] = 4

        elif sp in (np.string_('T5'), np.string_('T6'), np.string_('T7'), np.string_('T8'), np.string_('T9')):
            spt_disc[i] = 5

        elif np.string_('Y') in item:
            spt_disc[i] = 6

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc

def plot_color_magnitude(color,
                         magnitude,
                         sptype,
                         objects,
                         label_x,
                         label_y,
                         output,
                         xlim=None,
                         ylim=None):
    """
    :param color:
    :type color:

    :return: None
    """

    sys.stdout.write('Plotting color-magnitude diagram: '+output+'... ')
    sys.stdout.flush()

    plt.figure(1, figsize=(4, 4.8))
    gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.5])
    gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[2, 0])
    ax2 = plt.subplot(gridsp[0, 0])

    ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4), zorder=0)

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.set_xlabel(label_x, fontsize=14)
    ax1.set_ylabel(label_y, fontsize=14)

    ax1.invert_yaxis()

    ax1.get_xaxis().set_label_coords(0.5, -0.08)
    ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    if xlim:
        ax1.set_xlim(xlim[0], xlim[1])

    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])

    indices = np.where(sptype != 'None')[0]

    sptype = sptype[indices]
    color = color[indices]
    magnitude = magnitude[indices]

    spt_disc = sptype_discrete(sptype, color.shape)

    cmap = plt.cm.viridis
    bounds = np.arange(0, 8, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax1.scatter(color, magnitude, c=spt_disc, cmap=cmap, norm=norm,
                       zorder=3, s=25., alpha=0.6)

    cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                  ticklocation='top', format='%.2f')

    cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='white')
    cb.set_ticks(np.arange(0.5, 7., 1.))
    cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8', 'Y1-Y2'])

    if objects is not None:
        for item in objects:
            objdata = read_object.ReadObject(item[0])
            color = objdata.get_magnitude(item[1]) - objdata.get_magnitude(item[2])
            mag = objdata.get_magnitude(item[3])

            ax1.plot(color, mag, 's', ms=5, color='black')

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write('[DONE]\n')
    sys.stdout.flush()


def plot_color_color(colors,
                     sptype,
                     objects,
                     label_x,
                     label_y,
                     output):
    """
    :param colors:
    :type colors: tuple(tuple(str, str), tuple(str, str))

    :return: None
    """

    sys.stdout.write('Plotting color-color diagram: '+output+'... ')
    sys.stdout.flush()

    plt.figure(1, figsize=(4, 4.8))
    gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.5])
    gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[2, 0])
    ax2 = plt.subplot(gridsp[0, 0])

    ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4), zorder=0)

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.set_xlabel(label_x, fontsize=14)
    ax1.set_ylabel(label_y, fontsize=14)

    ax1.invert_yaxis()

    ax1.get_xaxis().set_label_coords(0.5, -0.08)
    ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    indices = np.where(sptype != 'None')[0]

    sptype = sptype[indices]
    color = color[indices]
    magnitude = magnitude[indices]

    spt_disc = sptype_discrete(sptype, color.shape)

    cmap = plt.cm.viridis
    bounds = np.arange(0, 8, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax1.scatter(colors[0], colors[1], c=spt_disc, cmap=cmap, norm=norm,
                       zorder=3, s=25., alpha=0.6)

    cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                  ticklocation='top', format='%.2f')

    cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='white')
    cb.set_ticks(np.arange(0.5, 7., 1.))
    cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8', 'Y1-Y2'])

    if objects is not None:
        for item in objects:
            objdata = read_object.ReadObject(item[0])
            color = objdata.get_magnitude(item[1]) - objdata.get_magnitude(item[2])
            mag = objdata.get_magnitude(item[3])

            ax1.plot(color, mag, 's', ms=5, color='black')

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write('[DONE]\n')
    sys.stdout.flush()
