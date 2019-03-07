'''
Module with functions for making plots.
'''

import os
import sys
import math
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colorbar import Colorbar

from species.plot import util
from species.read import read_object


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)

marker = itertools.cycle(('o', 's', 'p', '<', '>', 'p', 'v', '^', '*', 'd', 'x', '+', '1', '2', '3', '4'))

def plot_color_magnitude(colorbox,
                         objects,
                         label_x,
                         label_y,
                         output,
                         xlim=None,
                         ylim=None,
                         offset=None,
                         legend='top left'):
    '''
    :param colorbox: Box with the colors and magnitudes.
    :type colorbox: species.core.box.ColorMagBox
    :param objects: Tuple with individual objects. The objects require a tuple with their database
                    tag, the two filter IDs for the color, and the filter ID for the absolute
                    magnitude.
    :type objects: tuple(tuple(str, str, str, str), )
    :param label_x: Label for the x-axis.
    :type label_x: str
    :param label_y: Label for the y-axis.
    :type label_y: str
    :param output: Output filename.
    :type output: str
    :param xlim: Limits for the x-axis.
    :type xlim: tuple(float, float)
    :param ylim: Limits for the y-axis.
    :type ylim: tuple(float, float)
    :param legend: Legend position.
    :type legend: str

    :return: None
    '''

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

    if offset:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.08)
        ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    if xlim:
        ax1.set_xlim(xlim[0], xlim[1])

    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])

    cmap = plt.cm.viridis
    bounds = np.arange(0, 8, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    sptype = colorbox.sptype
    color = colorbox.color
    magnitude = colorbox.magnitude

    indices = np.where(sptype != 'None')[0]

    sptype = sptype[indices]
    color = color[indices]
    magnitude = magnitude[indices]

    spt_disc = util.sptype_discrete(sptype, color.shape)

    _, unique = np.unique(color, return_index=True)

    sptype = sptype[unique]
    color = color[unique]
    magnitude = magnitude[unique]
    spt_disc = spt_disc[unique]

    scat = ax1.scatter(color, magnitude, c=spt_disc, cmap=cmap, norm=norm,
                       zorder=1, s=25., alpha=0.6)

    cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                  ticklocation='top', format='%.2f')

    cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='white')
    cb.set_ticks(np.arange(0.5, 7., 1.))
    cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8', 'Y1-Y2'])

    if objects is not None:
        for item in objects:
            objdata = read_object.ReadObject(item[0])

            objcolor1 = objdata.get_photometry(item[1])
            objcolor2 = objdata.get_photometry(item[2])
            abs_mag = objdata.get_absmag(item[3])

            colorerr = math.sqrt(objcolor1[1]**2+objcolor2[1]**2)

            ax1.errorbar(objcolor1[0]-objcolor2[0], abs_mag[0], yerr=abs_mag[1], xerr=colorerr,
                         marker=next(marker), ms=6, color='black', label=objdata.object_name,
                         markerfacecolor='white', markeredgecolor='black', zorder=2)

    handles, labels = ax1.get_legend_handles_labels()

    if handles:
        handles = [h[0] for h in handles]
        ax1.legend(handles, labels, loc=legend, prop={'size':9}, frameon=False, numpoints=1)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write('[DONE]\n')
    sys.stdout.flush()


def plot_color_color(colorbox,
                     objects,
                     label_x,
                     label_y,
                     output,
                     xlim=None,
                     ylim=None,
                     offset=None,
                     legend='top left'):
    '''
    :param colorbox: Box with the colors and magnitudes.
    :type colorbox: species.core.box.ColorMagBox
    :param objects: Tuple with individual objects. The objects require a tuple with their database
                    tag, the two filter IDs for the color, and the filter ID for the absolute
                    magnitude.
    :type objects: tuple(tuple(str, str, str, str), )
    :param label_x: Label for the x-axis.
    :type label_x: str
    :param label_y: Label for the y-axis.
    :type label_y: str
    :param output: Output filename.
    :type output: str
    :param xlim: Limits for the x-axis.
    :type xlim: tuple(float, float)
    :param ylim: Limits for the y-axis.
    :type ylim: tuple(float, float)
    :param offset: Offset of the x- and y-axis label.
    :type offset: tuple(float, float)
    :param legend: Legend position.
    :type legend: str

    :return: None
    '''

    sys.stdout.write('Plotting color-color diagram: '+output+'... ')
    sys.stdout.flush()

    plt.figure(1, figsize=(4, 4.3))
    gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.])
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

    if offset:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.08)
        ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    if xlim:
        ax1.set_xlim(xlim[0], xlim[1])

    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])

    cmap = plt.cm.viridis
    bounds = np.arange(0, 8, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    sptype = colorbox.sptype
    color1 = colorbox.color1
    color2 = colorbox.color2

    indices = np.where(sptype != 'None')[0]

    sptype = sptype[indices]
    color1 = color1[indices]
    color2 = color2[indices]

    spt_disc = util.sptype_discrete(sptype, color1.shape)

    _, unique = np.unique(color1, return_index=True)

    sptype = sptype[unique]
    color1 = color1[unique]
    color2 = color2[unique]
    spt_disc = spt_disc[unique]

    scat = ax1.scatter(color1, color2, c=spt_disc, cmap=cmap, norm=norm,
                       zorder=1, s=25., alpha=0.6)

    cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                  ticklocation='top', format='%.2f')

    cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='white')
    cb.set_ticks(np.arange(0.5, 7., 1.))
    cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8', 'Y1-Y2'])

    if objects is not None:
        for item in objects:
            objdata = read_object.ReadObject(item[0])

            mag1 = objdata.get_photometry(item[1][0])[0]
            mag2 = objdata.get_photometry(item[1][1])[0]
            mag3 = objdata.get_photometry(item[2][0])[0]
            mag4 = objdata.get_photometry(item[2][1])[0]

            err1 = objdata.get_photometry(item[1][0])[1]
            err2 = objdata.get_photometry(item[1][1])[1]
            err3 = objdata.get_photometry(item[2][0])[1]
            err4 = objdata.get_photometry(item[2][1])[1]

            color1 = mag1 - mag2
            color2 = mag3 - mag4

            error1 = math.sqrt(err1**2+err2**2)
            error2 = math.sqrt(err3**2+err4**2)

            ax1.errorbar(color1, color2, xerr=error1, yerr=error2,
                         marker=next(marker), ms=6, color='black', label=objdata.object_name,
                         markerfacecolor='white', markeredgecolor='black', zorder=2)

    handles, labels = ax1.get_legend_handles_labels()

    if handles:
        handles = [h[0] for h in handles]
        ax1.legend(handles, labels, loc=legend, prop={'size':9}, frameon=False, numpoints=1)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write('[DONE]\n')
    sys.stdout.flush()
