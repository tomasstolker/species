"""
Plot module.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FormatStrFormatter

import species.read
import species.photometry


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)

def plot_color_magnitude(color,
                         magnitude,
                         sptype,
                         objects,
                         label_x,
                         label_y,
                         output):
    """
    :param color:
    :type color:

    :return: None
    """


    plt.figure(1, figsize=(4, 4.8))
    gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.5])
    gs.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gs[2, 0])
    ax2 = plt.subplot(gs[0, 0])

    ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4), zorder=0)

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=True)
    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=True)

    ax1.set_xlabel(label_x, fontsize=14)
    ax1.set_ylabel(label_y, fontsize=14)

    ax1.invert_yaxis()

    ax1.get_xaxis().set_label_coords(0.5, -0.08)
    ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    indices = np.where(sptype != "null")[0]

    sptype = sptype[indices]
    color = color[indices]
    magnitude = magnitude[indices]

    spt_disc = np.zeros(color.shape)

    for i, item in enumerate(sptype):
        sp = item[0:2]

        if sp == "M0" or sp == "M1" or sp == "M2" or sp == "M3" or sp == "M4":
            spt_disc[i] = 0

        elif sp == "M5" or sp == "M6" or sp == "M7" or sp == "M8" or sp == "M9":
            spt_disc[i] = 1

        elif sp == "L0" or sp == "L1" or sp == "L2" or sp == "L3" or sp == "L4":
            spt_disc[i] = 2

        elif sp == "L5" or sp == "L6" or sp == "L7" or sp == "L8" or sp == "L9":
            spt_disc[i] = 3

        elif sp == "T0" or sp == "T1" or sp == "T2" or sp == "T3" or sp == "T4":
            spt_disc[i] = 4

        elif sp == "T5" or sp == "T6" or sp == "T7" or sp == "T8" or sp == "T9":
            spt_disc[i] = 5

        elif item[0] == "Y":
            spt_disc[i] = 6

        elif len(item) > 7:
            if item[6] == "Y":
                spt_disc[i] = 7

        else:
            continue

    cmap = plt.cm.viridis
    bounds = np.arange(0, 8, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax1.scatter(color, magnitude, c=spt_disc, cmap=cmap, norm=norm, zorder=3, s=25., alpha=0.6)

    cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal', ticklocation='top', format='%.2f')
    cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='white')
    cb.set_ticks(np.arange(0.5, 7., 1.))
    cb.set_ticklabels(["M0-M4", "M5-M9", "L0-L4", "L5-L9", "T0-T4", "T6-T8", "Y1-Y2"])

    if objects is not None:
        for i, item in enumerate(objects):
            read_object = species.read.ReadObject(item[0])
            color = read_object.get_magnitude(item[1]) - read_object.get_magnitude(item[2])
            mag = read_object.get_magnitude(item[3])

            ax1.plot(color, mag, 's', ms=5, color="black")

    # h,l = ax.get_legend_handles_labels()
    # leg = ax.legend(h, l, loc='upper left', prop={'size':10}, frameon=True, bbox_to_anchor=(0.008, 0.01))
    # for i, text in enumerate(leg.get_texts()):
    #     text.set_color(colors[i])

    plt.savefig(os.getcwd()+"/"+output, bbox_inches='tight')
    plt.close()

def plot_spectrum(wavelength,
                  flux,
                  filters,
                  photometry,
                  output):
    """
    :param color:
    :type color:

    :return: None
    """

    plt.figure(1, figsize=(7, 4))
    gs = mpl.gridspec.GridSpec(1, 1)
    gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = ax1.twinx()

    ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4))

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=False)
    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=False)

    ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=False, right=True)
    ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=False, right=True)

    ax1.set_xlabel("Wavelength [micron]", fontsize=16)
    ax1.set_ylabel("Flux [W m$^{-2}$ micron$^{-1}$]", fontsize=16)
    ax2.set_ylabel("Filter transmission", fontsize=16, rotation=-90)

    ax1.set_xlim(0.7, 6.)
    ax2.set_ylim(0., 1.)

    ax1.get_xaxis().set_label_coords(0.5, -0.09)
    ax1.get_yaxis().set_label_coords(-0.13, 0.5)
    ax2.get_yaxis().set_label_coords(1.1, 0.5)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    if isinstance(wavelength[0], (np.float32, np.float64)):
        data = np.array(flux, dtype=np.float64)
        masked = np.ma.array(data, mask=np.isnan(data))

        ax1.plot(wavelength, masked, lw=1)

    else:
        for i, _ in enumerate(wavelength):
            data = np.array(flux[i], dtype=np.float64)
            masked = np.ma.array(data, mask=np.isnan(data))

            ax1.plot(wavelength[i], masked, lw=1)

    for i, item in enumerate(filters):
        transmission = species.read.ReadFilter(item)
        data = transmission.get_filter()

        ax2.plot(data[0, ], data[1, ], '-', lw=0.5, color="black")

    if photometry is not None:
        for i, item in enumerate(photometry):
            ax1.plot(item[0], item[1], 's', ms=5, color="black")

    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

    plt.savefig(os.getcwd()+"/"+output, bbox_inches='tight')
    plt.close()
