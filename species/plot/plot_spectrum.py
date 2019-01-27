"""
Module with functions for making plots.
"""

import os
import sys
import math
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from . util import quantity_unit, model_name
from .. core import box
from .. read import read_filter


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def plot_spectrum(boxes,
                  filters,
                  output,
                  colors,
                  residuals=None,
                  xlim=None,
                  ylim=None,
                  scale=('log', 'log'),
                  title=None,
                  offset=None):
    """
    :param boxes:
    :type boxes: tuple(species.analysis.box.SpectrumBox and/or
                 species.analysis.box.PhotometryBox and/or
                 species.analysis.box.ModelBox)
    :param filters:
    :type filters: tuple(str, )
    :param output:
    :type output: str

    :return: None
    """

    marker = itertools.cycle(('o', 's', '*', 'p', '<', '>', 'P', 'v', '^'))

    if residuals:
        plt.figure(1, figsize=(7, 5))
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    else:
        plt.figure(1, figsize=(7, 4))
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[0, 0])

    if filters:
        ax2 = ax1.twinx()

    ax1.grid(True, linestyle=':', linewidth=0.7, color='gray', dashes=(1, 4), alpha=0.3, zorder=0)

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=False)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=False)

    if filters:
        ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=5, labelsize=12, top=True,
                        bottom=True, left=False, right=True)

        ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=3, labelsize=12, top=True,
                        bottom=True, left=False, right=True)

    if residuals:
        ax3 = plt.subplot(gridsp[1, 0])

        ax3.grid(True, linestyle=':', linewidth=0.7, color='gray', dashes=(1, 4),
                 alpha=0.3, zorder=0)

        ax3.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=5, labelsize=12, top=True,
                        bottom=True, left=False, right=True)

        ax3.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=3, labelsize=12, top=True,
                        bottom=True, left=False, right=True)

    if residuals:
        ax1.set_xlabel('', fontsize=16)
        ax3.set_xlabel('Wavelength [micron]', fontsize=15)

    else:
        ax1.set_xlabel('Wavelength [micron]', fontsize=15)

    ax1.set_ylabel('Flux [W m$^{-2}$ $\mu$m$^{-1}$]', fontsize=15)

    if filters:
        ax2.set_ylabel('Filter transmission', fontsize=15, rotation=-90)

    if xlim:
        ax1.set_xlim(xlim[0], xlim[1])
    else:
        ax1.set_xlim(0.6, 6.)

    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])

    if filters:
        ax2.set_ylim(0., 1.)

    if residuals:
        xlim = ax1.get_xlim()
        ax3.set_xlim(xlim[0], xlim[1])

    ylim = ax1.get_ylim()

    if ylim[0] < 0.:
        ax1.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=1)

    if offset:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.12)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)

    if filters:
        ax2.get_yaxis().set_label_coords(1.1, 0.5)

    ax1.set_xscale(scale[0])
    ax1.set_yscale(scale[1])

    for j, boxitem in enumerate(boxes):

        if isinstance(boxitem, (box.SpectrumBox, box.ModelBox)):
            wavelength = boxitem.wavelength
            flux = boxitem.flux

            if isinstance(wavelength[0], (np.float32, np.float64)):
                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                if isinstance(boxitem, box.ModelBox):
                    par_key, par_unit = quantity_unit(list(boxitem.par_key))
                    par_val = list(boxitem.par_val)[:-1]

                    name = model_name(boxitem.model)
                    label = name+': '

                    for i, item in enumerate(par_key):
                        label += item+' = '+str(par_val[i])+' '+par_unit[i]

                        if i < len(par_key)-1:
                            label += ', '

                else:
                    label = None

                ax1.plot(wavelength, masked, color=colors[j], lw=1.0, label=label, zorder=4)

            if isinstance(wavelength[0], (np.ndarray)):
                for i, item in enumerate(wavelength):
                    data = np.array(flux[i], dtype=np.float64)
                    masked = np.ma.array(data, mask=np.isnan(data))

                    ax1.plot(item, masked, lw=1)

        elif isinstance(boxitem, tuple):
            for i, item in enumerate(boxitem):
                wavelength = item.wavelength
                flux = item.flux

                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                ax1.plot(wavelength, masked, lw=0.4, color=colors[j], alpha=0.5, zorder=3)

        elif isinstance(boxitem, box.PhotometryBox):
            ax1.plot(boxitem.wavelength, boxitem.flux, marker=next(marker), ms=6, \
                     color=colors[j], label=boxitem.name)

        elif isinstance(boxitem, box.ObjectBox):
            for item in boxitem.flux:
                transmission = read_filter.ReadFilter(item)
                wavelength = transmission.mean_wavelength()
                fwhm = transmission.filter_fwhm()

                ax1.errorbar(wavelength, boxitem.flux[item][0], xerr=fwhm/2.,
                             yerr=boxitem.flux[item][1], marker='s', ms=5, zorder=5,
                             color=colors[j], markerfacecolor=colors[j])

        elif isinstance(boxitem, box.SynphotBox):
            for item in boxitem.flux:
                transmission = read_filter.ReadFilter(item)
                wavelength = transmission.mean_wavelength()
                fwhm = transmission.filter_fwhm()

                ax1.errorbar(wavelength, boxitem.flux[item], xerr=fwhm/2., yerr=None, alpha=0.7,
                             marker='s', ms=5, zorder=6, color=colors[j], markerfacecolor='white')

    if filters:
        for i, item in enumerate(filters):
            transmission = read_filter.ReadFilter(item)
            data = transmission.get_filter()

            ax2.plot(data[0, ], data[1, ], '-', lw=0.5, color='midnightblue')

    if residuals:
        diff = np.zeros((2, len(residuals[0].flux)))

        for i, item in enumerate(residuals[0].flux):
            transmission = read_filter.ReadFilter(item)
            diff[0, i] = transmission.mean_wavelength()
            diff[1, i] = (residuals[0].flux[item][0]-residuals[1].flux[item]) / \
                residuals[0].flux[item][1]

        ax3.plot(diff[0, ], diff[1, ], marker='s', ms=5, linestyle='none', color='black')

        res_lim = math.ceil(np.amax(np.abs(diff[1, ])))

        ax3.set_ylabel('Residuals [$\sigma$]', fontsize=15)
        ax3.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=1)
        ax3.set_ylim(-res_lim, res_lim)

        if offset:
            ax3.get_xaxis().set_label_coords(0.5, offset[0])
            ax3.get_yaxis().set_label_coords(offset[1], 0.5)

        else:
            ax3.get_xaxis().set_label_coords(0.5, -0.12)
            ax3.get_yaxis().set_label_coords(-0.1, 0.5)

    if filters:
        ax2.set_ylim(0., 1.)

    sys.stdout.write('Plotting spectrum: '+output+'...')
    sys.stdout.flush()

    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

    if title:
        ax1.set_title(title, y=1.02, fontsize=16)

    handles, _ = ax1.get_legend_handles_labels()

    if handles:
        ax1.legend(loc='upper left', prop={'size':10}, frameon=False)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()
