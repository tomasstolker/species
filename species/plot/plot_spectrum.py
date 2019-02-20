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

from . util import quantity_unit
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
                  scale=('linear', 'linear'),
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

    if residuals and filters:
        plt.figure(1, figsize=(7, 5))
        gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[1, 3, 1])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[1, 0])
        ax2 = plt.subplot(gridsp[0, 0])
        ax3 = plt.subplot(gridsp[2, 0])

    elif residuals:
        plt.figure(1, figsize=(7, 5))
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])
        ax3 = plt.subplot(gridsp[1, 0])

    elif filters:
        plt.figure(1, figsize=(7, 5))
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[1, 0])
        ax2 = plt.subplot(gridsp[0, 0])

    else:
        plt.figure(1, figsize=(7, 4))
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])

    ax1.grid(True, linestyle=':', linewidth=0.7, color='gray', dashes=(1, 4), alpha=0.3, zorder=0)

    if residuals:
        labelbottom = False
    else:
        labelbottom = True

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=labelbottom)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=labelbottom)

    if filters:
        ax2.grid(True, linestyle=':', linewidth=0.7, color='gray', dashes=(1, 4),
                 alpha=0.3, zorder=0)

        ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=5, labelsize=12, top=True,
                        bottom=True, left=True, right=True, labelbottom=False)

        ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=3, labelsize=12, top=True,
                        bottom=True, left=True, right=True, labelbottom=False)

    if residuals:
        ax3.grid(True, linestyle=':', linewidth=0.7, color='gray', dashes=(1, 4),
                 alpha=0.3, zorder=0)

        ax3.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=5, labelsize=12, top=True,
                        bottom=True, left=True, right=True)

        ax3.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=0.8, length=3, labelsize=12, top=True,
                        bottom=True, left=True, right=True)

    if residuals and filters:
        ax1.set_xlabel('', fontsize=13)
        ax2.set_xlabel('', fontsize=13)
        ax3.set_xlabel('Wavelength [micron]', fontsize=13)

    elif residuals:
        ax1.set_xlabel('', fontsize=13)
        ax3.set_xlabel('Wavelength [micron]', fontsize=13)

    elif filters:
        ax1.set_xlabel('Wavelength [micron]', fontsize=13)
        ax2.set_xlabel('', fontsize=13)

    else:
        ax1.set_xlabel('Wavelength [micron]', fontsize=13)

    if filters:
        ax2.set_ylabel('Transmission', fontsize=13)

    if residuals:
        ax3.set_ylabel('Residual [$\sigma$]', fontsize=13)

    if xlim:
        ax1.set_xlim(xlim[0], xlim[1])
    else:
        ax1.set_xlim(0.6, 6.)

    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])

    if filters:
        ax2.set_ylim(0., 1.)

    xlim = ax1.get_xlim()

    if filters:
        ax2.set_xlim(xlim[0], xlim[1])

    if residuals:
        ax3.set_xlim(xlim[0], xlim[1])

    ylim = ax1.get_ylim()

    exponent = math.floor(math.log10(ylim[1]))
    scaling = 10.**exponent

    ax1.set_ylabel('Flux [10$^{'+str(exponent)+'}$ W m$^{-2}$ $\mu$m$^{-1}$]', fontsize=13)
    ax1.set_ylim(ylim[0]/scaling, ylim[1]/scaling)

    if ylim[0] < 0.:
        ax1.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=1)

    if offset and residuals and filters:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset and filters:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset and residuals:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.12)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)

    ax1.set_xscale(scale[0])
    ax1.set_yscale(scale[1])

    if filters:
        ax2.set_xscale(scale[0])

    if residuals:
        ax3.set_xscale(scale[0])

    for j, boxitem in enumerate(boxes):

        if isinstance(boxitem, (box.SpectrumBox, box.ModelBox)):
            wavelength = boxitem.wavelength
            flux = boxitem.flux

            if isinstance(wavelength[0], (np.float32, np.float64)):
                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                if isinstance(boxitem, box.ModelBox):
                    param = boxitem.parameters
                    #TODO fix luminosity
                    print(param)

                    par_key, par_unit = quantity_unit(list(param.keys()))
                    par_val = list(param.values())

                    label = ''
                    for i, item in enumerate(par_key):

                        if item == '$T_\mathregular{eff}$':
                            value = "{:.1f}".format(par_val[i])
                        elif item in ('$\log\,g$', '$R$', '$M$', '[Fe/H]'):
                            value = "{:.2f}".format(par_val[i])
                        elif item == '$L$':
                            # print(item, par_val[i], par_key[i])
                            value = "{0:.1e}".format(par_val[i])
                        else:
                            continue

                        label += item+' = '+str(value)+' '+par_unit[i]

                        if i < len(par_key)-1:
                            label += ', '

                else:
                    label = None

                # print(label)
                ax1.plot(wavelength, masked/scaling, color=colors[j], lw=1.0, label=label, zorder=4)

            elif isinstance(wavelength[0], (np.ndarray)):
                for i, item in enumerate(wavelength):
                    data = np.array(flux[i], dtype=np.float64)
                    masked = np.ma.array(data, mask=np.isnan(data))

                    ax1.plot(item, masked/scaling, lw=1)

        elif isinstance(boxitem, tuple):
            for i, item in enumerate(boxitem):
                wavelength = item.wavelength
                flux = item.flux

                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                ax1.plot(wavelength, masked/scaling, lw=0.4, color=colors[j], alpha=0.5, zorder=3)

        elif isinstance(boxitem, box.PhotometryBox):
            ax1.plot(boxitem.wavelength, boxitem.flux/scaling, marker=next(marker), ms=6, \
                     color=colors[j], label=boxitem.name)

        elif isinstance(boxitem, box.ObjectBox):
            for item in boxitem.flux:
                transmission = read_filter.ReadFilter(item)
                wavelength = transmission.mean_wavelength()
                fwhm = transmission.filter_fwhm()

                ax1.errorbar(wavelength, boxitem.flux[item][0]/scaling, xerr=fwhm/2.,
                             yerr=boxitem.flux[item][1]/scaling, marker='s', ms=5, zorder=5,
                             color=colors[j], markerfacecolor=colors[j])

        elif isinstance(boxitem, box.SynphotBox):
            for item in boxitem.flux:
                transmission = read_filter.ReadFilter(item)
                wavelength = transmission.mean_wavelength()
                fwhm = transmission.filter_fwhm()

                ax1.errorbar(wavelength, boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                             alpha=0.7, marker='s', ms=5, zorder=6, color=colors[j],
                             markerfacecolor='white')

    if filters:
        for i, item in enumerate(filters):
            transmission = read_filter.ReadFilter(item)
            data = transmission.get_filter()

            ax2.plot(data[0, ], data[1, ], '-', lw=0.7, color='black')

    if residuals:
        diff = np.zeros((2, len(residuals[0].flux)))

        for i, item in enumerate(residuals[0].flux):
            transmission = read_filter.ReadFilter(item)
            diff[0, i] = transmission.mean_wavelength()
            diff[1, i] = (residuals[0].flux[item][0]-residuals[1].flux[item]) / \
                residuals[0].flux[item][1]

        ax3.plot(diff[0, ], diff[1, ], marker='s', ms=5, linestyle='none', color='black')

        res_lim = math.ceil(np.amax(np.abs(diff[1, ])))

        ax3.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=1)
        ax3.set_ylim(-res_lim, res_lim)

    if filters:
        ax2.set_ylim(0., 1.1)

    sys.stdout.write('Plotting spectrum: '+output+'...')
    sys.stdout.flush()

    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax1.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

    if title:
        if filters:
            ax2.set_title(title, y=1.02, fontsize=15)
        else:
            ax1.set_title(title, y=1.02, fontsize=15)

    handles, _ = ax1.get_legend_handles_labels()

    if handles:
        ax1.legend(loc='upper left', prop={'size':9}, frameon=False)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()
