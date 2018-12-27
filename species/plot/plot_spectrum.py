"""
Module with functions for making plots.
"""

import os
import sys
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from .. core import box
from .. read import read_filter


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def plot_spectrum(boxes,
                  filters,
                  output):
    """
    :param boxes:
    :type boxes: species.analysis.box.SpectrumBox and/or
                 species.analysis.box.PhotometryBox
    :param filters:
    :type filters: tuple(str, )
    :param output:
    :type output: str

    :return: None
    """

    marker = itertools.cycle(('o', 's', '*', 'p', '<', '>', 'P', 'v', '^'))

    plt.figure(1, figsize=(7, 4))
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[0, 0])
    ax2 = ax1.twinx()

    ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4))

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=False)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=False)

    ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=False, right=True)

    ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=False, right=True)

    ax1.set_xlabel('Wavelength [micron]', fontsize=16)
    ax1.set_ylabel('Flux [W m$^{-2}$ micron$^{-1}$]', fontsize=16)
    ax2.set_ylabel('Filter transmission', fontsize=16, rotation=-90)

    ax1.set_xlim(0.6, 6.)
    ax2.set_ylim(0., 1.)

    ax1.get_xaxis().set_label_coords(0.5, -0.09)
    ax1.get_yaxis().set_label_coords(-0.13, 0.5)
    ax2.get_yaxis().set_label_coords(1.1, 0.5)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    for boxitem in boxes:

        if isinstance(boxitem, (box.SpectrumBox, box.ModelBox)):
            wavelength = boxitem.wavelength
            flux = boxitem.flux

            if isinstance(wavelength[0], (np.float32, np.float64)):
                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                ax1.plot(wavelength, masked, lw=1)

            if isinstance(wavelength[0], (np.ndarray)):
                for i, item in enumerate(wavelength):
                    data = np.array(flux[i], dtype=np.float64)
                    masked = np.ma.array(data, mask=np.isnan(data))

                    ax1.plot(item, masked, lw=1)

            if filters:
                for i, item in enumerate(filters):
                    transmission = read_filter.ReadFilter(item)
                    data = transmission.get_filter()

                    ax2.plot(data[0, ], data[1, ], '-', lw=0.5, color='black')

        if isinstance(boxitem, box.PhotometryBox):
            ax1.plot(boxitem.wavelength, boxitem.flux, marker=next(marker), ms=6, \
                     color='black', label=boxitem.name)

    sys.stdout.write('Plotting spectrum: '+output+'...')
    sys.stdout.flush()

    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        ax1.legend(loc='upper left', prop={'size':10}, frameon=True)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()
