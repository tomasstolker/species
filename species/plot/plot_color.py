"""
Module with functions for creating color-magnitude and color-color plots.
"""

import os
import sys
import math
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from matplotlib.colorbar import Colorbar, ColorbarBase

from species.read import read_object
from species.util import plot_util


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def plot_color_magnitude(colorbox=None,
                         objects=None,
                         isochrones=None,
                         models=None,
                         mass_labels=None,
                         label_x='color [mag]',
                         label_y='M [mag]',
                         xlim=None,
                         ylim=None,
                         offset=None,
                         legend='upper left',
                         ncolumn=1,
                         output='color-magnitude.pdf'):
    """
    Parameters
    ----------
    colorbox : list(species.core.box.ColorMagBox, ), None
        Boxes with the colors and magnitudes. Not used if set to None.
    objects : tuple(tuple(str, str, str, str), ), None
        Tuple with individual objects. The objects require a tuple with their database tag, the two
        filter IDs for the color, and the filter ID for the absolute magnitude. Not used if set to
        None.
    isochrones : tuple(species.core.box.IsochroneBox, ), None
        Tuple with boxes of isochrone data. Not used if set to None.
    models : tuple(species.core.box.ColorMagBox, ), None
        Tuple with :class:`~species.core.box.ColorMagBox` objects which have been created from the
        isochrone data with :func:`~species.read.read_isochrone.ReadIsochrone.get_color_magnitude`.
        These boxes contain synthetic photometry for a given age and a range of masses. Not used
        if set to None.
    mass_labels : list(float, ), None
        Plot labels with masses next to the isochrone data of `models`. The list with masses has
        to be provided in Jupiter mass. No labels are shown if set to None.
    label_x : str
        Label for the x-axis.
    label_y : str
        Label for the y-axis.
    xlim : tuple(float, float)
        Limits for the x-axis.
    ylim : tuple(float, float)
        Limits for the y-axis.
    legend : str, None
        Legend position. Not shown if set to None.
    ncolumn : int
        Numer of columns used for the legend.
    output : str
        Output filename.

    Returns
    -------
    None
    """

    marker = itertools.cycle(('o', 's', '<', '>', 'p', 'v', '^', '*',
                              'd', 'x', '+', '1', '2', '3', '4'))

    model_color = ('tomato', 'teal', 'dodgerblue')
    model_linestyle = ('-', '--', ':', '-.')

    sys.stdout.write('Plotting color-magnitude diagram: '+output+'... ')
    sys.stdout.flush()

    plt.figure(1, figsize=(4., 4.8))
    gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.5])
    gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[2, 0])
    ax2 = plt.subplot(gridsp[0, 0])

    # ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4), zorder=0)

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

    if models is not None:
        cmap_teff = plt.cm.afmhot

        teff_min = np.inf
        teff_max = -np.inf

        for item in models:

            if np.amin(item.sptype) < teff_min:
                teff_min = np.amin(item.sptype)

            if np.amax(item.sptype) > teff_max:
                teff_max = np.amax(item.sptype)

        norm_teff = mpl.colors.Normalize(vmin=teff_min, vmax=teff_max)

        count = 0

        model_dict = {}

        for j, item in enumerate(models):
            if item.library not in model_dict:
                model_dict[item.library] = [count, 0]
                count += 1

            else:
                model_dict[item.library] = [model_dict[item.library][0], model_dict[item.library][1]+1]

            model_count = model_dict[item.library]

            if model_count[1] == 0:
                label = plot_util.model_name(item.library)

                ax1.plot(item.color, item.magnitude, linestyle=model_linestyle[model_count[1]],
                         linewidth=1, zorder=3, color=model_color[model_count[0]], label=label)

                if mass_labels is not None:
                    interp_magnitude = interp1d(item.sptype, item.magnitude)
                    interp_color = interp1d(item.sptype, item.color)

                    for i, mass_item in enumerate(mass_labels):
                        if j == 0 or (j > 0 and mass_item < 20.):
                            pos_color = interp_color(mass_item)
                            pos_mag = interp_magnitude(mass_item)

                            mass_label = str(int(mass_item))+r' M$_\mathregular{J}$'

                            xlim = ax1.get_xlim()
                            ylim = ax1.get_ylim()

                            if xlim[0]+0.2 < pos_color < xlim[1]-0.2 and \
                                    ylim[1]+0.2 < pos_mag < ylim[0]-0.2:

                                ax1.scatter(pos_color, pos_mag, c=model_color[model_count[0]], s=15,
                                            zorder=3, edgecolor='none')

                                ax1.annotate(mass_label, (pos_color, pos_mag),
                                             color=model_color[model_count[0]], fontsize=9,
                                             xytext=(pos_color+0.05, pos_mag-0.1), zorder=9)

            else:
                ax1.plot(item.color, item.magnitude, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, zorder=3, color=model_color[model_count[0]])

            # scat_teff = ax1.scatter(item.color, item.magnitude, c=item.sptype, cmap=cmap_teff,
            #                         norm=norm_teff, zorder=4, s=15, alpha=1.0, edgecolor='none')

        # cb2 = ColorbarBase(ax=ax3, cmap=cmap_teff, norm=norm_teff, orientation='vertical', ticklocation='right')
        # cb2.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='black')
        # cb2.ax.set_ylabel('Temperature [K]', rotation=270, fontsize=12, labelpad=22)

    if colorbox is not None:
        cmap = plt.cm.viridis

        # bounds = np.linspace(0, 7, 8)
        # ticks = np.linspace(0.5, 6.5, 7)

        bounds = np.linspace(0, 6, 7)
        ticks = np.linspace(0.5, 5.5, 6)

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for item in colorbox:
            sptype = item.sptype
            color = item.color
            magnitude = item.magnitude

            indices = np.where(sptype != b'None')[0]

            sptype = sptype[indices]
            color = color[indices]
            magnitude = magnitude[indices]

            spt_disc = plot_util.sptype_substellar(sptype, color.shape)

            _, unique = np.unique(color, return_index=True)

            sptype = sptype[unique]
            color = color[unique]
            magnitude = magnitude[unique]
            spt_disc = spt_disc[unique]

            if item.object_type == 'field':
                scat = ax1.scatter(color, magnitude, c=spt_disc, cmap=cmap, norm=norm, zorder=6,
                                   s=40, alpha=0.6, edgecolor='none')

                cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal', ticklocation='top', format='%.2f')
                cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='black')

                # cb.set_ticks(np.linspace(0.5, 6.5, 7))
                # cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8', 'Y1-Y2'])

                cb.set_ticks(np.linspace(0.5, 5.5, 6))
                cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8'])

            elif item.object_type == 'young':
                ax1.plot(color, magnitude, marker='s', ms=4, linestyle='none', alpha=0.6, 
                         color='gray', markeredgecolor='black', zorder=7, label='Young/low-gravity')

    if isochrones is not None:
        for item in isochrones:
            ax1.plot(item.color, item.magnitude, linestyle='-', linewidth=1., color='black')

    if objects is not None:
        for item in objects:
            objdata = read_object.ReadObject(item[0])

            objcolor1 = objdata.get_photometry(item[1])
            objcolor2 = objdata.get_photometry(item[2])
            abs_mag = objdata.get_absmag(item[3])

            colorerr = math.sqrt(objcolor1[1]**2+objcolor2[1]**2)

            if item[0] in ('HD 135344A - CC'):
                ax1.errorbar(objcolor1[0]-objcolor2[0], abs_mag[0], yerr=abs_mag[1], xerr=colorerr,
                             marker='*', ms=12, color='magenta', label=objdata.object_name,
                             markerfacecolor='magenta', markeredgecolor='black', zorder=10)

            else:
                # if item[0] in ('PDS 70 b'):
                if item[0] in ('beta Pic b', 'HIP 65426 b', 'PZ Tel B', 'HD 206893 B'):
                    color_plot = 'mediumslateblue'
                else:
                    color_plot = 'black'

                ax1.errorbar(objcolor1[0]-objcolor2[0], abs_mag[0], yerr=abs_mag[1], xerr=colorerr,
                             marker=next(marker), ms=6, color=color_plot, label=objdata.object_name,
                             markerfacecolor='white', markeredgecolor=color_plot, zorder=10)

            # ax1.errorbar(objcolor1[0]-objcolor2[0], abs_mag[0], yerr=abs_mag[1], xerr=colorerr,
            #              marker=next(marker), ms=6, color='black', label=objdata.object_name,
            #              markerfacecolor='white', markeredgecolor='black', zorder=10)

    if legend is not None:
        handles, labels = ax1.get_legend_handles_labels()

        if handles:
            ax1.legend(loc=legend, prop={'size': 8}, frameon=False, numpoints=1, ncol=ncolumn)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    sys.stdout.write('[DONE]\n')
    sys.stdout.flush()


def plot_color_color(colorbox,
                     objects=None,
                     models=None,
                     mass_labels=None,
                     label_x='color [mag]',
                     label_y='color [mag]',
                     xlim=None,
                     ylim=None,
                     offset=None,
                     legend='upper left',
                     ncolumn=1,
                     output='color-color.pdf'):
    """
    Parameters
    ----------
    colorbox : species.core.box.ColorColorBox, None
        Box with the colors and magnitudes.
    objects : tuple(tuple(str, str, str, str), ), None
        Tuple with individual objects. The objects require a tuple with their database tag, the
        two filter IDs for the color, and the filter ID for the absolute magnitude. Not used if
        set to None.
    models : tuple(species.core.box.ColorMagBox, ), None
        Tuple with :class:`~species.core.box.ColorColorBox` objects which have been created from
        the isochrone data with :func:`~species.read.read_isochrone.ReadIsochrone.get_color_color`.
        These boxes contain synthetic photometry for a given age and a range of masses. Not used
        if set to None.
    mass_labels : list(float, ), None
        Plot labels with masses next to the isochrone data of `models`. The list with masses has
        to be provided in Jupiter mass. No labels are shown if set to None.
    label_x : str
        Label for the x-axis.
    label_y : str
        Label for the y-axis.
    output : str
        Output filename.
    xlim : tuple(float, float)
        Limits for the x-axis.
    ylim : tuple(float, float)
        Limits for the y-axis.
    offset : tuple(float, float)
        Offset of the x- and y-axis label.
    legend : str
        Legend position.
    ncolumn : int
        Numer of columns used for the legend.

    Returns
    -------
    None
    """

    marker = itertools.cycle(('o', 's', '<', '>', 'p', 'v', '^', '*',
                              'd', 'x', '+', '1', '2', '3', '4'))

    model_color = ('tomato', 'teal', 'dodgerblue')
    model_linestyle = ('-', '--', ':', '-.')

    sys.stdout.write('Plotting color-color diagram: '+output+'... ')
    sys.stdout.flush()

    plt.figure(1, figsize=(4, 4.3))
    gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.])
    gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[2, 0])
    ax2 = plt.subplot(gridsp[0, 0])

    # ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4), zorder=0)

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

    if models is not None:
        cmap_teff = plt.cm.afmhot

        teff_min = np.inf
        teff_max = -np.inf

        for item in models:

            if np.amin(item.sptype) < teff_min:
                teff_min = np.amin(item.sptype)

            if np.amax(item.sptype) > teff_max:
                teff_max = np.amax(item.sptype)

        norm_teff = mpl.colors.Normalize(vmin=teff_min, vmax=teff_max)

        count = 0

        model_dict = {}

        for j, item in enumerate(models):
            if item.library not in model_dict:
                model_dict[item.library] = [count, 0]
                count += 1

            else:
                model_dict[item.library] = [model_dict[item.library][0], model_dict[item.library][1]+1]

            model_count = model_dict[item.library]

            if model_count[1] == 0:
                label = plot_util.model_name(item.library)

                ax1.plot(item.color1, item.color2, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, zorder=3, color=model_color[model_count[0]], label=label)

                if mass_labels is not None:
                    interp_color1 = interp1d(item.sptype, item.color1)
                    interp_color2 = interp1d(item.sptype, item.color2)

                    for i, mass_item in enumerate(mass_labels):
                        if j == 0 or (j > 0 and mass_item < 20.):
                            pos_color1 = interp_color1(mass_item)
                            pos_color2 = interp_color2(mass_item)

                            mass_label = str(int(mass_item))+r' M$_\mathregular{J}$'

                            xlim = ax1.get_xlim()
                            ylim = ax1.get_ylim()

                            if xlim[0]+0.2 < pos_color1 < xlim[1]-0.2 and \
                                    ylim[0]+0.2 < pos_color2 < ylim[1]-0.2:

                                ax1.scatter(pos_color1, pos_color2, c=model_color[model_count[0]], s=15,
                                            zorder=3, edgecolor='none')

                                ax1.annotate(mass_label, (pos_color1, pos_color2),
                                             color=model_color[model_count[0]], fontsize=9,
                                             xytext=(pos_color1+0.05, pos_color2-0.1), zorder=9)

            else:
                ax1.plot(item.color1, item.color2, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, zorder=3, color=model_color[model_count[0]])

            # scat_teff = ax1.scatter(item.color, item.magnitude, c=item.sptype, cmap=cmap_teff,
            #                         norm=norm_teff, zorder=4, s=15, alpha=1.0, edgecolor='none')

        # cb2 = ColorbarBase(ax=ax3, cmap=cmap_teff, norm=norm_teff, orientation='vertical', ticklocation='right')
        # cb2.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='black')
        # cb2.ax.set_ylabel('Temperature [K]', rotation=270, fontsize=12, labelpad=22)

    if colorbox is not None:
        cmap = plt.cm.viridis

        # bounds = np.linspace(0, 7, 8)
        # ticks = np.linspace(0.5, 6.5, 7)

        bounds = np.linspace(0, 6, 7)
        ticks = np.linspace(0.5, 5.5, 6)

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for item in colorbox:
            sptype = item.sptype
            color1 = item.color1
            color2 = item.color2

            indices = np.where(sptype != 'None')[0]

            sptype = sptype[indices]
            color1 = color1[indices]
            color2 = color2[indices]

            spt_disc = plot_util.sptype_substellar(sptype, color1.shape)
            _, unique = np.unique(color1, return_index=True)

            sptype = sptype[unique]
            color1 = color1[unique]
            color2 = color2[unique]
            spt_disc = spt_disc[unique]

            if item.object_type == 'field':
                scat = ax1.scatter(color1, color2, c=spt_disc, cmap=cmap, norm=norm, zorder=6,
                                   s=40, alpha=0.6, edgecolor='none')

                cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal', ticklocation='top', format='%.2f')
                cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='black')

                # cb.set_ticks(np.linspace(0.5, 6.5, 7))
                # cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8', 'Y1-Y2'])

                cb.set_ticks(np.linspace(0.5, 5.5, 6))
                cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8'])

            elif item.object_type == 'young':
                ax1.plot(color1, color2, marker='s', ms=4, linestyle='none', alpha=0.6, 
                         color='gray', markeredgecolor='black', zorder=7, label='Young/low-gravity')

            # scat = ax1.scatter(color1, color2, c=spt_disc, cmap=cmap, norm=norm,
            #                    zorder=5, s=40, alpha=0.6, edgecolor='none')
            #
            # cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
            #               ticklocation='top', format='%.2f')

            # cb.ax.tick_params(width=0.8, length=5, labelsize=10, direction='in', color='black')
            # cb.set_ticks(np.arange(0.5, 7., 1.))
            # cb.set_ticklabels(['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T6-T8', 'Y1-Y2'])

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

            # if item[0] in ('PDS 70 b'):
            if item[0] in ('beta Pic b', 'HIP 65426 b', 'PZ Tel B', 'HD 206893 B'):
                color_plot = 'mediumslateblue'
            else:
                color_plot = 'black'

            ax1.errorbar(color1, color2, xerr=error1, yerr=error2,
                         marker=next(marker), ms=6, color=color_plot, label=objdata.object_name,
                         markerfacecolor='white', markeredgecolor=color_plot, zorder=10)

            # ax1.errorbar(color1, color2, xerr=error1, yerr=error2,
            #              marker=next(marker), ms=6, color='black', label=objdata.object_name,
            #              markerfacecolor='white', markeredgecolor='black', zorder=10)

    handles, labels = ax1.get_legend_handles_labels()

    if handles:
        ax1.legend(loc=legend, prop={'size': 8}, frameon=False, numpoints=1, ncol=ncolumn)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    sys.stdout.write('[DONE]\n')
    sys.stdout.flush()
