"""
Module with functions for creating color-magnitude and color-color plot.
"""

import os
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from matplotlib.colorbar import Colorbar

from species.core import box
from species.read import read_object
from species.util import plot_util


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2.5)


def plot_color_magnitude(boxes,
                         objects=None,
                         mass_labels=None,
                         companion_labels=False,
                         field_range=None,
                         label_x='Color [mag]',
                         label_y='Magnitude [mag]',
                         xlim=None,
                         ylim=None,
                         offset=None,
                         legend='upper left',
                         output='color-magnitude.pdf'):
    """
    Function for creating a color-magnitude diagram.

    Parameters
    ----------
    boxes : list(species.core.box.ColorMagBox, species.core.box.IsochroneBox, )
        Boxes with the color-magnitude and isochrone data from photometric libraries, spectral
        libraries, and/or atmospheric models. The synthetic data have to be created with
        :func:`~species.read.read_isochrone.ReadIsochrone.get_color_magnitude`. These boxes
        contain synthetic colors and magnitudes for a given age and a range of masses.
    objects : tuple(tuple(str, str, str, str), ),
              tuple(tuple(str, str, str, str, str, str, dict, dict), ), None
        Tuple with individual objects. The objects require a tuple with their database tag, the two
        filter names for the color, and the filter names for the absolute magnitude. Optionally, a
        dictionary with keyword arguments can be provided for the object's marker and label,
        respectively. For example, ``{'marker': 'o', 'ms': 10}`` for the marker and
        ``{'ha': 'left', 'va': 'bottom', 'xytext': (5, 5)})`` for the label. Not used if set to
        None.
    mass_labels : list(float, ), list(tuple(float, str), ), None
        Plot labels with masses next to the isochrone data of `models`. The list with masses has
        to be provided in Jupiter mass. Alternatively, a list of tuples can be provided with
        the planet mass and position of the label ('left' or 'right), for example
        ``[(10., 'left'), (20., 'right')]``. No labels are shown if set to None.
    companion_labels : bool
        Plot labels with the names of the directly imaged companions.
    field_range : tuple(str, str), None
        Range of the discrete colorbar for the field dwarfs. The tuple should contain the lower
        and upper value ('early M', 'late M', 'early L', 'late L', 'early T', 'late T', 'early Y).
        The full range is used if set to None.
    label_x : str
        Label for the x-axis.
    label_y : str
        Label for the y-axis.
    xlim : tuple(float, float), None
        Limits for the x-axis. Not used if set to None.
    ylim : tuple(float, float), None
        Limits for the y-axis. Not used if set to None.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label.
    legend : str, None
        Legend position. Not shown if set to None.
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None

    """

    print(f'Plotting color-magnitude diagram: {output}... ', end='', flush=True)

    model_color = ('#234398', '#f6a432', 'black')
    model_linestyle = ('-', '--', ':', '-.')

    isochrones = []
    models = []
    empirical = []

    for item in boxes:
        if isinstance(item, box.IsochroneBox):
            isochrones.append(item)

        elif isinstance(item, box.ColorMagBox):
            if item.object_type == 'model':
                models.append(item)
            else:
                empirical.append(item)

        else:
            raise ValueError(f'Found a {type(item)} while only ColorMagBox and IsochroneBox '
                             f'objects can be provided to \'boxes\'.')

    if empirical:
        plt.figure(1, figsize=(4., 4.8))
        gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.5])
        gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[2, 0])
        ax2 = plt.subplot(gridsp[0, 0])

    else:
        plt.figure(1, figsize=(4., 4.5))
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=1, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=1, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.set_xlabel(label_x, fontsize=14)
    ax1.set_ylabel(label_y, fontsize=14)

    ax1.invert_yaxis()

    if offset is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.08)
        ax1.get_yaxis().set_label_coords(-0.12, 0.5)

    if xlim is not None:
        ax1.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
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
                model_dict[item.library] = [model_dict[item.library][0],
                                            model_dict[item.library][1]+1]

            model_count = model_dict[item.library]

            if model_count[0] == 3:
                raise ValueError('Only three different types of model atmospheres can be added.')

            if model_count[1] == 0:
                label = plot_util.model_name(item.library)

                ax1.plot(item.color, item.magnitude, linestyle=model_linestyle[model_count[1]],
                         linewidth=1.2, color=model_color[model_count[0]], label=label, zorder=0)

                if mass_labels is not None:
                    interp_magnitude = interp1d(item.sptype, item.magnitude)
                    interp_color = interp1d(item.sptype, item.color)

                    for i, mass_item in enumerate(mass_labels):
                        if isinstance(mass_item, tuple):
                            mass_val = mass_item[0]
                            mass_pos = mass_item[1]

                        else:
                            mass_val = mass_item
                            mass_pos = 'right'

                        if j == 0 or (j > 0 and mass_val < 20.):
                            pos_color = interp_color(mass_val)
                            pos_mag = interp_magnitude(mass_val)

                            if mass_pos == 'left':
                                mass_ha = 'right'
                                mass_xytext = (pos_color-0.05, pos_mag)

                            else:
                                mass_ha = 'left'
                                mass_xytext = (pos_color+0.05, pos_mag)

                            mass_label = str(int(mass_val))+r' M$_\mathregular{J}$'

                            xlim = ax1.get_xlim()
                            ylim = ax1.get_ylim()

                            if xlim[0]+0.2 < pos_color < xlim[1]-0.2 and \
                                    ylim[1]+0.2 < pos_mag < ylim[0]-0.2:

                                ax1.scatter(pos_color, pos_mag, c=model_color[model_count[0]], s=15,
                                            edgecolor='none', zorder=0)

                                ax1.annotate(mass_label, (pos_color, pos_mag),
                                             color=model_color[model_count[0]], fontsize=9,
                                             xytext=mass_xytext, zorder=1, ha=mass_ha, va='center')

            else:
                ax1.plot(item.color, item.magnitude, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, color=model_color[model_count[0]], zorder=0)

    if empirical:
        cmap = plt.cm.viridis

        bounds, ticks, ticklabels = plot_util.field_bounds_ticks(field_range)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for item in empirical:
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

            if item.object_type == 'field' or item.object_type is None:
                scat = ax1.scatter(color, magnitude, c=spt_disc, cmap=cmap, norm=norm, s=50,
                                   alpha=0.7, edgecolor='none', zorder=2)

                cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                              ticklocation='top', format='%.2f')

                cb.ax.tick_params(width=1, length=5, labelsize=10, direction='in', color='black')

                cb.set_ticks(ticks)
                cb.set_ticklabels(ticklabels)

            elif item.object_type == 'young':
                ax1.plot(color, magnitude, marker='s', ms=4, linestyle='none', alpha=0.7,
                         color='gray', markeredgecolor='black', label='Young/low-gravity', zorder=2)

    if isochrones:
        for item in isochrones:
            ax1.plot(item.color, item.magnitude, linestyle='-', linewidth=1.2, color='black')

    if objects is not None:
        for i, item in enumerate(objects):
            objdata = read_object.ReadObject(item[0])

            objcolor1 = objdata.get_photometry(item[1])
            objcolor2 = objdata.get_photometry(item[2])
            abs_mag = objdata.get_absmag(item[3])

            colorerr = math.sqrt(objcolor1[1]**2+objcolor2[1]**2)

            x_color = objcolor1[0]-objcolor2[0]
            y_mag = abs_mag[0]

            if len(item) > 4 and item[4] is not None:
                kwargs = item[4]

            else:
                kwargs = {'marker': '>',
                          'ms': 6.,
                          'color': 'black',
                          'mfc': 'white',
                          'mec': 'black',
                          'label': 'Directly imaged'}

            ax1.errorbar(x_color, y_mag, yerr=abs_mag[1], xerr=colorerr, zorder=3, **kwargs)

            if companion_labels:
                x_range = ax1.get_xlim()
                y_range = ax1.get_ylim()

                if len(item) > 4:
                    kwargs = item[5]

                else:
                    kwargs = {'ha': 'left',
                              'va': 'bottom',
                              'fontsize': 8.5,
                              'xytext': (5., 5.),
                              'color': 'black'}

                ax1.annotate(objdata.object_name, (x_color, y_mag), zorder=3,
                             textcoords='offset points', **kwargs)

    if legend is not None:
        handles, labels = ax1.get_legend_handles_labels()

        # prevent duplicates
        by_label = dict(zip(labels, handles))

        if handles:
            ax1.legend(by_label.values(), by_label.keys(), loc=legend, fontsize=8.5,
                       frameon=False, numpoints=1)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print('[DONE]')


def plot_color_color(boxes,
                     objects=None,
                     mass_labels=None,
                     companion_labels=False,
                     field_range=None,
                     label_x='color [mag]',
                     label_y='color [mag]',
                     xlim=None,
                     ylim=None,
                     offset=None,
                     legend='upper left',
                     output='color-color.pdf'):
    """
    Function for creating a color-color diagram.

    Parameters
    ----------
    boxes : list(species.core.box.ColorColorBox, species.core.box.IsochroneBox, )
        Boxes with the color-color and isochrone data from photometric libraries, spectral
        libraries, and/or atmospheric models. The synthetic data have to be created with
        :func:`~species.read.read_isochrone.ReadIsochrone.get_color_color`. These boxes
        contain synthetic colors for a given age and a range of masses.
    objects : tuple(tuple(str, str, str, str), ),
              tuple(tuple(str, str, str, str, str, str, dict, dict), ), None
        Tuple with individual objects. The objects require a tuple with their database tag, the two
        filter names for the first color, and the two filter names for the second color.
        Optionally, a dictionary with keyword arguments can be provided for the object's marker and
        label, respectively. For example, ``{'marker': 'o', 'ms': 10}`` for the marker and
        ``{'ha': 'left', 'va': 'bottom', 'xytext': (5, 5)})`` for the label. Not used if set to
        None.
    mass_labels : list(float, ), list(tuple(float, str), ), None
        Plot labels with masses next to the isochrone data of `models`. The list with masses has
        to be provided in Jupiter mass. Alternatively, a list of tuples can be provided with
        the planet mass and position of the label ('left' or 'right), for example
        ``[(10., 'left'), (20., 'right')]``. No labels are shown if set to None.
    companion_labels : bool
        Plot labels with the names of the directly imaged companions.
    field_range : tuple(str, str), None
        Range of the discrete colorbar for the field dwarfs. The tuple should contain the lower
        and upper value ('early M', 'late M', 'early L', 'late L', 'early T', 'late T', 'early Y).
        The full range is used if set to None.
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
    offset : tuple(float, float), None
        Offset of the x- and y-axis label.
    legend : str
        Legend position.

    Returns
    -------
    NoneType
        None
    """

    print(f'Plotting color-color diagram: {output}... ', end='', flush=True)

    model_color = ('#234398', '#f6a432')
    model_linestyle = ('-', '--', ':', '-.')

    isochrones = []
    models = []
    empirical = []

    for item in boxes:
        if isinstance(item, box.IsochroneBox):
            isochrones.append(item)

        elif isinstance(item, box.ColorColorBox):
            if item.object_type == 'model':
                models.append(item)
            else:
                empirical.append(item)

        else:
            raise ValueError(f'Found a {type(item)} while only ColorColorBox and IsochroneBox '
                             f'objects can be provided to \'boxes\'.')

    plt.figure(1, figsize=(4, 4.3))
    gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[0.2, 0.1, 4.])
    gridsp.update(wspace=0., hspace=0., left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[2, 0])
    ax2 = plt.subplot(gridsp[0, 0])

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=1, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=1, length=3, labelsize=12, top=True,
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
                model_dict[item.library] = [model_dict[item.library][0],
                                            model_dict[item.library][1]+1]

            model_count = model_dict[item.library]

            if model_count[1] == 0:
                label = plot_util.model_name(item.library)

                ax1.plot(item.color1, item.color2, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, color=model_color[model_count[0]], label=label, zorder=0)

                if mass_labels is not None:
                    interp_color1 = interp1d(item.sptype, item.color1)
                    interp_color2 = interp1d(item.sptype, item.color2)

                    for i, mass_item in enumerate(mass_labels):
                        if isinstance(mass_item, tuple):
                            mass_val = mass_item[0]
                            mass_pos = mass_item[1]

                        else:
                            mass_val = mass_item
                            mass_pos = 'right'

                        if j == 0 or (j > 0 and mass_val < 20.):
                            pos_color1 = interp_color1(mass_val)
                            pos_color2 = interp_color2(mass_val)

                            if mass_pos == 'left':
                                mass_ha = 'right'
                                mass_xytext = (pos_color1-0.05, pos_color2)

                            else:
                                mass_ha = 'left'
                                mass_xytext = (pos_color1+0.05, pos_color2)

                            mass_label = str(int(mass_val))+r' M$_\mathregular{J}$'

                            xlim = ax1.get_xlim()
                            ylim = ax1.get_ylim()

                            if xlim[0]+0.2 < pos_color1 < xlim[1]-0.2 and \
                                    ylim[0]+0.2 < pos_color2 < ylim[1]-0.2:

                                ax1.scatter(pos_color1, pos_color2, c=model_color[model_count[0]],
                                            s=15, edgecolor='none', zorder=0)

                                ax1.annotate(mass_label, (pos_color1, pos_color2),
                                             color=model_color[model_count[0]], fontsize=9,
                                             xytext=mass_xytext, ha=mass_ha, va='center', zorder=1)

            else:
                ax1.plot(item.color1, item.color2, linestyle=model_linestyle[model_count[1]],
                         linewidth=0.6, color=model_color[model_count[0]], zorder=0)

    if empirical:
        cmap = plt.cm.viridis

        bounds, ticks, ticklabels = plot_util.field_bounds_ticks(field_range)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for item in empirical:
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
                scat = ax1.scatter(color1, color2, c=spt_disc, cmap=cmap, norm=norm, s=50,
                                   alpha=0.7, edgecolor='none', zorder=2)

                cb = Colorbar(ax=ax2, mappable=scat, orientation='horizontal',
                              ticklocation='top', format='%.2f')

                cb.ax.tick_params(width=1, length=5, labelsize=10, direction='in', color='black')

                cb.set_ticks(ticks)
                cb.set_ticklabels(ticklabels)

            elif item.object_type == 'young':
                ax1.plot(color1, color2, marker='s', ms=4, linestyle='none', alpha=0.7,
                         color='gray', markeredgecolor='black', label='Young/low-gravity', zorder=2)

    if isochrones:
        for item in isochrones:
            ax1.plot(item.colors[0], item.colors[1], linestyle='-', linewidth=1.2, color='black')

    if objects is not None:
        for i, item in enumerate(objects):
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

            if len(item) > 3 and item[3] is not None:
                kwargs = item[3]

            else:
                kwargs = {'marker': '>',
                          'ms': 6.,
                          'color': 'black',
                          'mfc': 'white',
                          'mec': 'black',
                          'label': 'Directly imaged'}

            ax1.errorbar(color1, color2, xerr=error1, yerr=error2, zorder=3, **kwargs)

            if companion_labels:
                x_range = ax1.get_xlim()
                y_range = ax1.get_ylim()

                if len(item) > 3:
                    kwargs = item[4]

                else:
                    kwargs = {'ha': 'left',
                              'va': 'bottom',
                              'fontsize': 8.5,
                              'xytext': (5., 5.),
                              'color': 'black'}

                ax1.annotate(objdata.object_name, (color1, color2), zorder=3,
                             textcoords='offset points', **kwargs)

    handles, labels = ax1.get_legend_handles_labels()

    if legend is not None:
        handles, labels = ax1.get_legend_handles_labels()

        # prevent duplicates
        by_label = dict(zip(labels, handles))

        if handles:
            ax1.legend(by_label.values(), by_label.keys(), loc=legend, fontsize=8.5,
                       frameon=False, numpoints=1)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print('[DONE]')
