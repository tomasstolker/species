"""
Module with a function for plotting spectra.
"""

import os
import math
import warnings
import itertools

from typing import Optional, Union, Tuple, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked

from species.core import box, constants
from species.read import read_filter
from species.util import plot_util


@typechecked
def plot_spectrum(boxes: list,
                  filters: Optional[List[str]] = None,
                  residuals: Optional[box.ResidualsBox] = None,
                  plot_kwargs: Optional[List[Optional[dict]]] = None,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  ylim_res: Optional[Tuple[float, float]] = None,
                  scale: Optional[Tuple[str, str]] = None,
                  title: Optional[str] = None,
                  offset: Optional[Tuple[float, float]] = None,
                  legend: Union[str, dict, Tuple[float, float],
                                List[Optional[Union[dict, str, Tuple[float, float]]]]] = None,
                  figsize: Optional[Tuple[float, float]] = (7., 5.),
                  object_type: str = 'planet',
                  quantity: str = 'flux',
                  output: str = 'spectrum.pdf'):
    """
    Parameters
    ----------
    boxes : list(species.core.box, )
        Boxes with data.
    filters : list(str, ), None
        Filter IDs for which the transmission profile is plotted. Not plotted if set to None.
    residuals : species.core.box.ResidualsBox, None
        Box with residuals of a fit. Not plotted if set to None.
    plot_kwargs : list(dict, ), None
        List with dictionaries of keyword arguments for each box. For example, if the ``boxes``
        are a ``ModelBox`` and ``ObjectBox``:

        .. code-block:: python

            plot_kwargs=[{'ls': '-', 'lw': 1., 'color': 'black'},
                         {'spectrum_1': {'marker': 'o', 'ms': 3., 'color': 'tab:brown', 'ls': 'none'},
                          'spectrum_2': {'marker': 'o', 'ms': 3., 'color': 'tab:blue', 'ls': 'none'},
                          'Paranal/SPHERE.IRDIS_D_H23_3': {'marker': 's', 'ms': 4., 'color': 'tab:cyan', 'ls': 'none'},
                          'Paranal/SPHERE.IRDIS_D_K12_1': [{'marker': 's', 'ms': 4., 'color': 'tab:orange', 'ls': 'none'},
                                                           {'marker': 's', 'ms': 4., 'color': 'tab:red', 'ls': 'none'}],
                          'Paranal/NACO.Lp': {'marker': 's', 'ms': 4., 'color': 'tab:green', 'ls': 'none'},
                          'Paranal/NACO.Mp': {'marker': 's', 'ms': 4., 'color': 'tab:green', 'ls': 'none'}}]

        For an ``ObjectBox``, the dictionary contains items for the different spectrum and filter
        names stored with :func:`~species.data.database.Database.add_object`. In case both
        and ``ObjectBox`` and a ``SynphotBox`` are provided, then the latter can be set to ``None``
        in order to use the same (but open) symbols as the data from the ``ObjectBox``. Note that
        if a filter name is duplicated in an ``ObjectBox`` (Paranal/SPHERE.IRDIS_D_K12_1 in the
        example) then a list with two dictionaries should be provided. Colors are automatically
        chosen if ``plot_kwargs`` is set to ``None``.
    xlim : tuple(float, float)
        Limits of the wavelength axis.
    ylim : tuple(float, float)
        Limits of the flux axis.
    ylim_res : tuple(float, float), None
        Limits of the residuals axis. Automatically chosen (based on the minimum and maximum
        residual value) if set to None.
    scale : tuple(str, str), None
        Scale of the x and y axes ('linear' or 'log'). The scale is set to ``('linear', 'linear')``
        if set to ``None``.
    title : str
        Title.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    legend : str, tuple, dict, list(dict, dict), None
        Location of the legend (str or tuple(float, float)) or a dictionary with the ``**kwargs``
        of ``matplotlib.pyplot.legend``, for example ``{'loc': 'upper left', 'fontsize: 12.}``.
        Alternatively, a list with two values can be provided to separate the model and data
        handles in two legends. Each of these two elements can be set to ``None``. For example,
        ``[None, {'loc': 'upper left', 'fontsize: 12.}]``, if only the data points should be
        included in a legend.                  
    figsize : tuple(float, float)
        Figure size.
    object_type : str
        Object type ('planet' or 'star'). With 'planet', the radius and mass are expressed in
        Jupiter units. With 'star', the radius and mass are expressed in solar units.
    quantity: str
        The quantity of the y-axis ('flux' or 'magnitude').
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.2)
    plt.rcParams['axes.axisbelow'] = False

    if plot_kwargs is None:
        plot_kwargs = []

    elif plot_kwargs is not None and len(boxes) != len(plot_kwargs):
        raise ValueError(f'The number of \'boxes\' ({len(boxes)}) should be equal to the '
                         f'number of items in \'plot_kwargs\' ({len(plot_kwargs)}).')

    if residuals is not None and filters is not None:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(3, 1, height_ratios=[1, 3, 1])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[1, 0])
        ax2 = plt.subplot(gridsp[0, 0])
        ax3 = plt.subplot(gridsp[2, 0])

    elif residuals is not None:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])
        ax3 = plt.subplot(gridsp[1, 0])

    elif filters is not None:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[1, 0])
        ax2 = plt.subplot(gridsp[0, 0])

    else:
        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gridsp[0, 0])

    if residuals is not None:
        labelbottom = False
    else:
        labelbottom = True

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=1, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=labelbottom)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=1, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=labelbottom)

    if filters is not None:
        ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True,
                        bottom=True, left=True, right=True, labelbottom=False)

        ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True,
                        bottom=True, left=True, right=True, labelbottom=False)

    if residuals is not None:
        ax3.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True,
                        bottom=True, left=True, right=True)

        ax3.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True,
                        bottom=True, left=True, right=True)

    if residuals is not None and filters is not None:
        ax1.set_xlabel('', fontsize=13)
        ax2.set_xlabel('', fontsize=13)
        ax3.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)

    elif residuals is not None:
        ax1.set_xlabel('', fontsize=13)
        ax3.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)

    elif filters is not None:
        ax1.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)
        ax2.set_xlabel('', fontsize=13)

    else:
        ax1.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)

    if filters is not None:
        ax2.set_ylabel('Transmission', fontsize=13)

    if residuals is not None:
        ax3.set_ylabel(r'$\Delta$$F_\lambda$ ($\sigma$)', fontsize=13)

    if xlim is not None:
        ax1.set_xlim(xlim[0], xlim[1])
    else:
        ax1.set_xlim(0.6, 6.)

    if quantity == 'magnitude':
        scaling = 1.
        ax1.set_ylabel('Flux contrast (mag)', fontsize=13)

        if ylim:
            ax1.set_ylim(ylim[0], ylim[1])

    elif quantity == 'flux':
        if ylim:
            ax1.set_ylim(ylim[0], ylim[1])

            ylim = ax1.get_ylim()

            exponent = math.floor(math.log10(ylim[1]))
            scaling = 10.**exponent

            ylabel = r'$F_\lambda$ (10$^{'+str(exponent)+r'}$ W m$^{-2}$ $\mu$m$^{-1}$)'

            ax1.set_ylabel(ylabel, fontsize=13)
            ax1.set_ylim(ylim[0]/scaling, ylim[1]/scaling)

            if ylim[0] < 0.:
                ax1.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=0.5)

        else:
            ax1.set_ylabel(r'$F_\lambda$ (W m$^{-2}$ $\mu$m$^{-1}$)', fontsize=13)
            scaling = 1.

    if filters is not None:
        ax2.set_ylim(0., 1.)

    xlim = ax1.get_xlim()

    if filters is not None:
        ax2.set_xlim(xlim[0], xlim[1])

    if residuals is not None:
        ax3.set_xlim(xlim[0], xlim[1])

    if offset is not None and residuals is not None and filters is not None:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None and filters is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None and residuals is not None:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.12)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)

    if scale is None:
        scale = ('linear', 'linear')

    ax1.set_xscale(scale[0])
    ax1.set_yscale(scale[1])

    if filters is not None:
        ax2.set_xscale(scale[0])

    if residuals is not None:
        ax3.set_xscale(scale[0])

    for j, boxitem in enumerate(boxes):
        if j < len(boxes):
            plot_kwargs.append(None)

        if isinstance(boxitem, (box.SpectrumBox, box.ModelBox)):
            wavelength = boxitem.wavelength
            flux = boxitem.flux

            if isinstance(wavelength[0], (np.float32, np.float64)):
                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                if isinstance(boxitem, box.ModelBox):
                    param = boxitem.parameters

                    par_key, par_unit, par_label = plot_util.quantity_unit(
                        param=list(param.keys()), object_type=object_type)

                    label = ''
                    newline = False

                    for i, item in enumerate(par_key):

                        if item[:4] == 'teff':
                            value = f'{param[item]:.0f}'

                        elif item in ['logg', 'feh', 'co', 'fsed']:
                            value = f'{param[item]:.2f}'

                        elif item[:6] == 'radius':

                            if object_type == 'planet':
                                value = f'{param[item]:.1f}'

                            elif object_type == 'star':
                                value = f'{param[item]*constants.R_JUP/constants.R_SUN:.1f}'

                        elif item == 'mass':
                            if object_type == 'planet':
                                value = f'{param[item]:.2f}'
                            elif object_type == 'star':
                                value = f'{param[item]*constants.M_JUP/constants.M_SUN:.2f}'

                        elif item == 'luminosity':
                            value = f'{np.log10(param[item]):.1f}'

                        else:
                            continue

                        # if len(label) > 80 and newline == False:
                        #     label += '\n'
                        #     newline = True

                        if par_unit[i] is None:
                            label += f'{par_label[i]} = {value}'
                        else:
                            label += f'{par_label[i]} = {value} {par_unit[i]}'

                        if i < len(par_key)-1:
                            label += ', '

                else:
                    label = None

                if plot_kwargs[j]:
                    kwargs_copy = plot_kwargs[j].copy()

                    if 'label' in kwargs_copy:
                        if kwargs_copy['label'] is None:
                            label = None
                        else:
                            label = kwargs_copy['label']

                        del kwargs_copy['label']

                    ax1.plot(wavelength, masked/scaling, zorder=2, label=label, **kwargs_copy)

                else:
                    ax1.plot(wavelength, masked/scaling, lw=0.5, label=label, zorder=2)

            elif isinstance(wavelength[0], (np.ndarray)):
                for i, item in enumerate(wavelength):
                    data = np.array(flux[i], dtype=np.float64)
                    masked = np.ma.array(data, mask=np.isnan(data))

                    if isinstance(boxitem.name[i], bytes):
                        label = boxitem.name[i].decode('utf-8')
                    else:
                        label = boxitem.name[i]

                    ax1.plot(item, masked/scaling, lw=0.5, label=label)

        elif isinstance(boxitem, list):
            for i, item in enumerate(boxitem):
                wavelength = item.wavelength
                flux = item.flux

                data = np.array(flux, dtype=np.float64)
                masked = np.ma.array(data, mask=np.isnan(data))

                if plot_kwargs[j]:
                    ax1.plot(wavelength, masked/scaling, zorder=1, **plot_kwargs[j])
                else:
                    ax1.plot(wavelength, masked/scaling, color='gray', lw=0.2, alpha=0.5, zorder=1)

        elif isinstance(boxitem, box.PhotometryBox):
            for i, item in enumerate(boxitem.wavelength):
                transmission = read_filter.ReadFilter(boxitem.filter_name[i])
                fwhm = transmission.filter_fwhm()

                if plot_kwargs[j]:
                    ax1.errorbar(item, boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                 yerr=boxitem.flux[i][1]/scaling, zorder=3, **plot_kwargs[j])
                else:
                    ax1.errorbar(item, boxitem.flux[i][0]/scaling, xerr=fwhm/2.,
                                 yerr=boxitem.flux[i][1]/scaling, marker='s', ms=6, color='black',
                                 zorder=3)

        elif isinstance(boxitem, box.ObjectBox):
            if boxitem.spectrum is not None:
                spec_list = []
                wavel_list = []

                for item in boxitem.spectrum:
                    spec_list.append(item)
                    wavel_list.append(boxitem.spectrum[item][0][0, 0])

                sort_index = np.argsort(wavel_list)
                spec_sort = []

                for i in range(sort_index.size):
                    spec_sort.append(spec_list[sort_index[i]])

                for key in spec_sort:
                    masked = np.ma.array(boxitem.spectrum[key][0],
                                         mask=np.isnan(boxitem.spectrum[key][0]))

                    if not plot_kwargs[j] or key not in plot_kwargs[j]:
                        plot_obj = ax1.errorbar(masked[:, 0], masked[:, 1]/scaling,
                                                yerr=masked[:, 2]/scaling, ms=2, marker='s',
                                                zorder=2.5, ls='none')

                        plot_kwargs[j][key] = {'marker': 's', 'ms': 2., 'ls': 'none',
                                               'color': plot_obj[0].get_color()}

                    else:
                        ax1.errorbar(masked[:, 0], masked[:, 1]/scaling, yerr=masked[:, 2]/scaling,
                                     zorder=2.5, **plot_kwargs[j][key])

            if boxitem.flux is not None:
                filter_list = []
                wavel_list = []

                for item in boxitem.flux:
                    read_filt = read_filter.ReadFilter(item)
                    filter_list.append(item)
                    wavel_list.append(read_filt.mean_wavelength())

                sort_index = np.argsort(wavel_list)
                filter_sort = []

                for i in range(sort_index.size):
                    filter_sort.append(filter_list[sort_index[i]])

                for item in filter_sort:
                    transmission = read_filter.ReadFilter(item)
                    wavelength = transmission.mean_wavelength()
                    fwhm = transmission.filter_fwhm()

                    if not plot_kwargs[j] or item not in plot_kwargs[j]:
                        if not plot_kwargs[j]:
                            plot_kwargs[j] = {}

                        if isinstance(boxitem.flux[item][0], np.ndarray):
                            for i in range(boxitem.flux[item].shape[1]):

                                plot_obj = ax1.errorbar(wavelength, boxitem.flux[item][0, i]/scaling, xerr=fwhm/2.,
                                             yerr=boxitem.flux[item][1, i]/scaling, marker='s', ms=5, zorder=3)

                        else:
                            plot_obj = ax1.errorbar(wavelength, boxitem.flux[item][0]/scaling, xerr=fwhm/2.,
                                         yerr=boxitem.flux[item][1]/scaling, marker='s', ms=5, zorder=3)

                        plot_kwargs[j][item] = {'marker': 's', 'ms': 5., 'color': plot_obj[0].get_color()}

                    else:
                        if isinstance(boxitem.flux[item][0], np.ndarray):
                            if not isinstance(plot_kwargs[j][item], list):
                                raise ValueError(f'A list with {boxitem.flux[item].shape[1]} '
                                                 f'dictionaries are required because the filter '
                                                 f'{item} has {boxitem.flux[item].shape[1]} '
                                                 f'values.')

                            for i in range(boxitem.flux[item].shape[1]):

                                ax1.errorbar(wavelength, boxitem.flux[item][0, i]/scaling, xerr=fwhm/2.,
                                             yerr=boxitem.flux[item][1, i]/scaling, zorder=3, **plot_kwargs[j][item][i])

                        else:
                            ax1.errorbar(wavelength, boxitem.flux[item][0]/scaling, xerr=fwhm/2.,
                                         yerr=boxitem.flux[item][1]/scaling, zorder=3, **plot_kwargs[j][item])

        elif isinstance(boxitem, box.SynphotBox):
            for i, find_item in enumerate(boxes):
                if isinstance(find_item, box.ObjectBox):
                    obj_index = i
                    break

            for item in boxitem.flux:
                transmission = read_filter.ReadFilter(item)
                wavelength = transmission.mean_wavelength()
                fwhm = transmission.filter_fwhm()

                if not plot_kwargs[obj_index] or item not in plot_kwargs[obj_index]:
                    ax1.errorbar(wavelength, boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                                 alpha=0.7, marker='s', ms=5, zorder=4, mfc='white')

                else:
                    if isinstance(plot_kwargs[obj_index][item], list):
                        # In case of multiple photometry values for the same filter, use the
                        # plot_kwargs of the first data point

                        kwargs_copy = plot_kwargs[obj_index][item][0].copy()

                        if 'label' in kwargs_copy:
                            del kwargs_copy['label']

                        ax1.errorbar(wavelength, boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                                     zorder=4, mfc='white', **kwargs_copy)

                    else:
                        kwargs_copy = plot_kwargs[obj_index][item].copy()

                        if 'label' in kwargs_copy:
                            del kwargs_copy['label']

                        ax1.errorbar(wavelength, boxitem.flux[item]/scaling, xerr=fwhm/2., yerr=None,
                                     zorder=4, mfc='white', **kwargs_copy)

    if filters is not None:
        for i, item in enumerate(filters):
            transmission = read_filter.ReadFilter(item)
            data = transmission.get_filter()

            ax2.plot(data[:, 0], data[:, 1], '-', lw=0.7, color='black', zorder=1)

    if residuals is not None:
        for i, find_item in enumerate(boxes):
            if isinstance(find_item, box.ObjectBox):
                obj_index = i
                break

        res_max = 0.

        if residuals.photometry is not None:
            for item in residuals.photometry:
                if not plot_kwargs[obj_index] or item not in plot_kwargs[obj_index]:
                    ax3.plot(residuals.photometry[item][0], residuals.photometry[item][1], marker='s',
                             ms=5, linestyle='none', zorder=2)

                else:
                    if residuals.photometry[item].ndim == 1:
                        ax3.plot(residuals.photometry[item][0], residuals.photometry[item][1], zorder=2,
                                 **plot_kwargs[obj_index][item])

                    elif residuals.photometry[item].ndim == 2:
                        for i in range(residuals.photometry[item].shape[1]):
                            if isinstance(plot_kwargs[obj_index][item], list):
                                ax3.plot(residuals.photometry[item][0, i], residuals.photometry[item][1, i], zorder=2,
                                         **plot_kwargs[obj_index][item][i])

                            else:
                                ax3.plot(residuals.photometry[item][0, i], residuals.photometry[item][1, i], zorder=2,
                                         **plot_kwargs[obj_index][item])

                res_max = np.nanmax(np.abs(residuals.photometry[item][1]))

        if residuals.spectrum is not None:
            for key, value in residuals.spectrum.items():
                if not plot_kwargs[obj_index] or key not in plot_kwargs[obj_index]:
                    ax3.plot(value[:, 0], value[:, 1], marker='o', ms=2, ls='none', zorder=1)

                else:
                    ax3.plot(value[:, 0], value[:, 1], zorder=1, **plot_kwargs[obj_index][key])

                max_tmp = np.nanmax(np.abs(value[:, 1]))

                if max_tmp > res_max:
                    res_max = max_tmp

        res_lim = math.ceil(1.1*res_max)

        if res_lim > 10.:
            res_lim = 5.

        ax3.axhline(0.0, linestyle='--', color='gray', dashes=(2, 4), zorder=0.5)

        if ylim_res is None:
            ax3.set_ylim(-res_lim, res_lim)
        else:
            ax3.set_ylim(ylim_res[0], ylim_res[1])

    if filters is not None:
        ax2.set_ylim(0., 1.1)

    print(f'Plotting spectrum: {output}...', end='', flush=True)

    if title is not None:
        if filters:
            ax2.set_title(title, y=1.02, fontsize=15)
        else:
            ax1.set_title(title, y=1.02, fontsize=15)

    handles, labels = ax1.get_legend_handles_labels()

    if handles and legend is not None:
        if isinstance(legend, list):
            model_handles = []
            data_handles = []

            model_labels = []
            data_labels = []

            for i, item in enumerate(handles):
                if isinstance(item, mpl.lines.Line2D):
                    model_handles.append(item)
                    model_labels.append(labels[i])

                elif isinstance(item, mpl.container.ErrorbarContainer):
                    data_handles.append(item)
                    data_labels.append(labels[i])

                else:
                    warnings.warn(f'The object type {item} is not implemented for the legend.')

            if legend[0] is not None:
                if isinstance(legend[0], (str, tuple)):
                    leg_1 = ax1.legend(model_handles, model_labels, loc=legend[0], fontsize=10., frameon=False)
                else:
                    leg_1 = ax1.legend(model_handles, model_labels, **legend[0])

            else:
                leg_1 = None

            if legend[1] is not None:
                if isinstance(legend[1], (str, tuple)):
                    leg_2 = ax1.legend(data_handles, data_labels, loc=legend[1], fontsize=8, frameon=False)
                else:
                    leg_2 = ax1.legend(data_handles, data_labels, **legend[1])

            if leg_1 is not None:
                ax1.add_artist(leg_1)

        elif isinstance(legend, (str, tuple)):
            ax1.legend(loc=legend, fontsize=8, frameon=False)

        else:
            ax1.legend(**legend)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')
