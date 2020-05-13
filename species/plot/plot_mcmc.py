"""
Module for plotting MCMC results.
"""

import os

from typing import Optional, Tuple

import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import ScalarFormatter

from species.core import constants
from species.data import database
from species.util import plot_util


@typechecked
def plot_walkers(tag: str,
                 nsteps: Optional[int] = None,
                 offset: Optional[Tuple[float, float]] = None,
                 output: str = 'walkers.pdf') -> None:
    """
    Function to plot the step history of the walkers.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    nsteps : int, None
        Number of steps that are plotted. All steps are plotted if set to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values are used if if set to ``None``.
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    print(f'Plotting walkers: {output}...', end='', flush=True)

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.2)

    species_db = database.Database()
    box = species_db.get_samples(tag)

    samples = box.samples
    labels = plot_util.update_labels(box.parameters)

    ndim = samples.shape[-1]

    plt.figure(1, figsize=(6, ndim*1.5))
    gridsp = mpl.gridspec.GridSpec(ndim, 1)
    gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

    for i in range(ndim):
        ax = plt.subplot(gridsp[i, 0])

        if i == ndim-1:
            ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                           direction='in', width=1, length=5, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=True)

            ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                           direction='in', width=1, length=3, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=True)

        else:
            ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                           direction='in', width=1, length=5, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=False)

            ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                           direction='in', width=1, length=3, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=False)

        if i == ndim-1:
            ax.set_xlabel('Step number', fontsize=10)
        else:
            ax.set_xlabel('', fontsize=10)

        ax.set_ylabel(labels[i], fontsize=10)

        if offset is not None:
            ax.get_xaxis().set_label_coords(0.5, offset[0])
            ax.get_yaxis().set_label_coords(offset[1], 0.5)

        else:
            ax.get_xaxis().set_label_coords(0.5, -0.22)
            ax.get_yaxis().set_label_coords(-0.09, 0.5)

        if nsteps is not None:
            ax.set_xlim(0, nsteps)

        for j in range(samples.shape[0]):
            ax.plot(samples[j, :, i], ls='-', lw=0.5, color='black', alpha=0.5)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')


@typechecked
def plot_posterior(tag: str,
                   burnin: Optional[int] = None,
                   title: Optional[str] = None,
                   offset: Optional[Tuple[float, float]] = None,
                   title_fmt: str = '.2f',
                   limits: Optional[Tuple[Tuple[float, float]]] = None,
                   max_posterior: bool = False,
                   inc_luminosity: bool = False,
                   output: str = 'posterior.pdf') -> None:
    """
    Function to plot the posterior distribution.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    burnin : int, None
        Number of burnin steps to exclude. All samples are used if set to ``None``.
    title : str, None
        Plot title. No title is shown if set to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values are used if set to ``None``.
    title_fmt : str
        Format of the median and error values.
    limits : tuple(tuple(float, float), ), None
        Axis limits of all parameters. Automatically set if set to ``None``.
    max_posterior : bool
        Plot the position of the sample with the maximum posterior probability.
    inc_luminosity : bool
        Include the log10 of the luminosity in the posterior plot as calculated from the
        effective temperature and radius.
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

    if burnin is None:
        burnin = 0

    species_db = database.Database()
    box = species_db.get_samples(tag, burnin=burnin)

    print(f'Median sample:')
    for key, value in box.median_sample.items():
        print(f'   - {key} = {value:.2f}')

    samples = box.samples
    ndim = samples.shape[-1]

    if box.prob_sample is not None:
        par_val = tuple(box.prob_sample.values())

        print(f'Maximum posterior sample:')
        for key, value in box.prob_sample.items():
            print(f'   - {key} = {value:.2f}')

    print(f'Plotting the posterior: {output}...', end='', flush=True)

    if inc_luminosity:
        ndim += 1

        if 'teff' in box.parameters and 'radius' in box.parameters:
            teff_index = np.argwhere(np.array(box.parameters) == 'teff')[0]
            radius_index = np.argwhere(np.array(box.parameters) == 'radius')[0]

            luminosity = 4. * np.pi * (samples[..., radius_index]*constants.R_JUP)**2 * \
                constants.SIGMA_SB * samples[..., teff_index]**4. / constants.L_SUN

            samples = np.append(samples, np.log10(luminosity), axis=-1)
            box.parameters.append('luminosity')

        elif 'teff_0' in box.parameters and 'radius_0' in box.parameters:
            luminosity = 0.

            for i in range(100):
                teff_index = np.argwhere(np.array(box.parameters) == f'teff_{i}')
                radius_index = np.argwhere(np.array(box.parameters) == f'radius_{i}')

                if len(teff_index) > 0 and len(radius_index) > 0:
                    luminosity += 4. * np.pi * (samples[..., radius_index[0]]*constants.R_JUP)**2 \
                        * constants.SIGMA_SB * samples[..., teff_index[0]]**4. / constants.L_SUN

                else:
                    break

            samples = np.append(samples, np.log10(luminosity), axis=-1)
            box.parameters.append('luminosity')

    labels = plot_util.update_labels(box.parameters)

    samples = samples.reshape((-1, ndim))

    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        label_kwargs={'fontsize': 13}, show_titles=True,
                        title_kwargs={'fontsize': 12}, title_fmt=title_fmt)

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        for j in range(ndim):
            if i >= j:
                ax = axes[i, j]

                ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

                if j == 0 and i != 0:
                    labelleft = True
                else:
                    labelleft = False

                if i == ndim-1:
                    labelbottom = True
                else:
                    labelbottom = False

                ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                               direction='in', width=1, length=5, labelsize=12, top=True,
                               bottom=True, left=True, right=True, labelleft=labelleft,
                               labelbottom=labelbottom, labelright=False, labeltop=False)

                ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                               direction='in', width=1, length=3, labelsize=12, top=True,
                               bottom=True, left=True, right=True, labelleft=labelleft,
                               labelbottom=labelbottom, labelright=False, labeltop=False)

                if limits is not None:
                    ax.set_xlim(limits[j])

                if max_posterior:
                    ax.axvline(par_val[j], color='tomato')

                if i > j:
                    if max_posterior:
                        ax.axhline(par_val[i], color='tomato')
                        ax.plot(par_val[j], par_val[i], 's', color='tomato')

                    if limits is not None:
                        ax.set_ylim(limits[i])

                if offset is not None:
                    ax.get_xaxis().set_label_coords(0.5, offset[0])
                    ax.get_yaxis().set_label_coords(offset[1], 0.5)

                else:
                    ax.get_xaxis().set_label_coords(0.5, -0.26)
                    ax.get_yaxis().set_label_coords(-0.27, 0.5)

    if title:
        fig.suptitle(title, y=1.02, fontsize=16)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')


def plot_photometry(tag,
                    filter_id,
                    burnin=None,
                    xlim=None,
                    output='photometry.pdf'):
    """
    Function to plot the posterior distribution of the synthetic photometry.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    filter_id : str
        Filter ID.
    burnin : int, None
        Number of burnin steps to exclude. All samples are used if set to None.
    xlim : tuple(float, float), None
        Axis limits. Automatically set if set to None.
    output : strr
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.2)

    species_db = database.Database()

    samples = species_db.get_mcmc_photometry(tag, burnin, filter_id)

    print(f'Plotting photometry samples: {output}...', end='', flush=True)

    fig = corner.corner(samples, labels=['Magnitude'], quantiles=[0.16, 0.5, 0.84],
                        label_kwargs={'fontsize': 13}, show_titles=True,
                        title_kwargs={'fontsize': 12}, title_fmt='.2f')

    axes = np.array(fig.axes).reshape((1, 1))

    ax = axes[0, 0]

    ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                   direction='in', width=1, length=5, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                   direction='in', width=1, length=3, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.get_xaxis().set_label_coords(0.5, -0.26)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')
