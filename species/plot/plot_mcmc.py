"""
Module for plotting MCMC results.
"""

import os
import sys

import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from species.data import database
from species.util import plot_util


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def plot_walkers(tag,
                 output,
                 nsteps=None,
                 offset=None):
    """
    Function to plot the step history of the walkers.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    output : str
        Output filename.
    nsteps : int
        Number of steps.
    offset : tuple(float, float)
        Offset of the x- and y-axis label.

    Returns
    -------
    None
    """

    sys.stdout.write('Plotting walkers: '+output+'...')
    sys.stdout.flush()

    species_db = database.Database()
    box = species_db.get_samples(tag)

    samples = box.samples
    labels = plot_util.update_labels(box.parameters)

    ndim = samples.shape[-1]

    plt.figure(1, figsize=(6, 5))
    gridsp = mpl.gridspec.GridSpec(4, 1)
    gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

    for i in range(ndim):
        ax = plt.subplot(gridsp[i, 0])

        ax.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4))

        if i == ndim-1:
            ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                           direction='in', width=0.8, length=5, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=True)

            ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                           direction='in', width=0.8, length=3, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=True)

        else:
            ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                           direction='in', width=0.8, length=5, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=False)

            ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                           direction='in', width=0.8, length=3, labelsize=12, top=True,
                           bottom=True, left=True, right=True, labelbottom=False)

        if i == ndim-1:
            ax.set_xlabel('Step number', fontsize=10)
        else:
            ax.set_xlabel('', fontsize=10)

        ax.set_ylabel(labels[i], fontsize=10)

        if offset:
            ax.get_xaxis().set_label_coords(0.5, offset[0])
            ax.get_yaxis().set_label_coords(offset[1], 0.5)
        else:
            ax.get_xaxis().set_label_coords(0.5, -0.22)
            ax.get_yaxis().set_label_coords(-0.09, 0.5)

        if nsteps:
            ax.set_xlim(0, nsteps)

        for j in range(samples.shape[0]):
            ax.plot(samples[j, :, i], ls='-', lw=0.5, color='black', alpha=0.5)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()


def plot_posterior(tag,
                   burnin,
                   output,
                   title=None,
                   offset=None,
                   title_fmt='.2f',
                   limits=None):
    """
    Function to plot the posterior distributions.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    burnin : int
        Number of burnin steps to exclude.
    output : str
        Output filename.
    title : str
        Plot title.
    offset : tuple(float, float)
        Offset of the x- and y-axis label.
    title_fmt : str
        Format of the median and error values.
    limits : tuple(tuple(float, float), )
        Axis limits of all parameters. Automatically set if set to None.

    Returns
    -------
    None
    """

    sys.stdout.write('Plotting posteriors: '+output+'...')
    sys.stdout.flush()

    species_db = database.Database()
    box = species_db.get_samples(tag)

    samples = box.samples
    par_val = box.best_sample

    labels = plot_util.update_labels(box.parameters)

    ndim = samples.shape[-1]

    samples = samples[:, int(burnin):, :].reshape((-1, ndim))

    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        label_kwargs={'fontsize': 13}, show_titles=True,
                        title_kwargs={'fontsize': 12}, title_fmt=title_fmt)

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        for j in range(ndim):
            if i >= j:
                ax = axes[i, j]

                ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                               direction='in', width=0.8, length=5, labelsize=12, top=True,
                               bottom=True, left=True, right=True)

                ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                               direction='in', width=0.8, length=3, labelsize=12, top=True,
                               bottom=True, left=True, right=True)

                if limits is not None:
                    ax.set_xlim(limits[j])

                ax.axvline(par_val[j], color='tomato')

                if i > j:
                    ax.axhline(par_val[i], color='tomato')
                    ax.plot(par_val[j], par_val[i], 's', color='tomato')

                    if limits is not None:
                        ax.set_ylim(limits[i])

        if i >= j:
            if offset:
                ax.get_xaxis().set_label_coords(0.5, offset[0])
                ax.get_yaxis().set_label_coords(offset[1], 0.5)
            else:
                ax.get_xaxis().set_label_coords(0.5, -0.26)
                ax.get_yaxis().set_label_coords(-0.27, 0.5)

    if title:
        fig.suptitle(title, y=1.02, fontsize=16)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()


def plot_photometry(tag,
                    filter_id,
                    burnin,
                    output,
                    xlim=None):
    """
    Function to plot the posterior distribution of the synthetic photometry.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    filter_id : str
        Filter ID.
    burnin : int
        Number of burnin steps to exclude.
    output : str
        Output filename.
    xlim : tuple(float, float)
        Axis limits. Automatically set if set to None.

    Returns
    -------
    None
    """

    species_db = database.Database()

    samples = species_db.get_mcmc_photometry(tag, burnin, filter_id)

    sys.stdout.write('Plotting photometry samples: '+output+'...')
    sys.stdout.flush()

    fig = corner.corner(samples, labels=['Magnitude'], quantiles=[0.16, 0.5, 0.84],
                        label_kwargs={'fontsize': 13}, show_titles=True,
                        title_kwargs={'fontsize': 12}, title_fmt='.2f')

    axes = np.array(fig.axes).reshape((1, 1))

    ax = axes[0, 0]

    ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                   direction='in', width=0.8, length=5, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                   direction='in', width=0.8, length=3, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.get_xaxis().set_label_coords(0.5, -0.26)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()
