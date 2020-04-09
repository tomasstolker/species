"""
Module for plotting MCMC results.
"""

import os

import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter

from species.data import database
from species.util import plot_util, retrieval_util


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2.2)


def plot_walkers(tag,
                 nsteps=None,
                 offset=None,
                 output='walkers.pdf'):
    """
    Function to plot the step history of the walkers.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    nsteps : int
        Number of steps.
    offset : tuple(float, float)
        Offset of the x- and y-axis label.
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    print(f'Plotting walkers: {output}...', end='', flush=True)

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


def plot_posterior(tag,
                   burnin=None,
                   title=None,
                   offset=None,
                   title_fmt='.2f',
                   limits=None,
                   max_prob=False,
                   vmr=False,
                   output='posterior.pdf'):
    """
    Function to plot the posterior distribution of the fitted parameters.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    burnin : int, None
        Number of burnin steps to exclude. All samples are used if set to None.
    title : str, None
        Plot title.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label.
    title_fmt : str
        Format of the median and error values.
    limits : tuple(tuple(float, float), ), None
        Axis limits of all parameters. Automatically set if set to None.
    max_prob : bool
        Plot the position of the sample with the maximum posterior probability.
    vmr : bool
        Plot the volume mixing ratios (i.e. number fractions) instead of the mass fractions of the
        retrieved species with :class:`~species.analysis.retrieval.AtmosphericRetrieval`.
    output : str
        Output filename.

    Returns
    -------
    NoneType
        None
    """

    species_db = database.Database()

    samples_box = species_db.get_samples(tag, burnin=burnin)
    samples = samples_box.samples

    if 'CO' or 'CO_all_iso' in samples_box.parameters:
        samples_box.parameters.append('c_h_ratio')
        samples_box.parameters.append('o_h_ratio')

        abund_index = {}
        for i, item in enumerate(samples_box.parameters):
            if item == 'CO':
                abund_index['CO'] = i

            elif item == 'CO_all_iso':
                abund_index['CO_all_iso'] = i

            elif item == 'CO2':
                abund_index['CO2'] = i

            elif item == 'CH4':
                abund_index['CH4'] = i

            elif item == 'H2O':
                abund_index['H2O'] = i

            elif item == 'NH3':
                abund_index['NH3'] = i

            elif item == 'H2S':
                abund_index['H2S'] = i

        c_h_ratio = np.zeros(samples.shape[0])
        o_h_ratio = np.zeros(samples.shape[0])

        for i, item in enumerate(samples):
            abund = {}

            if 'CO' in samples_box.parameters:
                abund['CO'] = item[abund_index['CO']]

            if 'CO_all_iso' in samples_box.parameters:
                abund['CO_all_iso'] = item[abund_index['CO_all_iso']]

            if 'CO2' in samples_box.parameters:
                abund['CO2'] = item[abund_index['CO2']]

            if 'CH4' in samples_box.parameters:
                abund['CH4'] = item[abund_index['CH4']]

            if 'H2O' in samples_box.parameters:
                abund['H2O'] = item[abund_index['H2O']]

            if 'NH3' in samples_box.parameters:
                abund['NH3'] = item[abund_index['NH3']]

            if 'H2S' in samples_box.parameters:
                abund['H2S'] = item[abund_index['H2S']]

            c_h_ratio[i], o_h_ratio[i] = retrieval_util.calc_metal_ratio(abund)

    if vmr and samples_box.spectrum == 'petitradtrans':
        print('Changing mass fractions to number fractions...', end='', flush=True)

        # get all available line species
        line_species = retrieval_util.get_line_species()

        # get the atommic and molecular masses
        masses = retrieval_util.atomic_masses()

        # creates array for the updated samples
        updated_samples = np.zeros(samples.shape)

        for i, samples_item in enumerate(samples_box.samples):
            # initiate a dictionary for the log10 mass fraction of the metals
            log_x_abund = {}

            for param_item in samples_box.parameters:
                if param_item in line_species:
                    # get the index of the parameter
                    param_index = samples_box.parameters.index(param_item)

                    # store log10 mass fraction in the dictionary
                    log_x_abund[param_item] = samples_item[param_index]

            # create a dictionary with all mass fractions, including H2 and He
            x_abund = retrieval_util.mass_fractions(log_x_abund)

            # calculate the mean molecular weight from the input mass fractions
            mmw = retrieval_util.mean_molecular_weight(x_abund)

            for param_item in samples_box.parameters:
                if param_item in line_species:
                    # get the index of the parameter
                    param_index = samples_box.parameters.index(param_item)

                    # overwrite the sample with the log10 number fraction
                    samples_item[param_index] = np.log10(10.**samples_item[param_index] * mmw/masses[param_item])

            # store the updated sample to the array
            updated_samples[i, ] = samples_item

        # overwrite the samples in the SamplesBox
        samples_box.samples = updated_samples

        print(' [DONE]')

    print(f'Median sample:')
    for key, value in samples_box.median_sample.items():
        print(f'   - {key} = {value:.2f}')

    if samples_box.prob_sample is not None:
        par_val = tuple(samples_box.prob_sample.values())

        print(f'Maximum posterior sample:')
        for key, value in samples_box.prob_sample.items():
            print(f'   - {key} = {value:.2f}')

    print(f'Plotting the posterior: {output}...', end='', flush=True)

    labels = plot_util.update_labels(samples_box.parameters)

    ndim = len(samples_box.parameters)

    samples = np.column_stack((samples, c_h_ratio, o_h_ratio))

    if samples.ndim == 3:
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

                if max_prob:
                    ax.axvline(par_val[j], color='tomato')

                if i > j:
                    if max_prob:
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
