"""
Module with functions for making plots.
"""

import os
import sys

import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from species.data import database
from species.plot import util


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def plot_walkers(tag,
                 output,
                 nsteps=None):
    """
    :param box:
    :type box: species.core.box.SamplesBox

    :return: None
    """

    sys.stdout.write('Plotting walkers: '+output+'...')
    sys.stdout.flush()

    species_db = database.Database()
    box = species_db.get_samples(tag)

    samples = box.samples
    labels = util.update_labels(box.parameters)

    plt.figure(1, figsize=(6, 5))
    gridsp = mpl.gridspec.GridSpec(4, 1)
    gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[0, 0])
    ax2 = plt.subplot(gridsp[1, 0])
    ax3 = plt.subplot(gridsp[2, 0])
    ax4 = plt.subplot(gridsp[3, 0])

    ax1.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4))
    ax2.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4))
    ax3.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4))
    ax4.grid(True, linestyle=':', linewidth=0.7, color='silver', dashes=(1, 4))

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=False)

    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=False)

    ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=False)

    ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=False)

    ax3.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=False)

    ax3.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=False)

    ax4.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=5, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=True)

    ax4.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                    direction='in', width=0.8, length=3, labelsize=12, top=True,
                    bottom=True, left=True, right=True, labelbottom=True)

    ax4.set_xlabel('Step number', fontsize=10)

    ax1.set_ylabel(labels[0], fontsize=10)
    ax2.set_ylabel(labels[1], fontsize=10)
    ax3.set_ylabel(labels[2], fontsize=10)
    ax4.set_ylabel(labels[3], fontsize=10)

    if nsteps:
        ax1.set_xlim(0, nsteps)
        ax2.set_xlim(0, nsteps)
        ax3.set_xlim(0, nsteps)
        ax4.set_xlim(0, nsteps)

    ax4.get_xaxis().set_label_coords(0.5, -0.22)

    ax1.get_yaxis().set_label_coords(-0.09, 0.5)
    ax2.get_yaxis().set_label_coords(-0.09, 0.5)
    ax3.get_yaxis().set_label_coords(-0.09, 0.5)
    ax4.get_yaxis().set_label_coords(-0.09, 0.5)

    for i in range(samples.shape[0]):
        ax1.plot(samples[i, :, 0], ls='-', lw=0.5, color="black", alpha=0.5)
        ax2.plot(samples[i, :, 1], ls='-', lw=0.5, color="black", alpha=0.5)
        ax3.plot(samples[i, :, 2], ls='-', lw=0.5, color="black", alpha=0.5)
        ax4.plot(samples[i, :, 3], ls='-', lw=0.5, color="black", alpha=0.5)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()


def plot_posterior(tag,
                   burnin,
                   title,
                   output):
    """
    :param box:
    :type box: species.core.box.SamplesBox

    :return: None
    """

    sys.stdout.write('Plotting posteriors: '+output+'...')
    sys.stdout.flush()

    species_db = database.Database()
    box = species_db.get_samples(tag)

    samples = box.samples
    labels = util.update_labels(box.parameters)

    ndim = samples.shape[2]

    samples = samples[:, int(burnin):, :].reshape((-1, ndim))

    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        label_kwargs={"fontsize": 13}, show_titles=True,
                        title_kwargs={"fontsize": 12}, title_fmt='.2f')

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]

            ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                           direction='in', width=0.8, length=5, labelsize=12, top=True,
                           bottom=True, left=True, right=True)

            ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                           direction='in', width=0.8, length=3, labelsize=12, top=True,
                           bottom=True, left=True, right=True)

            ax.get_xaxis().set_label_coords(0.5, -0.26)
            ax.get_yaxis().set_label_coords(-0.27, 0.5)

    par_val = box.chisquare

    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(par_val[i], color="tomato")

        for j in range(i+1, ndim):
            ax = axes[j, i]
            ax.axvline(par_val[i], color="tomato")
            ax.axhline(par_val[j], color="tomato")
            ax.plot(par_val[i], par_val[j], "s", color="tomato")

    fig.suptitle(title, y=1.02, fontsize=16)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()
