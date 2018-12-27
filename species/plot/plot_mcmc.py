"""
Module with functions for making plots.
"""

import os
import sys

import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
mpl.rcParams['font.family'] = 'serif'

plt.rc('axes', edgecolor='black', linewidth=2)


def update_labels(param):
    """
    :param param:
    :type param: list

    :return:
    :rtype: list
    """

    if 'teff' in param:
        index = param.index('teff')
        param[index] = r'$T_\mathregular{eff}$ [K]'

    if 'logg' in param:
        index = param.index('logg')
        param[index] = r'$\log\,g$'

    if 'feh' in param:
        index = param.index('feh')
        param[index] = r'[Fe/H]'

    if 'radius' in param:
        index = param.index('radius')
        param[index] = r'$R$ [$\mathregular{R_{Jup}}$]'

    return param


def plot_walkers(box,
                 output,
                 nsteps=None):
    """
    :param box:
    :type box: species.core.box.SamplesBox

    :return: None
    """

    sys.stdout.write('Plotting walkers: '+output+'...')
    sys.stdout.flush()

    samples = box.samples
    labels = update_labels(box.parameters)

    # ndim = samples.shape[2]

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

    ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=False)
    ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=False)
    ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=False)
    ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=False)
    ax3.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=False)
    ax3.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=False)
    ax4.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=True)
    ax4.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=True, labelbottom=True)

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


def plot_posterior(box,
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

    samples = box.samples
    labels = update_labels(box.parameters)

    ndim = samples.shape[2]

    samples = samples[:, int(burnin):, :].reshape((-1, ndim))

    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], label_kwargs={"fontsize": 13},
                        show_titles=True, title_kwargs={"fontsize": 13}, title_fmt='.2f')

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]

            ax.tick_params(axis='both', which='major', colors='black', labelcolor='black', direction='in', width=0.8, length=5, labelsize=12, top=True, bottom=True, left=True, right=True)
            ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black', direction='in', width=0.8, length=3, labelsize=12, top=True, bottom=True, left=True, right=True)

            ax.get_xaxis().set_label_coords(0.5, -0.26)
            ax.get_yaxis().set_label_coords(-0.27, 0.5)

    # for i in range(ndim):
    #     ax = axes[i, i]
    #     ax.axvline(simplex[i], color="tomato")
    #
    # ax = axes[1, 0]
    # ax.axvline(simplex[0], color="tomato")
    # ax.axhline(simplex[1], color="tomato")
    # ax.plot(simplex[0], simplex[1], "s", color="tomato")
    #
    # ax = axes[2, 0]
    # ax.axvline(simplex[0], color="tomato")
    # ax.axhline(simplex[2], color="tomato")
    # ax.plot(simplex[0], simplex[2], "s", color="tomato")
    #
    # ax = axes[2, 1]
    # ax.axvline(simplex[1], color="tomato")
    # ax.axhline(simplex[2], color="tomato")
    # ax.plot(simplex[1], simplex[2], "s", color="tomato")

    fig.suptitle(title, y=1.02, fontsize=16)

    plt.savefig(os.getcwd()+'/'+output, bbox_inches='tight')
    plt.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()
