import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from petitRADTRANS_ck_test_speed import nat_cst as nc

from species.data import database
from species.util import retrieval_util


def plot_pt_profile(tag,
                    random=100,
                    xlim=None,
                    ylim=None,
                    offset=None,
                    output='pt_profile.pdf'):
    """
    Function to plot the posterior distribution.

    Parameters
    ----------
    tag : str
        Database tag with the MCMC samples.
    random : int
        Number of randomly selected samples from the posterior.
    xlim : tuple(float, float)
        Limits of the wavelength axis.
    ylim : tuple(float, float)
        Limits of the flux axis.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label.
    output : str
        Output filename.

    Returns
    -------
    None
    """

    print(f'Plotting the P-T profiles: {output}...', end='', flush=True)

    species_db = database.Database()
    box = species_db.get_samples(tag, burnin=0)

    parameters = np.asarray(box.parameters)
    samples = box.samples
    median = box.median_sample

    indices = np.random.randint(samples.shape[0], size=random)
    samples = samples[indices, ]

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.5)

    plt.figure(1, figsize=(4., 5.))
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                   direction='in', width=1, length=5, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                   direction='in', width=1, length=3, labelsize=12, top=True,
                   bottom=True, left=True, right=True)

    ax.set_xlabel('Temperature [K]', fontsize=13)
    ax.set_ylabel('Pressure [bar]', fontsize=13)

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(1000., 5000.)

    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(1e3, 1e-6)

    ax.set_yscale('log')

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.06)
        ax.get_yaxis().set_label_coords(-0.14, 0.5)

    # create pressure levels

    temp_params = {}
    temp_params['log_delta'] = -6.
    temp_params['log_gamma'] = 1.
    temp_params['t_int'] = 750.
    temp_params['t_equ'] = 0.
    temp_params['log_p_trans'] = -3.
    temp_params['alpha'] = 0.

    pressure, _ = nc.make_press_temp(temp_params)

    tint_index = np.argwhere(parameters == 'tint')[0]
    t1_index = np.argwhere(parameters == 't1')[0]
    t2_index = np.argwhere(parameters == 't2')[0]
    t3_index = np.argwhere(parameters == 't3')[0]
    alpha_index = np.argwhere(parameters == 'alpha')[0]
    log_delta_index = np.argwhere(parameters == 'log_delta')[0]
    feh_index = np.argwhere(parameters == 'feh')[0]
    co_index = np.argwhere(parameters == 'co')[0]

    for item in samples:
        temp, _, _ = retrieval_util.pt_ret_model(
            np.array([item[t1_index][0], item[t2_index][0], item[t3_index][0]]),
            10.**item[log_delta_index][0], item[alpha_index][0], item[tint_index][0], pressure,
            item[feh_index][0], item[co_index][0])

        ax.plot(temp, pressure, '-', lw=0.3, color='gray', alpha=0.5, zorder=1)

    temp, _, _ = retrieval_util.pt_ret_model(
        np.array([median['t1'], median['t2'], median['t3']]), 10.**median['log_delta'],
        median['alpha'], median['tint'], pressure, median['feh'], median['co'])

    ax.plot(temp, pressure, '-', lw=1, color='black', zorder=2)

    plt.savefig(output, bbox_inches='tight')

    print(' [DONE]')
