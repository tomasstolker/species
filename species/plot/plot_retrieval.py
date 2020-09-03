"""
Module for plotting atmospheric retrieval results.
"""

from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from petitRADTRANS import nat_cst as nc
from poor_mans_nonequ_chem_FeH.poor_mans_nonequ_chem.poor_mans_nonequ_chem import \
    interpol_abundances

from species.data import database
from species.read import read_radtrans
from species.util import retrieval_util


@typechecked
def plot_pt_profile(tag: str,
                    random: int = 100,
                    xlim: Optional[Tuple[float, float]] = None,
                    ylim: Optional[Tuple[float, float]] = None,
                    offset: Optional[Tuple[float, float]] = None,
                    output: str = 'pt_profile.pdf',
                    radtrans: Optional[read_radtrans.ReadRadtrans] = None) -> None:
    """
    Function to plot the posterior distribution.

    Parameters
    ----------
    tag : str
        Database tag with the posterior samples.
    random : int
        Number of randomly selected samples from the posterior.
    xlim : tuple(float, float), None
        Limits of the wavelength axis.
    ylim : tuple(float, float), None
        Limits of the flux axis.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label.
    output : str
        Output filename.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`. Only required with
        ``spectrum='petitradtrans'`. Make sure that the ``wavel_range`` of the ``ReadRadtrans``
        instance is sufficiently broad to cover all the photometric and spectroscopic data of
        ``inc_phot`` and ``inc_spec``. Not used if set to ``None``.

    Returns
    -------
    NoneType
        None
    """

    print(f'Plotting the P-T profiles: {output}...', end='', flush=True)

    species_db = database.Database()
    box = species_db.get_samples(tag, burnin=0)

    parameters = np.asarray(box.parameters)
    samples = box.samples
    median = box.median_sample

    # indices = np.argwhere(samples[:, 0] > 4.5)
    # indices = indices[:, 0]
    # samples = samples[indices, ]

    indices = np.random.randint(samples.shape[0], size=random)
    samples = samples[indices, ]

    mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
    mpl.rcParams['font.family'] = 'serif'

    plt.rc('axes', edgecolor='black', linewidth=2.5)

    plt.figure(1, figsize=(4., 5.))
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    if 'fe_fraction' in median or 'mgsio3_fraction' in median or 'al2o3_fraction' in median:
        top = False
    else:
        top = True

    ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                   direction='in', width=1, length=5, labelsize=12, top=top,
                   bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                   direction='in', width=1, length=3, labelsize=12, top=top,
                   bottom=True, left=True, right=True)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel('Pressure (bar)', fontsize=13)

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

    if 'tint' in parameters:
        pt_profile = 'molliere'

        tint_index = np.argwhere(parameters == 'tint')[0]
        t1_index = np.argwhere(parameters == 't1')[0]
        t2_index = np.argwhere(parameters == 't2')[0]
        t3_index = np.argwhere(parameters == 't3')[0]
        alpha_index = np.argwhere(parameters == 'alpha')[0]
        log_delta_index = np.argwhere(parameters == 'log_delta')[0]

    else:
        pt_profile = 'free'

        temp_index = []
        for i in range(15):
            temp_index.append(np.argwhere(parameters == f't{i}')[0])

        knot_press = np.logspace(np.log10(pressure[0]), np.log10(pressure[-1]), 15)

    for item in samples:
        if pt_profile == 'molliere':
            metallicity_index = np.argwhere(parameters == 'metallicity')[0]
            c_o_ratio_index = np.argwhere(parameters == 'c_o_ratio')[0]

            temp, _, _ = retrieval_util.pt_ret_model(
                np.array([item[t1_index][0], item[t2_index][0], item[t3_index][0]]),
                10.**item[log_delta_index][0], item[alpha_index][0], item[tint_index][0], pressure,
                item[metallicity_index][0], item[c_o_ratio_index][0])

        elif pt_profile == 'free':
            knot_temp = []
            for i in range(15):
                knot_temp.append(item[temp_index[i]][0])

            knot_temp = np.asarray(knot_temp)

            temp = retrieval_util.pt_spline_interp(knot_press, knot_temp, pressure)

        ax.plot(temp, pressure, '-', lw=0.3, color='gray', alpha=0.5, zorder=1)

    if pt_profile == 'molliere':
        temp, _, _ = retrieval_util.pt_ret_model(
            np.array([median['t1'], median['t2'], median['t3']]), 10.**median['log_delta'],
            median['alpha'], median['tint'], pressure, median['metallicity'], median['c_o_ratio'])

    elif pt_profile == 'free':
        knot_temp = []
        for i in range(15):
            knot_temp.append(median[f't{i}'])

        knot_temp = np.asarray(knot_temp)

        ax.plot(knot_temp, knot_press, 'o', ms=5., mew=0., color='tomato', zorder=3.)

        temp = retrieval_util.pt_spline_interp(knot_press, knot_temp, pressure)

    ax.plot(temp, pressure, '-', lw=1, color='black', zorder=2)

    if 'metallicity' in parameters and 'c_o_ratio' in parameters:
        if 'log_p_quench' in median:
            quench_press = 10.**median['log_p_quench']
        else:
            quench_press = None

        abund = interpol_abundances(np.full(pressure.shape[0], median['c_o_ratio']),
                                    np.full(pressure.shape[0], median['metallicity']),
                                    temp,
                                    pressure,
                                    Pquench_carbon=quench_press)

        if 'fe_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_Fe_comb(median['metallicity'],
                                                                       median['c_o_ratio'],
                                                                       MMW=np.mean(abund['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:blue', zorder=2)

        if 'mgsio3_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_MgSiO3(median['metallicity'],
                                                                      median['c_o_ratio'],
                                                                      MMW=np.mean(abund['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:orange', zorder=2)

        if 'al2o3_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_Al2O3(median['metallicity'],
                                                                     median['c_o_ratio'],
                                                                     MMW=np.mean(abund['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:green', zorder=2)

    if radtrans is not None:
        if 'fe_fraction' in median or 'mgsio3_fraction' in median or 'al2o3_fraction' in median:
            ax2 = ax.twiny()

            ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                            direction='in', width=1, length=5, labelsize=12, top=True,
                            bottom=False, left=True, right=True)

            ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                            direction='in', width=1, length=3, labelsize=12, top=True,
                            bottom=False, left=True, right=True)

            if ylim:
                ax2.set_ylim(ylim[0], ylim[1])
            else:
                ax2.set_ylim(1e3, 1e-6)

            ax2.set_xscale('log')
            ax2.set_yscale('log')

            ax2.set_xlabel('Average particle radius (µm)', fontsize=13, va='bottom')

            # Recalculate the best-fit model to update the r_g attribute of radtrans.rt_object
            radtrans.get_model(median)

            if offset is not None:
                ax2.get_xaxis().set_label_coords(0.5, 1.+abs(offset[0]))
            else:
                ax2.get_xaxis().set_label_coords(0.5, 1.06)

        if 'fe_fraction' in median:
            # Convert from (cm) to (um)
            ax2.plot(radtrans.rt_object.r_g[:, radtrans.rt_object.cloud_species.index('Fe(c)')]*1e4, pressure[::3], lw=0.8, color='tab:blue')

        if 'mgsio3_fraction' in median:
            # Convert from (cm) to (um)
            ax2.plot(radtrans.rt_object.r_g[:, radtrans.rt_object.cloud_species.index('MgSiO3(c)')]*1e4, pressure[::3], lw=0.8, color='tab:orange')

        if 'al2o3_fraction' in median:
            # Convert from (cm) to (um)
            ax2.plot(radtrans.rt_object.r_g[:, radtrans.rt_object.cloud_species.index('Al2O3(c)')]*1e4, pressure[::3], lw=0.8, color='tab:green')

    plt.savefig(output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')
