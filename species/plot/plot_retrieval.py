"""
Module for plotting atmospheric retrieval results.
"""

import warnings

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from typeguard import typechecked

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
                    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
                    extra_axis: str = 'photosphere') -> None:
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
    extra_axis : str, None
        The quantify that is plotted at the top axis ('photosphere', 'grains', None).

    Returns
    -------
    NoneType
        None
    """

    print(f'Plotting the P-T profiles: {output}...', end='', flush=True)

    cloud_species = ['Fe(c)', 'MgSio3(c)', 'Al2O3(c)', 'Na2S(c)', 'KCl(c)']
    cloud_check = ['fe', 'mgsio3', 'al2o3', 'na2s', 'kcl']

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

    top = True

    for item in cloud_check:
        if f'{item}_fraction' in median:
            top = False

        elif f'{item}_tau' in median:
            top = False

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

    # Create the pressure levels (bar)
    pressure = np.logspace(-6, 3, 180)

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

            temp, _ = retrieval_util.pt_ret_model(
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
        temp, _= retrieval_util.pt_ret_model(
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

        # Import interpol_abundances here because it is slow

        from poor_mans_nonequ_chem_FeH.poor_mans_nonequ_chem.poor_mans_nonequ_chem import \
            interpol_abundances

        abund_in = interpol_abundances(np.full(pressure.shape[0], median['c_o_ratio']),
                                       np.full(pressure.shape[0], median['metallicity']),
                                       temp,
                                       pressure,
                                       Pquench_carbon=quench_press)

        for item in cloud_species:
            if f'{item[:-3].lower()}_tau' in median:
                # Calculate the scaled mass fraction of the clouds
                median[f'{item[:-3].lower()}_fraction'] = retrieval_util.scale_cloud_abund(
                    median, radtrans.rt_object, pressure, temp, abund_in['MMW'], 'equilibrium',
                    abund_in, item, median[f'{item[:-3].lower()}_tau'],
                    pressure_grid=radtrans.pressure_grid)

        if 'fe_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_Fe_comb(median['metallicity'],
                                                                       median['c_o_ratio'],
                                                                       MMW=np.mean(abund_in['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:blue', zorder=2)

        if 'mgsio3_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_MgSiO3(median['metallicity'],
                                                                      median['c_o_ratio'],
                                                                      MMW=np.mean(abund_in['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:orange', zorder=2)

        if 'al2o3_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_Al2O3(median['metallicity'],
                                                                     median['c_o_ratio'],
                                                                     MMW=np.mean(abund_in['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:green', zorder=2)

        if 'na2s_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_Na2S(median['metallicity'],
                                                                    median['c_o_ratio'],
                                                                    MMW=np.mean(abund_in['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:cyan', zorder=2)

        if 'kcl_fraction' in median:
            sat_press, sat_temp = retrieval_util.return_T_cond_KCl(median['metallicity'],
                                                                   median['c_o_ratio'],
                                                                   MMW=np.mean(abund_in['MMW']))

            ax.plot(sat_temp, sat_press, '--', lw=0.8, color='tab:pink', zorder=2)

    if radtrans is not None:

        # Recalculate the best-fit model to update the attributes of radtrans.rt_object
        radtrans.get_model(median)

        if extra_axis == 'photosphere':
            radtrans.rt_object.calc_opt_depth(10.**median['logg'])
            radtrans.rt_object.calc_tau_cloud(10.**median['logg'])

            wavelength = radtrans.rt_object.lambda_angstroem*1e-4  # (um)

            # From Paul: The first axis of total_tay is the coordinate of the cumulative opacity
            # distribution function (ranging from 0 to 1). A correct average is obtained by
            # multiplying the first axis with self.w_gauss, then summing them. This is then the
            # actual wavelength-mean.

            # From petitRADTRANS: Only use 0 index for species because for lbl or
            # test_ck_shuffle_comp = True everything has been moved into the 0th index

            # Extract the optical depth of the line species
            w_gauss = radtrans.rt_object.w_gauss[..., np.newaxis, np.newaxis]
            optical_depth = np.sum(w_gauss*radtrans.rt_object.total_tau[:, :, 0, :], axis=0)

            # Add the optical depth of the cloud species
            # TODO is this correct?
            optical_depth += np.sum(radtrans.rt_object.tau_cloud[0, :, :, :], axis=1)

            if radtrans.rt_object.tau_cloud.shape[0] != 1:
                raise ValueError(f'Unexpected shape? {radtrans.rt_object.tau_cloud.shape}.')

            if radtrans.rt_object.tau_cloud.shape[2] != 1:
                raise ValueError(f'Unexpected shape? {radtrans.rt_object.tau_cloud.shape}.')

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

            ax2.set_yscale('log')

            ax2.set_xlabel('Wavelength (µm)', fontsize=13, va='bottom')

            if offset is not None:
                ax2.get_xaxis().set_label_coords(0.5, 1.+abs(offset[0]))
            else:
                ax2.get_xaxis().set_label_coords(0.5, 1.06)

            photo_press = np.zeros(wavelength.shape[0])

            for i in range(photo_press.shape[0]):
                press_interp = interp1d(optical_depth[i, :], radtrans.rt_object.press)
                photo_press[i] = press_interp(1.)*1e-6  # cgs to (bar)

            ax2.plot(wavelength, photo_press, lw=0.5, color='tab:blue')

        elif extra_axis == 'grains':

            if 'fe_fraction' in median or 'mgsio3_fraction' in median or 'al2o3_fraction' in median \
                    or 'na2s_fraction' in median or 'kcl_fraction' in median:
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
                cloud_index = radtrans.rt_object.cloud_species.index('Fe(c)')

                # Convert from (cm) to (um)
                ax2.plot(radtrans.rt_object.r_g[:, cloud_index]*1e4, pressure[::3],
                         lw=0.8, color='tab:blue')

            if 'mgsio3_fraction' in median:
                cloud_index = radtrans.rt_object.cloud_species.index('MgSiO3(c)')

                # Convert from (cm) to (um)
                ax2.plot(radtrans.rt_object.r_g[:, cloud_index]*1e4, pressure[::3],
                         lw=0.8, color='tab:orange')

            if 'al2o3_fraction' in median:
                cloud_index = radtrans.rt_object.cloud_species.index('Al2O3(c)')

                # Convert from (cm) to (um)
                ax2.plot(radtrans.rt_object.r_g[:, cloud_index]*1e4, pressure[::3],
                         lw=0.8, color='tab:green')

            if 'na2s_fraction' in median:
                cloud_index = radtrans.rt_object.cloud_species.index('Na2S(c)')

                # Convert from (cm) to (um)
                ax2.plot(radtrans.rt_object.r_g[:, cloud_index]*1e4, pressure[::3],
                         lw=0.8, color='tab:cyan')

            if 'kcl_fraction' in median:
                cloud_index = radtrans.rt_object.cloud_species.index('KCl(c)')

                # Convert from (cm) to (um)
                ax2.plot(radtrans.rt_object.r_g[:, cloud_index]*1e4, pressure[::3],
                         lw=0.8, color='tab:pink')

    else:
        if extra_axis is not None:
            warnings.warn('The argument of extra_axis is ignored because radtrans does not '
                          'contain a ReadRadtrans object.')

    plt.savefig(output, bbox_inches='tight')
    plt.clf()
    plt.close()

    print(' [DONE]')
