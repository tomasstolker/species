"""
Utility functions for atmospheric retrieval.
"""

import copy

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
from scipy.ndimage.filters import gaussian_filter
from typeguard import typechecked

from petitRADTRANS.radtrans import Radtrans as Radtrans
from petitRADTRANS_ck_test_speed import nat_cst as nc
from petitRADTRANS_ck_test_speed.radtrans import Radtrans as RadtransScatter
from poor_mans_nonequ_chem_FeH.poor_mans_nonequ_chem.poor_mans_nonequ_chem import \
    interpol_abundances


@typechecked
def get_line_species() -> list:
    """
    Function to get the list of the molecular and atomic line species.

    Returns
    -------
    list
        List with the line species.
    """

    return ['CH4', 'CO', 'CO_all_iso', 'CO2', 'H2O', 'H2S', 'HCN', 'K', 'K_lor_cut', 'K_burrows',
            'NH3', 'Na', 'Na_lor_cut', 'Na_burrows', 'OH', 'PH3', 'TiO', 'VO', 'FeH']


@typechecked
def pt_ret_model(T3: np.ndarray,
                 delta: float,
                 alpha: float,
                 tint: float,
                 press: np.ndarray,
                 FeH: float,
                 CO: float,
                 conv: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Self-luminous retrieval P-T model.

    Parameters
    ----------
    T3 = np.array([t1, t2, t3]): temperature points to be added on top
      of the radiative Eddington structure (above tau = 0.1).
      Use spline interpolation, t1 < t2 < t3 < tconnect as prior.

    delta: proportionality factor in tau = delta * press_cgs**alpha

    alpha: power law index in tau = delta * press_cgs**alpha
       For the tau model: use proximity to kappa_rosseland photosphere
       as prior.

    tint: internal temperature of the Eddington model

    press: input pressure profile in bar

    FeH: metallicity for the nabla_ad interpolation

    CO: C/O for the nabla_ad interpolation

    conv: enforce convective adiabat yes/no

    Returns
    -------
    np.ndarray
    float
    np.ndarray
    """

    # Go grom bar to cgs
    press_cgs = press*1e6

    # Calculate the optical depth
    tau = delta*press_cgs**alpha

    # This is the eddington temperature
    tedd = (3./4.*tint**4.*(2./3.+tau))**0.25

    ab = interpol_abundances(CO*np.ones_like(tedd), FeH*np.ones_like(tedd), tedd, press)

    nabla_ad = ab['nabla_ad']

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = np.diff(np.log(tedd))/np.diff(np.log(press_cgs))

        # Extend to array of same length as pressure structure
        nabla_rad = np.ones_like(tedd)
        nabla_rad[0] = nab_rad[0]
        nabla_rad[-1] = nab_rad[-1]
        nabla_rad[1:-1] = (nab_rad[1:]+nab_rad[:-1])/2.

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        for i in range(10):

            if i == 0:
                t_take = copy.copy(tedd)
            else:
                t_take = copy.copy(tfinal)

            ab = interpol_abundances(CO*np.ones_like(t_take),
                                     FeH*np.ones_like(t_take),
                                     t_take,
                                     press)

            nabla_ad = ab['nabla_ad']

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean[1:] = (nabla_ad[1:]+nabla_ad[:-1])/2.

            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index]*np.mean(np.diff(np.log(press_cgs)))

            # What is the last radiative temperature?
            tstart = np.log(t_take[~conv_index][-1])

            # Integrate and translate to temperature from log(temperature)
            tnew = np.exp(np.cumsum(tnew)+tstart)

            # Add upper radiative and
            # lower conective part into one single array
            tfinal = copy.copy(t_take)
            tfinal[conv_index] = tnew

            if np.max(np.abs(t_take-tfinal)/t_take) < 0.01:
                # print('n_ad', 1./(1.-nabla_ad[conv_index]))
                break

    else:
        tfinal = tedd

    # Add the three temperature-point P-T description above tau = 0.1
    @typechecked
    def press_tau(tau: float) -> float:
        """
        Parameters
        ----------

        Returns
        -------

        """
        # Returns the pressure at a given tau, in cgs
        return (tau/delta)**(1./alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    for i_intp in range(2):

        if i_intp == 0:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]), np.log10(p_bot_spline), 4)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative-convective solution
            support_points_high = 1e1**np.arange(np.log10(p_bot_spline),
                                                 np.log10(press_cgs[-1]),
                                                 np.diff(np.log10(support_points_low))[0])

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low)+len(support_points_high)-1)
            support_points[:4] = support_points_low
            support_points[4:] = support_points_high[1:]

        else:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]), np.log10(p_bot_spline), 7)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative-convective solution
            support_points_high = np.logspace(np.log10(p_bot_spline), np.log10(press_cgs[-1]), 7)

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low)+len(support_points_high)-1)
            support_points[:7] = support_points_low
            support_points[7:] = support_points_high[1:]

        # Define the temperature values at the node points.
        t_support = np.zeros_like(support_points)

        if i_intp == 0:
            tfintp = interp1d(press_cgs, tfinal)

            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low))-1] = tfintp(p_bot_spline)

            # The temperature at pressures below p_bot_spline (free parameters)
            t_support[:(int(len(support_points_low))-1)] = T3
            # t_support[:3] = tfintp(support_points_low)

            # The temperature at pressures above p_bot_spline (from the
            # radiative-convectice solution)
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        else:
            tfintp1 = interp1d(press_cgs, tret)

            t_support[:(int(len(support_points_low))-1)] = \
                tfintp1(support_points[:(int(len(support_points_low))-1)])

            tfintp = interp1d(press_cgs, tfinal)

            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low))-1] = tfintp(p_bot_spline)

            # print('diff', t_connect_calc - tfintp(p_bot_spline))
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        # Make the temperature spline interpolation to be returned to the user
        # tret = spline(np.log10(support_points), t_support, np.log10(press_cgs), order = 3)

        cs = CubicSpline(np.log10(support_points), t_support)
        tret = cs(np.log10(press_cgs))

    # Return the temperature, the pressure at tau = 1, and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret, press_tau(1.)/1e6, tfintp(p_bot_spline)


@typechecked
def pt_spline_interp(knot_press: np.ndarray,
                     knot_temp: np.ndarray,
                     pressure: np.ndarray) -> np.ndarray:
    """
    Function for interpolating the P/T knots with a PCHIP 1-D monotonic cubic interpolation. The
    interpolated temperature is smoothed with a Gaussian kernel of width 0.3 dex in pressure
    (see Piette & Madhusudhan 2020).

    Parameters
    ----------
    knot_press : np.ndarray
        Pressure knots (bar).
    knot_temp : np.ndarray
        Temperature knots (K).
    pressure : np.ndarray
        Pressure points (bar) at which the temperatures is interpolated.

    Returns
    -------
    np.ndarray
        Interpolated, smoothed temperature points (K).
    """

    pt_interp = PchipInterpolator(np.log10(knot_press), knot_temp)

    temp_interp = pt_interp(np.log10(pressure))

    log_press = np.log10(pressure)
    log_diff = np.mean(np.diff(log_press))

    if np.std(np.diff(log_press))/log_diff > 1e-6:
        raise ValueError('Expecting equally spaced pressures in log space.')

    return gaussian_filter(temp_interp, sigma=0.3/log_diff, mode='nearest')


@typechecked
def make_half_pressure_better(p_base: dict,
                              pressure: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    press_plus_index = np.zeros(len(pressure)*2).reshape(len(pressure), 2)
    press_plus_index[:, 0] = pressure
    press_plus_index[:, 1] = range(len(pressure))

    press_small = press_plus_index[::24, :]
    press_plus_index = press_plus_index[::2, :]

    indexes_small = press_small[:, 0] > 0.
    indexes = press_plus_index[:, 0] > 0.

    for key, P_cloud in p_base.items():
        indexes_small = indexes_small & ((np.log10(press_small[:, 0]/P_cloud) > 0.05) |
                                         (np.log10(press_small[:, 0]/P_cloud) < -0.3))

        indexes = indexes & ((np.log10(press_plus_index[:, 0]/P_cloud) > 0.05) |
                             (np.log10(press_plus_index[:, 0]/P_cloud) < -0.3))

    press_cut = press_plus_index[~indexes, :]
    press_small_cut = press_small[indexes_small, :]

    press_out = np.zeros((len(press_cut)+len(press_small_cut))*2).reshape(
        (len(press_cut)+len(press_small_cut)), 2)

    press_out[:len(press_small_cut), :] = press_small_cut
    press_out[len(press_small_cut):, :] = press_cut

    press_out = np.sort(press_out, axis=0)

    return press_out[:, 0], press_out[:, 1].astype('int')


@typechecked
def create_abund_dict(abund_in: dict,
                      line_species: list,
                      chemistry: str,
                      half: bool = True,
                      indices: Optional[np.array] = None) -> dict:
    """
    Function to update the names in the abundance dictionary.

    Parameters
    ----------
    abund_in : dict
        Dictionary with the mass fractions.
    line_species : list
        List with the line species.
    chemistry : str
        Chemistry type ('equilibrium' or 'free').
    half : bool
        Use every third pressure point.
    indices : np.ndarray, None
        Pressure indices from the adaptive refinement in a cloudy atmosphere.

    Returns
    -------
    dict
        Dictionary with the updated names of the abundances.
    """

    # create a dictionary with the updated abundance names

    abund_out = {}

    if indices is not None:
        for item in line_species:
            if chemistry == 'equilibrium':
                item_replace = item.replace('_all_iso', '')
                item_replace = item_replace.replace('_lor_cut', '')
                item_replace = item_replace.replace('_burrows', '')

                abund_out[item] = abund_in[item_replace][indices]

            elif chemistry == 'free':
                abund_out[item] = abund_in[item][indices]

        if 'Fe(c)' in abund_in:
            abund_out['Fe(c)'] = abund_in['Fe(c)'][indices]

        if 'MgSiO3(c)' in abund_in:
            abund_out['MgSiO3(c)'] = abund_in['MgSiO3(c)'][indices]

        if 'Al2O3(c)' in abund_in:
            abund_out['Al2O3(c)'] = abund_in['Al2O3(c)'][indices]

        if 'Na2S(c)' in abund_in:
            abund_out['Na2S(c)'] = abund_in['Na2S(c)'][indices]

        if 'KCL(c)' in abund_in:
            abund_out['KCL(c)'] = abund_in['KCL(c)'][indices]

        abund_out['H2'] = abund_in['H2'][indices]
        abund_out['He'] = abund_in['He'][indices]

    elif half:
        for item in line_species:
            if chemistry == 'equilibrium':
                item_replace = item.replace('_all_iso', '')
                item_replace = item_replace.replace('_lor_cut', '')
                item_replace = item_replace.replace('_burrows', '')

                abund_out[item] = abund_in[item_replace][::3]

            elif chemistry == 'free':
                abund_out[item] = abund_in[item][::3]

        if 'Fe(c)' in abund_in:
            abund_out['Fe(c)'] = abund_in['Fe(c)'][::3]

        if 'MgSiO3(c)' in abund_in:
            abund_out['MgSiO3(c)'] = abund_in['MgSiO3(c)'][::3]

        if 'Al2O3(c)' in abund_in:
            abund_out['Al2O3(c)'] = abund_in['Al2O3(c)'][::3]

        if 'Na2S(c)' in abund_in:
            abund_out['Na2S(c)'] = abund_in['Na2S(c)'][::3]

        if 'KCL(c)' in abund_in:
            abund_out['KCL(c)'] = abund_in['KCL(c)'][::3]

        abund_out['H2'] = abund_in['H2'][::3]
        abund_out['He'] = abund_in['He'][::3]

    else:
        for item in line_species:
            if chemistry == 'equilibrium':
                item_replace = item.replace('_all_iso', '')
                item_replace = item_replace.replace('_lor_cut', '')
                item_replace = item_replace.replace('_burrows', '')

                abund_out[item] = abund_in[item_replace]

            elif chemistry == 'free':
                abund_out[item] = abund_in[item]

        if 'Fe(c)' in abund_in:
            abund_out['Fe(c)'] = abund_in['Fe(c)']

        if 'MgSiO3(c)' in abund_in:
            abund_out['MgSiO3(c)'] = abund_in['MgSiO3(c)']

        if 'Al2O3(c)' in abund_in:
            abund_out['Al2O3(c)'] = abund_in['Al2O3(c)']

        if 'Na2S(c)' in abund_in:
            abund_out['Na2S(c)'] = abund_in['Na2S(c)']

        if 'KCL(c)' in abund_in:
            abund_out['KCL(c)'] = abund_in['KCL(c)']

        abund_out['H2'] = abund_in['H2']
        abund_out['He'] = abund_in['He']

    # Corretion for the nuclear spin degeneracy that was not included in the partition function
    # See Charnay et al. (2018)

    if 'FeH' in abund_out:
        abund_out['FeH'] = abund_out['FeH']/2.

    return abund_out


@typechecked
def calc_spectrum_clear(rt_object: Radtrans,
                        pressure: np.ndarray,
                        temperature: np.ndarray,
                        logg: float,
                        c_o_ratio: Optional[float],
                        metallicity: Optional[float],
                        log_p_quench: Optional[float],
                        log_x_abund: Optional[dict],
                        chemistry: str,
                        half: bool = False,
                        contribution: bool = False) -> Tuple[np.ndarray,
                                                             np.ndarray,
                                                             Optional[np.ndarray]]:
    """
    Function to simulate an emission spectrum of a clear atmosphere. The function supports both
    equilibrium chemistry (``chemistry='equilibrium'``) and free abundances (``chemistry='free'``).

    rt_object : Radtrans
        Instance of ``Radtrans``.
    pressure : np.ndarray
        Array with the pressure points (bar).
    temperature : np.ndarray
        Array with the temperature points (K) corresponding to ``pressure``.
    logg : float
        Log10 of the surface gravity (cm s-2).
    c_o_ratio : float
        Carbon-to-oxygen ratio.
    metallicity : float
        Metallicity.
    log_p_quench : float
        Log10 of the quench pressure.
    log_x_abund : dict, None
        Dictionary with the log10 of the abundances. Only required when ``chemistry='free'``.
    chemistry : str
        Chemistry type (``'equilibrium'`` or ``'free'``).
    half: bool
        Only use every third P/T point.
    contribution : bool
        Calculate the emission contribution.

    Returns
    -------
    np.ndarray
        Wavelength (um).
    np.ndarray
        Flux (W m-2 um-1).
    np.ndarray, None
        Emission contribution.
    """

    if chemistry == 'equilibrium':
        # chemical equilibrium
        abund_in = interpol_abundances(np.full(pressure.shape, c_o_ratio),
                                       np.full(pressure.shape, metallicity),
                                       temperature,
                                       pressure,
                                       Pquench_carbon=10.**log_p_quench)

        # mean molecular weight
        mmw = abund_in['MMW']

    elif chemistry == 'free':
        # free abundances

        # create a dictionary with all mass fractions
        abund_in = mass_fractions(log_x_abund)

        # mean molecular weight
        mmw = mean_molecular_weight(abund_in)

        # create arrays of constant atmosphere abundance
        for item in abund_in:
            abund_in[item] *= np.ones_like(pressure)

        # create an array of a constant mean molecular weight
        mmw *= np.ones_like(pressure)

    # extract every three levels if half=True

    if half:
        temperature = temperature[::3]
        pressure = pressure[::3]
        mmw = mmw[::3]

    abundances = create_abund_dict(abund_in,
                                   rt_object.line_species,
                                   chemistry,
                                   half=half,
                                   indices=None)

    # calculate the emission spectrum
    rt_object.calc_flux(temperature, abundances, 10.**logg, mmw, contribution=contribution)

    # convert frequency (Hz) to wavelength (cm)
    wavel = nc.c/rt_object.freq

    # optionally return the emission contribution
    if contribution:
        contr_em = rt_object.contr_em
    else:
        contr_em = None

    # return wavelength (micron), flux (W m-2 um-1), and emission contribution
    return 1e4*wavel, 1e-7*rt_object.flux*nc.c/wavel**2., contr_em


@typechecked
def calc_spectrum_clouds(rt_object: Union[Radtrans, RadtransScatter],
                         pressure: np.ndarray,
                         temperature: np.ndarray,
                         c_o_ratio: float,
                         metallicity: float,
                         log_p_quench: float,
                         log_x_base: dict,
                         fsed: float,
                         Kzz: float,
                         logg: float,
                         sigma_lnorm: float,
                         chemistry: str,
                         half: bool = False,
                         plotting: bool = False,
                         contribution: bool = False) -> Tuple[np.ndarray,
                                                              np.ndarray,
                                                              Optional[np.ndarray]]:
    """
    Function to simulate an emission spectrum of a cloudy atmosphere. Currently, the function
    only supports equilibrium chemistry (i.e. ``chemistry='equilibrium'``).

    Parameters
    ----------
    rt_object : Radtrans, RadtransScatter
        Instance of ``Radtrans``.
    pressure : np.ndarray
        Array with the pressure points (bar).
    temperature : np.ndarray
        Array with the temperature points (K) corresponding to ``pressure``.
    c_o_ratio : float
        Carbon-to-oxygen ratio.
    metallicity : float
        Metallicity.
    log_p_quench : float
        Log10 of the quench pressure.
    log_x_base : dict
        Dictionary with the log10 of the mass fractions at the cloud base.
    fsed : float
        Sedimentation parameter.
    Kzz : float
        Log 10 of the eddy diffusion coefficient (cm2 s-1).
    logg : float
        Log10 of the surface gravity (cm s-2).
    sigma_lnorm : float
        Geometric standard deviation of the log-normal size distribution.
    chemistry : str
        Chemistry type (only ``'equilibrium'`` is supported).
    half: bool
        Only use every third P/T point.
    plotting : bool
        Create plots.
    contribution : bool
        Calculate the emission contribution.

    Returns
    -------
    np.ndarray
        Wavelength (um).
    np.ndarray
        Flux (W m-2 um-1).
    np.ndarray, None
        Emission contribution.
    """

    # interpolate the abundances, following chemical equilibrium
    abund_in = interpol_abundances(np.full(pressure.shape, c_o_ratio),
                                   np.full(pressure.shape, metallicity),
                                   temperature,
                                   pressure,
                                   Pquench_carbon=1e1**log_p_quench)

    # extract the mean molecular weight
    mmw = abund_in['MMW']

    p_base = {}

    for item in log_x_base:
        p_base_item = find_cloud_deck(item,
                                      pressure,
                                      temperature,
                                      metallicity,
                                      c_o_ratio,
                                      mmw=np.mean(mmw),
                                      plotting=plotting)

        abund_in[f'{item}(c)'] = np.zeros_like(temperature)

        abund_in[f'{item}(c)'][pressure < p_base_item] = 10.**log_x_base[item] * \
            (pressure[pressure <= p_base_item] / p_base_item)**fsed

        p_base[f'{item}(c)'] = p_base_item

    # adaptive pressure refinement around the cloud base
    # _, small_index = make_half_pressure_better(p_base, pressure)

    # TODO
    small_index = None

    abundances = create_abund_dict(abund_in,
                                   rt_object.line_species,
                                   chemistry,
                                   half=half,
                                   indices=small_index)

    Kzz_use = np.full(pressure.shape, 10.**Kzz)

    if half:
        temperature = temperature[::3]
        pressure = pressure[::3]
        mmw = mmw[::3]
        Kzz_use = Kzz_use[::3]

    fseds = {}

    for item in log_x_base:
        fseds[f'{item}(c)'] = fsed

    if plotting:
        if 'CO_all_iso' in abundances:
            plt.plot(abundances['CO_all_iso'], pressure, label='CO')
        if 'CH4' in abundances:
            plt.plot(abundances['CH4'], pressure, label='CH4')
        if 'H2O' in abundances:
            plt.plot(abundances['H2O'], pressure, label='H2O')
        plt.xlim(1e-10, 1.)
        plt.ylim(pressure[-1], pressure[0])
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Mass fraction')
        plt.ylabel('Pressure (bar)')
        plt.axhline(10.**log_p_quench)
        plt.legend(loc='best')
        plt.savefig('abundances.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(temperature, pressure, 'o', ls='none', ms=2.)

        for item in log_x_base:
            plt.axhline(p_base[f'{item}(c)'], label=f'Cloud deck {item}')

        plt.yscale('log')
        plt.ylim(1e3, 1e-6)
        plt.xlim(0., 4000.)
        plt.savefig('pt_cloud_deck.pdf', bbox_inches='tight')
        plt.clf()

        for item in log_x_base:
            plt.plot(abundances[f'{item}(c)'], pressure)
            plt.axhline(p_base[f'{item}(c)'])
            plt.yscale('log')
            if np.count_nonzero(abundances[f'{item}(c)']) > 0:
                plt.xscale('log')
            plt.ylim(1e3, 1e-6)
            plt.xlim(1e-10, 1.)
            log_x_base_item = log_x_base[item]
            plt.title(f'fsed = {fsed:.2f}, lgK = {Kzz:.2f}, X_b = {log_x_base_item:.2f}')
            plt.savefig(f'{item.lower()}_clouds.pdf', bbox_inches='tight')
            plt.clf()

    # Turn off clouds
    # abundances['MgSiO3(c)'] = np.zeros_like(pressure)
    # abundances['Fe(c)'] = np.zeros_like(pressure)

    # reinitiate the pressure layers after make_half_pressure_better
    rt_object.setup_opa_structure(pressure)

    if isinstance(rt_object, Radtrans):
        # the argument of fsed is a float
        rt_object.calc_flux(temperature,
                            abundances,
                            10.**logg,
                            mmw,
                            Kzz=Kzz_use,
                            fsed=fsed,
                            sigma_lnorm=sigma_lnorm,
                            add_cloud_scat_as_abs=False,
                            contribution=contribution)

    elif isinstance(rt_object, RadtransScatter):
        # the argument of fsed is a dictionary
        rt_object.calc_flux(temperature,
                            abundances,
                            10.**logg,
                            mmw,
                            Kzz=Kzz_use,
                            fsed=fseds,
                            sigma_lnorm=sigma_lnorm,
                            contribution=contribution)

    wlen_micron = nc.c/rt_object.freq/1e-4
    wlen = nc.c/rt_object.freq
    flux = rt_object.flux

    # convert flux f_nu to f_lambda
    f_lambda = flux*nc.c/wlen**2.

    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7

    # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim([1e2,1e-6])
    # plt.ylabel('P (bar)')
    # plt.xlabel('Average particle size of MgSiO3 particles (microns)')
    # plt.plot(rt_object.r_g[:,rt_object.cloud_species.index('MgSiO3(c)')]/1e-4, pressure)
    # plt.savefig('mgsio3_size.png')
    # plt.show()
    # plt.clf()

    # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim([1e2,1e-6])
    # plt.ylabel('P (bar)')
    # plt.xlabel('Average particle size of Fe particles (microns)')
    # plt.plot(rt_object.r_g[:,rt_object.cloud_species.index('Fe(c)')]/1e-4, pressure)
    # plt.savefig('fe_size.png')
    # plt.show()rt_object
    # plt.clf()

    # optionally return the emission contribution
    if contribution:
        contr_em = rt_object.contr_em
    else:
        contr_em = None

    # return wlen_micron, f_lambda, rt_object.pphot, rt_object.tau_pow, np.mean(rt_object.tau_cloud)
    return wlen_micron, f_lambda, contr_em


@typechecked
def mass_fractions(log_x_abund: dict) -> dict:
    """
    Function to return a dictionary with the mass fractions of all species.

    Parameters
    ----------
    log_x_abund : dict
        Dictionary with the log10 of the mass fractions of metals.

    Returns
    -------
    dict
        Dictionary with the mass fractions of all species.
    """

    # initiate abundance dictionary
    abund = {}

    # initiate the total mass fraction of the metals
    metal_sum = 0.

    for item in log_x_abund:
        # add the mass fraction to the dictionary
        abund[item] = 10.**log_x_abund[item]

        # update the total mass fraction of the metals
        metal_sum += abund[item]

    # mass fraction of H2 and He
    ab_h2_he = 1. - metal_sum

    # add H2 and He mass fraction to the dictionary
    abund['H2'] = ab_h2_he*0.75
    abund['He'] = ab_h2_he*0.25

    return abund


@typechecked
def calc_metal_ratio(log_x_abund: dict) -> Tuple[float, float]:
    """
    Parameters
    ----------
    log_x_abund : dict
        Dictionary with the log10 values of the mass fractions.

    Returns
    -------
    float
    float
    """

    # solar C/H from Asplund et al. (2009)
    c_h_solar = 10.**(8.43-12.)

    # solar O/H from Asplund et al. (2009)
    o_h_solar = 10.**(8.69-12.)

    # get the atomic masses
    masses = atomic_masses()

    # create a dictionary with all mass fractions
    abund = mass_fractions(log_x_abund)

    # calculate the mean molecular weight from the input mass fractions
    mmw = mean_molecular_weight(abund)

    # initiate the C, H, and O abundance
    c_abund = 0.
    o_abund = 0.
    h_abund = 0.

    # calculate the total C abundance

    if 'CO' in abund:
        c_abund += abund['CO'] * mmw/masses['CO']

    if 'CO_all_iso' in abund:
        c_abund += abund['CO_all_iso'] * mmw/masses['CO']

    if 'CO2' in abund:
        c_abund += abund['CO2'] * mmw/masses['CO2']

    if 'CH4' in abund:
        c_abund += abund['CH4'] * mmw/masses['CH4']

    # calculate the total O abundance

    if 'CO' in abund:
        o_abund += abund['CO'] * mmw/masses['CO']

    if 'CO_all_iso' in abund:
        o_abund += abund['CO_all_iso'] * mmw/masses['CO']

    if 'CO2' in abund:
        o_abund += 2. * abund['CO2'] * mmw/masses['CO2']

    if 'H2O' in abund:
        o_abund += abund['H2O'] * mmw/masses['H2O']

    # calculate the total H abundance

    h_abund += 2. * abund['H2'] * mmw/masses['H2']

    if 'CH4' in abund:
        h_abund += 4. * abund['CH4'] * mmw/masses['CH4']

    if 'H2O' in abund:
        h_abund += 2. * abund['H2O'] * mmw/masses['H2O']

    if 'NH3' in abund:
        h_abund += 3. * abund['NH3'] * mmw/masses['NH3']

    if 'H2S' in abund:
        h_abund += 2. * abund['H2S'] * mmw/masses['H2S']

    return np.log10(c_abund/h_abund/c_h_solar), np.log10(o_abund/h_abund/o_h_solar)


@typechecked
def mean_molecular_weight(abundances: dict) -> float:
    """
    Function to calculate the mean molecular weight from the abundances.

    Parameters
    ----------
    abundances : dict
        Dictionary with the mass fraction of each species.

    Returns
    -------
    float
        Mean molecular weight in atomic mass units.
    """

    mol_weight = atomic_masses()

    mmw = 0.

    for key in abundances:
        if key == 'CO_all_iso':
            mmw += abundances[key]/mol_weight['CO']

        elif key in ['Na_lor_cut', 'Na_burrows']:
            mmw += abundances[key]/mol_weight['Na']

        elif key in ['K_lor_cut', 'K_burrows']:
            mmw += abundances[key]/mol_weight['K']

        else:
            mmw += abundances[key]/mol_weight[key]

    return 1./mmw


@typechecked
def potassium_abundance(log_x_abund: dict) -> float:
    """
    Function to calculate the mass fraction of potassium at a solar ratio of the sodium and
    potassium abundances.

    Parameters
    ----------
    log_x_abund : dict
        Dictionary with the log10 of the mass fractions.

    Returns
    -------
    float
        Log10 of the mass fraction of potassium.
    """

    # solar volume mixing ratios of Na and K (Asplund et al. 2009)
    n_na_solar = 1.60008694353205e-06
    n_k_solar = 9.86605611925677e-08

    # get the atomic masses
    masses = atomic_masses()

    # create a dictionary with all mass fractions
    x_abund = mass_fractions(log_x_abund)

    # calculate the mean molecular weight from the input mass fractions
    mmw = mean_molecular_weight(x_abund)

    # volume mixing ratio of sodium
    if 'Na' in log_x_abund:
        n_na_abund = x_abund['Na'] * mmw/masses['Na']

    elif 'Na_lor_cut' in log_x_abund:
        n_na_abund = x_abund['Na_lor_cut'] * mmw/masses['Na']

    elif 'Na_burrows' in log_x_abund:
        n_na_abund = x_abund['Na_burrows'] * mmw/masses['Na']

    # volume mixing ratio of potassium
    n_k_abund = n_na_abund * n_k_solar/n_na_solar

    return np.log10(n_k_abund * masses['K']/mmw)


@typechecked
def log_x_cloud_base(c_o_ratio: float,
                     metallicity: float,
                     cloud_fractions: dict) -> dict:
    """
    Function for returning a dictionary with the log10 mass fractions at the cloud base.

    Parameters
    ----------
    c_o_ratio : float
        C/O ratio.
    metallicity : float
        Metallicity, [Fe/H].
    cloud_fractions : dict
        Dictionary with mass fractions at the cloud base, relative to the maximum values allowed
        from elemental abundances. The dictionary keys are the cloud species without the structure
        and shape index (e.g. Na2S(c) instead of Na2S(c)_cd).

    Returns
    -------
    dict
        Dictionary with the log10 mass fractions at the cloud base.
    """

    log_x_base = {}

    for item in cloud_fractions:
        # Mass fraction
        x_cloud = cloud_mass_fraction(f'{item[:-3]}', metallicity, c_o_ratio)

        # Log10 of the mass fraction at the cloud base
        log_x_base[f'{item[:-3]}'] = np.log10(10.**cloud_fractions[item]*x_cloud)

    return log_x_base


#############################################################
# To calculate X_Fe from [Fe/H], C/O
#############################################################

# metal species
# metals = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Fe', 'Ni']

@typechecked
def solar_mixing_ratios() -> dict:
    """
    Function which returns the volume mixing ratios of a solar elemental abundances (i.e.
    [Fe/H] = 0), adopted from Asplund et al. (2009).

    Returns
    -------
    dict
        Dictionary with the solar number fractions (volume mixing ratios).
    """

    n_fracs = {}
    n_fracs['H'] = 0.9207539305
    n_fracs['He'] = 0.0783688694
    n_fracs['C'] = 0.0002478241
    n_fracs['N'] = 6.22506056949881e-05
    n_fracs['O'] = 0.0004509658
    n_fracs['Na'] = 1.60008694353205e-06
    n_fracs['Mg'] = 3.66558742055362e-05
    n_fracs['Al'] = 2.595e-06
    n_fracs['Si'] = 2.9795e-05
    n_fracs['P'] = 2.36670201997668e-07
    n_fracs['S'] = 1.2137900734604e-05
    n_fracs['Cl'] = 2.91167958499589e-07
    n_fracs['K'] = 9.86605611925677e-08
    n_fracs['Ca'] = 2.01439011429255e-06
    n_fracs['Ti'] = 8.20622804366359e-08
    n_fracs['V'] = 7.83688694089992e-09
    n_fracs['Fe'] = 2.91167958499589e-05
    n_fracs['Ni'] = 1.52807116806281e-06

    return n_fracs


@typechecked
def atomic_masses() -> dict:
    """
    Function which returns the atomic and molecular masses.

    Returns
    -------
    dict
        Dictionary with the atomic and molecular masses.
    """

    masses = {}

    # atoms
    masses['H'] = 1.
    masses['He'] = 4.
    masses['C'] = 12.
    masses['N'] = 14.
    masses['O'] = 16.
    masses['Na'] = 23.
    masses['Na_lor_cur'] = 23.
    masses['Na_burrows'] = 23.
    masses['Mg'] = 24.3
    masses['Al'] = 27.
    masses['Si'] = 28.
    masses['P'] = 31.
    masses['S'] = 32.
    masses['Cl'] = 35.45
    masses['K'] = 39.1
    masses['K_lor_cut'] = 39.1
    masses['K_burrows'] = 39.1
    masses['Ca'] = 40.
    masses['Ti'] = 47.9
    masses['V'] = 51.
    masses['Fe'] = 55.8
    masses['Ni'] = 58.7

    # molecules
    masses['H2'] = 2.
    masses['H2O'] = 18.
    masses['CH4'] = 16.
    masses['CO2'] = 44.
    masses['CO'] = 28.
    masses['CO_all_iso'] = 28.
    masses['NH3'] = 17.
    masses['HCN'] = 27.
    masses['C2H2,acetylene'] = 26.
    masses['PH3'] = 34.
    masses['H2S'] = 34.
    masses['VO'] = 67.
    masses['TiO'] = 64.
    masses['FeH'] = 57.
    masses['OH'] = 17.

    return masses


@typechecked
def cloud_mass_fraction(composition: str,
                        metallicity: float,
                        c_o_ratio: float) -> float:
    """
    Function to calculate the mass fraction for a cloud species.

    Parameters
    ----------
    composition : str
        Cloud composition ('Fe', 'MgSiO3', 'Al2O3', 'Na2S', or 'KCL').
    metallicity : float
        Metallicity [Fe/H].
    c_o_ratio : float
        Carbon-to-oxygen ratio.

    Returns
    -------
    float
        Mass fraction.
    """

    # Solar fractional number densities (i.e. volume mixing ratios; VMR)
    nfracs = solar_mixing_ratios()

    # Atomic masses
    masses = atomic_masses()

    # Make a copy of the dictionary with the solar number densities
    nfracs_use = copy.copy(nfracs)

    # Scale the solar number densities by the [Fe/H], except H and He
    for item in nfracs.keys():
        if item != 'H' and item != 'He':
            nfracs_use[item] = nfracs[item]*10.**metallicity

    # Adjust the VMR of O with the C/O ratio
    nfracs_use['O'] = nfracs_use['C']/c_o_ratio

    if composition == 'Fe':
        nfrac_cloud = nfracs_use['Fe']
        mass_cloud = masses['Fe']

    elif composition == 'MgSiO3':
        nfrac_cloud = np.min([nfracs_use['Mg'], nfracs_use['Si'], nfracs_use['O']/3.])
        mass_cloud = masses['Mg'] + masses['Si'] + 3. * masses['O']

    elif composition == 'Al2O3':
        nfrac_cloud = np.min([nfracs_use['Al']/2., nfracs_use['O']/3.])
        mass_cloud = 2. * masses['Al'] + 3. * masses['O']

    elif composition == 'Na2S':
        nfrac_cloud = np.min([nfracs_use['Na']/2., nfracs_use['S']])
        mass_cloud = 2. * masses['Na'] + masses['S']

    elif composition == 'KCL':
        nfrac_cloud = np.min([nfracs_use['K'], nfracs_use['Cl']])
        mass_cloud = masses['K'] + masses['Cl']

    # Cloud mass fraction
    x_cloud = mass_cloud*nfrac_cloud

    mass_norm = 0.
    for item in nfracs_use:
        # Sum up the mass fractions of all species
        mass_norm += masses[item]*nfracs_use[item]

    # Normalize the cloud mass fraction by the total mass fraction
    return x_cloud/mass_norm


@typechecked
def find_cloud_deck(composition: str,
                    press: np.ndarray,
                    temp: np.ndarray,
                    metallicity: float,
                    c_o_ratio: float,
                    mmw: float = 2.33,
                    plotting: bool = False) -> float:
    """
    Function to find the base of the cloud deck by intersecting the P/T profile with the
    saturation vapor pressure.

    Parameters
    ----------
    composition : str
        Cloud composition ('Fe', 'MgSiO3', 'Al2O3', 'Na2S', or 'KCL').
    press : np.ndarray
        Pressures (bar).
    temp : np.ndarray
        Temperatures (K).
    metallicity : float
        Metallicity [Fe/H].
    c_o_ratio : float
        Carbon-to-oxygen ratio.
    mmw : float
        Mean molecular weight.
    plotting : bool
        Create a plot.

    Returns
    -------
    float
        Pressure (bar) at the base of the cloud deck.
    """

    if composition == 'Fe':
        Pc, Tc = return_T_cond_Fe_comb(metallicity, c_o_ratio, mmw)

    elif composition == 'MgSiO3':
        Pc, Tc = return_T_cond_MgSiO3(metallicity, c_o_ratio, mmw)

    elif composition == 'Al2O3':
        Pc, Tc = return_T_cond_Al2O3(metallicity, c_o_ratio, mmw)

    elif composition == 'Na2S':
        Pc, Tc = return_T_cond_Na2S(metallicity, c_o_ratio, mmw)

    elif composition == 'KCL':
        Pc, Tc = return_T_cond_KCl(metallicity, c_o_ratio, mmw)

    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]

    tcond_p = interp1d(Pc, Tc)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = float(P_clouds[-1])

    else:
        P_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim(0., 3000.)
        plt.ylim(1e2, 1e-6)
        plt.savefig(f'{composition.lower()}_clouds_cdf.pdf', bbox_inches='tight')
        plt.clf()

    return P_cloud


@typechecked
def return_T_cond_Fe(FeH: float,
                     CO: float,
                     MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for solid Fe.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Arrays with the saturation pressure and temperature.
    """

    masses = atomic_masses()

    T = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including erratum (P_vap is in bar, not cgs!)
    P_vap = lambda x: np.exp(15.71 - 47664./x)

    XFe = cloud_mass_fraction('Fe', FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T


@typechecked
def return_T_cond_Fe_l(FeH: float,
                       CO: float,
                       MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for liquid Fe.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Arrays with the saturation pressure and temperature.
    """

    masses = atomic_masses()

    T = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including erratum (P_vap is in bar, not cgs!)
    P_vap = lambda x: np.exp(9.86 - 37120./x)

    XFe = cloud_mass_fraction('Fe', FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T


@typechecked
def return_T_cond_Fe_comb(FeH: float,
                          CO: float,
                          MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for Fe.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Arrays with the saturation pressure and temperature.
    """

    P1, T1 = return_T_cond_Fe(FeH, CO, MMW)
    P2, T2 = return_T_cond_Fe_l(FeH, CO, MMW)

    retP = np.zeros_like(P1)
    index = P1 < P2
    retP[index] = P1[index]
    retP[~index] = P2[~index]

    return retP, T2


@typechecked
def return_T_cond_MgSiO3(FeH: float,
                         CO: float,
                         MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for MgSiO3.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Arrays with the saturation pressure and temperature.
    """

    masses = atomic_masses()

    T = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including erratum (P_vap is in bar, not cgs!)
    P_vap = lambda x: np.exp(25.37 - 58663./x)

    Xmgsio3 = cloud_mass_fraction('MgSiO3', FeH, CO)

    m_mgsio3 = masses['Mg'] + masses['Si'] + 3. * masses['O']

    return P_vap(T)/(Xmgsio3*MMW/m_mgsio3), T


@typechecked
def return_T_cond_Al2O3(FeH: float,
                        CO: float,
                        MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the condensation temperature for Al2O3.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    np.ndarray
        Array with the pressures (bar).
    np.ndarray
        Array with the condensation temperatures (K).
    """

    # Return dictionary with atomic masses
    # masses = atomic_masses()

    # Create pressures (bar)
    pressure = np.logspace(-6, 3, 1000)

    # Equilibrium mass fraction of Al2O3
    # Xal2o3 = cloud_mass_fraction('Al2O3', FeH, CO)

    # Molecular mass of Al2O3
    # m_al2o3 = 3. * masses['Al'] + 2. * masses['O']

    # Partial pressure of Al2O3
    # part_press = pressure/(Xal2o3*MMW/m_al2o3)

    # Condensation temperature of Al2O3 (see Eq. 4 in Wakeford et al. 2017)
    t_cond = 1e4 / (5.014 - 0.2179*np.log10(pressure) + 2.264e-3*np.log10(pressure)**2 - 0.580*FeH)

    return pressure, t_cond


@typechecked
def return_T_cond_Na2S(FeH: float,
                       CO: float,
                       MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for Na2S.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Arrays with the saturation pressure and temperature.
    """

    masses = atomic_masses()

    # Taken from Charnay+2018
    T = np.linspace(100., 10000., 1000)
    # This is the partial pressure of Na, so divide by factor 2 to get the partial pressure of
    # the hypothetical Na2S gas particles, this is OK: there are more S than Na atoms at solar
    # abundance ratios.
    P_vap = lambda x: 1e1**(8.55 - 13889./x - 0.5*FeH)/2.

    Xna2s = cloud_mass_fraction('Na2S', FeH, CO)

    m_na2s = 2.*masses['Na'] + masses['S']

    return P_vap(T)/(Xna2s*MMW/m_na2s), T


@typechecked
def return_T_cond_KCl(FeH: float,
                      CO: float,
                      MMW: float = 2.33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for calculating the saturation pressure for KCl.

    Parameters
    ----------
    FeH : float
        Metallicity.
    CO : float
        Carbon-to-oxygen ratio.
    MMW : float
        Mean molecular weight.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Arrays with the saturation pressure and temperature.
    """

    masses = atomic_masses()

    # Taken from Charnay+2018
    T = np.linspace(100., 10000., 1000)
    P_vap = lambda x: 1e1**(7.611 - 11382./T)

    Xkcl = cloud_mass_fraction('KCL', FeH, CO)

    m_kcl = masses['K'] + masses['Cl']

    return P_vap(T)/(Xkcl*MMW/m_kcl), T


@typechecked
def convolve(input_wavel: np.ndarray,
             input_flux: np.ndarray,
             spec_res: float) -> np.ndarray:
    """
    Function to convolve a spectrum with a Gaussian filter.

    Parameters
    ----------
    input_wavel : np.ndarray
        Input wavelengths.
    input_flux : np.ndarrau
        Input flux
    spec_res : float
        Spectral resolution of the Gaussian filter.

    Returns
    -------
    np.ndarray
        Convolved spectrum.
    """

    # From talking to Ignas: delta lambda of resolution element
    # is FWHM of the LSF's standard deviation, hence:
    sigma_lsf = 1./spec_res/(2.*np.sqrt(2.*np.log(2.)))

    # The input spacing of petitRADTRANS is 1e3, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the spacing is constant
    # as a function of wavelength
    spacing = np.mean(2.*np.diff(input_wavel)/(input_wavel[1:]+input_wavel[:-1]))

    # Calculate the sigma to be used in the gauss filter in units
    # of input wavelength bins
    sigma_lsf_gauss_filter = sigma_lsf/spacing

    return gaussian_filter(input_flux, sigma=sigma_lsf_gauss_filter, mode='nearest')
