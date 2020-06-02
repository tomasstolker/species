"""
Utility functions for plotting data.
"""

from typing import Optional, Tuple, List

import numpy as np

from typeguard import typechecked


def sptype_substellar(sptype,
                      shape):
    """
    Parameters
    ----------
    sptype :
    shape :

    Returns
    -------
    numpy.ndarray
    """

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        if item[0:2] in ['M0', 'M1', 'M2', 'M3', 'M4']:
            spt_disc[i] = 0.5

        elif item[0:2] in ['M5', 'M6', 'M7', 'M8', 'M9']:
            spt_disc[i] = 1.5

        elif item[0:2] in ['L0', 'L1', 'L2', 'L3', 'L4']:
            spt_disc[i] = 2.5

        elif item[0:2] in ['L5', 'L6', 'L7', 'L8', 'L9']:
            spt_disc[i] = 3.5

        elif item[0:2] in ['T0', 'T1', 'T2', 'T3', 'T4']:
            spt_disc[i] = 4.5

        elif item[0:2] in ['T5', 'T6', 'T7', 'T8', 'T9']:
            spt_disc[i] = 5.5

        elif 'Y' in item:
            spt_disc[i] = 6.5

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc


def sptype_stellar(sptype,
                   shape):
    """
    Parameters
    ----------
    sptype :
    shape :

    Returns
    -------
    numpy.ndarray
    """

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        if item[0] == 'O':
            spt_disc[i] = 0.5

        elif item[0] == 'B':
            spt_disc[i] = 1.5

        elif item[0] == 'A':
            spt_disc[i] = 2.5

        elif item[0] == 'F':
            spt_disc[i] = 3.5

        elif item[0] == 'G':
            spt_disc[i] = 4.5

        elif item[0] == 'K':
            spt_disc[i] = 5.5

        elif item[0] == 'M':
            spt_disc[i] = 6.5

        elif item[0] == 'L':
            spt_disc[i] = 7.5

        elif item[0] == 'T':
            spt_disc[i] = 8.5

        elif item[0] == 'Y':
            spt_disc[i] = 9.5

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc


@typechecked
def update_labels(param: List[str]) -> List[str]:
    """
    Function for formatting the model parameters to use them as labels in the posterior plot.

    Parameters
    ----------
    param : list
        List with names of the model parameters.

    Returns
    -------
    list
        List with parameter labels for plots.
    """

    if 'teff' in param:
        index = param.index('teff')
        param[index] = r'$T_\mathregular{eff}$ (K)'

    if 'logg' in param:
        index = param.index('logg')
        param[index] = r'$\log\,g$'

    if 'metallicity' in param:
        index = param.index('metallicity')
        param[index] = r'[Fe/H]'

    if 'feh' in param:
        index = param.index('feh')
        param[index] = r'[Fe/H]'

    if 'fsed' in param:
        index = param.index('fsed')
        param[index] = r'f$_\mathregular{sed}$'

    if 'c_o_ratio' in param:
        index = param.index('c_o_ratio')
        param[index] = r'C/O'

    if 'radius' in param:
        index = param.index('radius')
        param[index] = r'$R$ ($\mathregular{R_{J}}$)'

    if 'luminosity' in param:
        index = param.index('luminosity')
        param[index] = r'$\log\,L$/L$_\odot$'

    if 'dust_radius' in param:
        index = param.index('dust_radius')
        param[index] = r'$\log\,r_\mathregular{g}/\mathrm{µm}$'

    if 'dust_sigma' in param:
        index = param.index('dust_sigma')
        param[index] = r'$\sigma_\mathregular{g}$'

    if 'dust_ext' in param:
        index = param.index('dust_ext')
        param[index] = r'$A_\mathregular{V}$ (mag)'

    if 'tint' in param:
        index = param.index('tint')
        param[index] = r'$T_\mathregular{int}$ (K)'

    for i in range(15):
        if f't{i}' in param:
            index = param.index(f't{i}')
            param[index] = rf'$T_\mathregular{{{i}}}$ (K)'

    if 'alpha' in param:
        index = param.index('alpha')
        param[index] = r'$\alpha$'

    if 'log_delta' in param:
        index = param.index('log_delta')
        param[index] = r'$\log\,\delta$'

    if 'log_p_quench' in param:
        index = param.index('log_p_quench')
        param[index] = r'$\log\,P_\mathregular{quench}$'

    if 'sigma_lnorm' in param:
        index = param.index('sigma_lnorm')
        param[index] = r'$\sigma_{g}$'

    if 'kzz' in param:
        index = param.index('kzz')
        param[index] = r'$\log\,K_{zz}$'

    if 'na2s_fraction' in param:
        index = param.index('na2s_fraction')
        param[index] = r'$\log\,\tilde{X}_\mathregular{Na_{2}S}$'

    if 'kcl_fraction' in param:
        index = param.index('kcl_fraction')
        param[index] = r'$\log\,\tilde{X}_\mathregular{KCl}$'

    for i, item in enumerate(param):
        if item[0:8] == 'scaling_':
            param[i] = rf'$a_\mathregular{{{item[8:]}}}$'

        elif item[0:6] == 'error_':
            param[i] = rf'$b_\mathregular{{{item[6:]}}}$'

        elif item[0:11] == 'wavelength_':
            param[i] = rf'$c_\mathregular{{{item[11:]}}}$ (nm)'

        elif item[0:9] == 'corr_len_':
            param[i] = rf'$\log\,\ell_\mathregular{{{item[9:]}}}$'

        elif item[0:9] == 'corr_amp_':
            param[i] = rf'$f_\mathregular{{{item[9:]}}}$'

    if 'c_h_ratio' in param:
        index = param.index('c_h_ratio')
        param[index] = r'[C/H]'

    if 'o_h_ratio' in param:
        index = param.index('o_h_ratio')
        param[index] = r'[O/H]'

    for i in range(100):
        if f'teff_{i}' in param:
            index = param.index(f'teff_{i}')
            param[index] = rf'$T_\mathregular{{{i+1}}}$ (K)'

        else:
            break

    for i in range(100):
        if f'radius_{i}' in param:
            index = param.index(f'radius_{i}')
            param[index] = rf'$R_\mathregular{{{i+1}}}$ ' + r'($\mathregular{R_{J}}$)'

        else:
            break

    return param


@typechecked
def model_name(key: str) -> str:
    """
    Function for updating a model name for use in plots.

    Parameters
    ----------
    key : str
        Model name as used by species.

    Returns
    -------
    str
        Updated model name for plots.
    """

    if key == 'drift-phoenix':
        name = 'DRIFT-PHOENIX'

    elif key == 'ames-cond':
        name = 'AMES-Cond'

    elif key == 'ames-dusty':
        name = 'AMES-Dusty'

    elif key == 'bt-settl':
        name = 'BT-Settl'

    elif key == 'bt-settl-cifist':
        name = 'BT-Settl'

    elif key == 'bt-nextgen':
        name = 'BT-NextGen'

    elif key == 'petitcode-cool-clear':
        name = 'petitCODE'

    elif key == 'petitcode-cool-cloudy':
        name = 'petitCODE'

    elif key == 'petitcode-hot-clear':
        name = 'petitCODE'

    elif key == 'petitcode-hot-cloudy':
        name = 'petitCODE'

    elif key == 'exo-rem':
        name = 'Exo-REM'

    elif key == 'planck':
        name = 'Blackbody radiation'

    elif key == 'zhu2015':
        name = 'Zhu (2015)'

    return name


@typechecked
def quantity_unit(param: List[str],
                  object_type: str) -> Tuple[List[str], List[Optional[str]], List[str]]:
    """
    Function for creating lists with quantities, units, and labels for fitted parameter.

    Parameters
    ----------
    param : list
        List with parameter names.
    object_type : str
        Object type (``'planet'`` or ``'star'``).

    Returns
    -------
    list
        List with the quantities.
    list
        List with the units.
    list
        List with the parameter labels for plots.
    """

    quantity = []
    unit = []
    label = []

    if 'teff' in param:
        quantity.append('teff')
        unit.append('K')
        label.append(r'$T_\mathregular{eff}$')

    if 'logg' in param:
        quantity.append('logg')
        unit.append(None)
        label.append(r'$\log\,g$')

    if 'metallicity' in param:
        quantity.append('metallicity')
        unit.append(None)
        label.append(r'[Fe/H]')

    if 'feh' in param:
        quantity.append('feh')
        unit.append(None)
        label.append('[Fe/H]')

    if 'fsed' in param:
        quantity.append('fsed')
        unit.append(None)
        label.append(r'f$_\mathregular{sed}$')

    if 'c_o_ratio' in param:
        quantity.append('c_o_ratio')
        unit.append(None)
        label.append('C/O')

    if 'radius' in param:
        quantity.append('radius')

        if object_type == 'planet':
            unit.append(r'$R_\mathregular{J}}$')
        elif object_type == 'star':
            unit.append(r'$R_\mathregular{\odot}}$')

        label.append(r'$R$')

    for i in range(100):
        if f'teff_{i}' in param:
            quantity.append(f'teff_{i}')
            unit.append('K')
            label.append(rf'$T_\mathregular{{{i+1}}}$')

        else:
            break

    for i in range(100):
        if f'radius_{i}' in param:
            quantity.append(f'radius_{i}')
            unit.append(r'$R_\mathregular{{J}}$')
            label.append(rf'$R_\mathregular{{{i+1}}}$')

        else:
            break

    if 'distance' in param:
        quantity.append('distance')
        unit.append('pc')
        label.append(r'$d$')

    if 'mass' in param:
        quantity.append('mass')

        if object_type == 'planet':
            unit.append(r'$M_\mathregular{J}$')
        elif object_type == 'star':
            unit.append(r'$M_\mathregular{\odot}$')

        label.append(r'$M$')

    if 'luminosity' in param:
        quantity.append('luminosity')
        unit.append(None)
        label.append(r'$\log\,L$/L$_\odot$')

    if 'dust_ext' in param:
        quantity.append('dust_ext')
        unit.append('mag')
        label.append('A$_\mathregular{V}$')

    return quantity, unit, label


def field_bounds_ticks(field_range):
    """
    Parameters
    ----------
    field_range : tuple(str, str), None
        Range of the discrete colorbar for the field dwarfs. The tuple should contain the lower
        and upper value ('early M', 'late M', 'early L', 'late L', 'early T', 'late T', 'early Y).
        The full range is used if set to None.

    Returns
    -------
    np.ndarray
    np.ndarray
    list(str, )
    """

    spectral_ranges = ['M0-M4', 'M5-M9', 'L0-L4', 'L5-L9', 'T0-T4', 'T5-T9', 'Y1-Y2']

    if field_range is None:
        index_start = 0
        index_end = 7

    else:
        if field_range[0] == 'early M':
            index_start = 0
        elif field_range[0] == 'late M':
            index_start = 1
        elif field_range[0] == 'early L':
            index_start = 2
        elif field_range[0] == 'late L':
            index_start = 3
        elif field_range[0] == 'early T':
            index_start = 4
        elif field_range[0] == 'late T':
            index_start = 5
        elif field_range[0] == 'early Y':
            index_start = 6

        if field_range[1] == 'early M':
            index_end = 1
        elif field_range[1] == 'late M':
            index_end = 2
        elif field_range[1] == 'early L':
            index_end = 3
        elif field_range[1] == 'late L':
            index_end = 4
        elif field_range[1] == 'early T':
            index_end = 5
        elif field_range[1] == 'late T':
            index_end = 6
        elif field_range[1] == 'early Y':
            index_end = 7

    index_range = index_end - index_start + 1

    bounds = np.linspace(index_start, index_end, index_range)
    ticks = np.linspace(index_start+0.5, index_end-0.5, index_range-1)
    labels = spectral_ranges[index_start:index_end]

    return bounds, ticks, labels


@typechecked
def remove_color_duplicates(object_names: List[str],
                            empirical_names: np.ndarray) -> List[int]:
    """"
    Function for deselecting young/low-gravity objects that will already be plotted individually
    as directly imaged objects.

    Parameters
    ----------
    object_names : list(str)
        List with names of directly imaged planets and brown dwarfs.
    empirical_names : np.ndarray
        Array with names of young/low-gravity objects.

    Returns
    -------
    list
        List with selected indices of the young/low-gravity objects.
    """

    indices = []

    for i, item in enumerate(empirical_names):
        if item == 'beta_Pic_b' and 'beta Pic b' in object_names:
            continue

        if item == 'HR8799b' and 'HR 8799 b' in object_names:
            continue

        if item == 'HR8799c' and 'HR 8799 c' in object_names:
            continue

        if item == 'HR8799d' and 'HR 8799 d' in object_names:
            continue

        if item == 'HR8799e' and 'HR 8799 e' in object_names:
            continue

        if item == 'kappa_And_B' and 'kappa And b' in object_names:
            continue

        if item == 'HD1160B' and 'HD 1160 B' in object_names:
            continue

        indices.append(i)

    return indices
