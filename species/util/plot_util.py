"""
Utility functions for plotting data.
"""

import numpy as np


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


def update_labels(param):
    """
    Parameters
    ----------
    param : list

    Returns
    -------
    list
    """

    if 'teff' in param:
        index = param.index('teff')
        param[index] = r'$T_\mathregular{eff}$ (K)'

    if 'logg' in param:
        index = param.index('logg')
        param[index] = r'$\log\,g$ (dex)'

    if 'feh' in param:
        index = param.index('feh')
        param[index] = r'[Fe/H] (dex)'

    if 'fsed' in param:
        index = param.index('fsed')
        param[index] = r'f$_\mathregular{sed}$'

    if 'co' in param:
        index = param.index('co')
        param[index] = r'C/O'

    if 'radius' in param:
        index = param.index('radius')
        param[index] = r'$R$ ($\mathregular{R_{Jup}}$)'

    if 'teff' in param:
        index = param.index('teff')
        param[index] = r'$T_\mathregular{eff}$ (K)'

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
        param[index] = r'$\log\,\delta$ (dex)'

    if 'log_p_quench' in param:
        index = param.index('log_p_quench')
        param[index] = r'$\log\,P_\mathregular{quench}$ (dex)'

    for i, item in enumerate(param):
        if item[0:8] == 'scaling_':
            param[i] = rf'$a_\mathregular{{{item[8:]}}}$'

        elif item[0:8] == 'error_':
            param[i] = rf'$b_\mathregular{{{item[8:]}}}$'

    return param


def model_name(key):
    """
    Parameters
    ----------
    key : str

    Returns
    -------
    str
    """

    if key == 'drift-phoenix':
        name = 'DRIFT-PHOENIX'

    elif key == 'ames-cond':
        name = 'AMES-Cond'

    elif key == 'ames-dusty':
        name = 'AMES-Dusty'

    elif key == 'bt-settl':
        name = 'BT-Settl'

    elif key == 'bt-nextgen':
        name = 'BT-NextGen'

    elif key == 'planck':
        name = 'Planck radiation'

    elif key == 'zhu2015':
        name = 'Zhu (2015)'

    return name


def quantity_unit(param,
                  object_type):
    """
    Parameters
    ----------
    param : list
    object_type : str

    Returns
    -------
    list
    list
    list
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
        unit.append('dex')
        label.append(r'$\log\,g$')

    if 'feh' in param:
        quantity.append('feh')
        unit.append('dex')
        label.append(r'[Fe/H]')

    if 'fsed' in param:
        quantity.append('fsed')
        unit.append(None)
        label.append(r'f$_\mathregular{sed}$')

    if 'co' in param:
        quantity.append('co')
        unit.append(None)
        label.append(r'C/O')

    if 'radius' in param:
        quantity.append('radius')

        if object_type == 'planet':
            unit.append(r'$R_\mathregular{Jup}}$')
        elif object_type == 'star':
            unit.append(r'$R_\mathregular{\odot}}$')

        label.append(r'$R$')

    if 'distance' in param:
        quantity.append('distance')
        unit.append('pc')
        label.append(r'$d$')

    if 'mass' in param:
        quantity.append('mass')

        if object_type == 'planet':
            unit.append(r'$M_\mathregular{Jup}$')
        elif object_type == 'star':
            unit.append(r'$M_\mathregular{\odot}$')

        label.append(r'$M$')

    if 'luminosity' in param:
        quantity.append('luminosity')
        unit.append(r'$L_\odot$')
        label.append(r'$L$')

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
