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
        if item[0:2] in (b'M0', b'M1', b'M2', b'M3', b'M4'):
            spt_disc[i] = 0.5

        elif item[0:2] in (b'M5', b'M6', b'M7', b'M8', b'M9'):
            spt_disc[i] = 1.5

        elif item[0:2] in (b'L0', b'L1', b'L2', b'L3', b'L4'):
            spt_disc[i] = 2.5

        elif item[0:2] in (b'L5', b'L6', b'L7', b'L8', b'L9'):
            spt_disc[i] = 3.5

        elif item[0:2] in (b'T0', b'T1', b'T2', b'T3', b'T4'):
            spt_disc[i] = 4.5

        elif item[0:2] in (b'T5', b'T6', b'T7', b'T8', b'T9'):
            spt_disc[i] = 5.5

        elif b'Y' in item:
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
        if item[0] == b'O':
            spt_disc[i] = 0.5

        elif item[0] == b'B':
            spt_disc[i] = 1.5

        elif item[0] == b'A':
            spt_disc[i] = 2.5

        elif item[0] == b'F':
            spt_disc[i] = 3.5

        elif item[0] == b'G':
            spt_disc[i] = 4.5

        elif item[0] == b'K':
            spt_disc[i] = 5.5

        elif item[0] == b'M':
            spt_disc[i] = 6.5

        elif item[0] == b'L':
            spt_disc[i] = 7.5

        elif item[0] == b'T':
            spt_disc[i] = 8.5

        elif item[0] == b'Y':
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
        param[index] = r'$T_\mathregular{eff}$ [K]'

    if 'logg' in param:
        index = param.index('logg')
        param[index] = r'$\log\,g$ [dex]'

    if 'feh' in param:
        index = param.index('feh')
        param[index] = r'Fe/H [dex]'

    if 'fsed' in param:
        index = param.index('fsed')
        param[index] = r'f$_\mathregular{sed}$'

    if 'co' in param:
        index = param.index('co')
        param[index] = r'C/O'

    if 'radius' in param:
        index = param.index('radius')
        param[index] = r'$R$ [$\mathregular{R_{Jup}}$]'

    if 'scaling' in param:
        index = param.index('scaling')
        param[index] = 'Scaling'

    if 'offset' in param:
        index = param.index('offset')
        param[index] = 'Offset'

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
    """

    quantity = []
    unit = []

    if 'teff' in param:
        quantity.append(r'$T_\mathregular{eff}$')
        unit.append('K')

    if 'logg' in param:
        quantity.append(r'$\log\,g$')
        unit.append('dex')

    if 'feh' in param:
        quantity.append(r'[Fe/H]')
        unit.append('dex')

    if 'fsed' in param:
        quantity.append(r'f$_\mathregular{sed}$')
        unit.append(None)

    if 'co' in param:
        quantity.append(r'C/O')
        unit.append(None)

    if 'radius' in param:
        quantity.append(r'$R$')

        if object_type == 'planet':
            unit.append(r'$R_\mathregular{Jup}}$')
        elif object_type == 'star':
            unit.append(r'$R_\mathregular{\odot}}$')

    if 'distance' in param:
        quantity.append(r'$d$')
        unit.append('pc')

    if 'mass' in param:
        quantity.append(r'$M$')

        if object_type == 'planet':
            unit.append(r'$M_\mathregular{Jup}$')
        elif object_type == 'star':
            unit.append(r'$M_\mathregular{\odot}$')

    if 'luminosity' in param:
        quantity.append(r'$L$')
        unit.append(r'$L_\odot$')

    return quantity, unit


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
