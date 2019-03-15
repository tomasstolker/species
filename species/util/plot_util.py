"""
Text.
"""

import numpy as np


def sptype_substellar(sptype,
                      shape):
    """
    Args:
        sptype
        shape

    Returns:
        numpy.ndarray:
    """

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        sp = item[0:2]

        if sp in (np.string_('M0'), np.string_('M1'), np.string_('M2'), np.string_('M3'), np.string_('M4')):
            spt_disc[i] = 0.5

        elif sp in (np.string_('M5'), np.string_('M6'), np.string_('M7'), np.string_('M8'), np.string_('M9')):
            spt_disc[i] = 1.5

        elif sp in (np.string_('L0'), np.string_('L1'), np.string_('L2'), np.string_('L3'), np.string_('L4')):
            spt_disc[i] = 2.5

        elif sp in (np.string_('L5'), np.string_('L6'), np.string_('L7'), np.string_('L8'), np.string_('L9')):
            spt_disc[i] = 3.5

        elif sp in (np.string_('T0'), np.string_('T1'), np.string_('T2'), np.string_('T3'), np.string_('T4')):
            spt_disc[i] = 4.5

        elif sp in (np.string_('T5'), np.string_('T6'), np.string_('T7'), np.string_('T8'), np.string_('T9')):
            spt_disc[i] = 5.5

        elif np.string_('Y') in item:
            spt_disc[i] = 6.5

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc


def sptype_stellar(sptype,
                   shape):
    """
    Args:
        sptype
        shape

    Returns:
        numpy.ndarray
    """

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        if str(item)[2] == 'O':
            spt_disc[i] = 0.5

        elif str(item)[2] == 'B':
            spt_disc[i] = 1.5

        elif str(item)[2] == 'A':
            spt_disc[i] = 2.5

        elif str(item)[2] == 'F':
            spt_disc[i] = 3.5

        elif str(item)[2] == 'G':
            spt_disc[i] = 4.5

        elif str(item)[2] == 'K':
            spt_disc[i] = 5.5

        elif str(item)[2] == 'M':
            spt_disc[i] = 6.5

        elif str(item)[2] == 'L':
            spt_disc[i] = 7.5

        elif str(item)[2] == 'T':
            spt_disc[i] = 8.5

        elif str(item)[2] == 'Y':
            spt_disc[i] = 9.5

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc


def update_labels(param):
    """
    Args:
        param(list): 

    Returns:
        list:
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
    Args:
        key(str):

    Returns:
        str:
    """

    if key == 'drift-phoenix':
        name = 'DRIFT-PHOENIX'

    elif key == 'bt-nextgen':
        name = 'BT-NextGen'

    return name


def quantity_unit(param):
    """
    Args:
        param(list):

    Returns:
        list:
        list:
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

    if 'radius' in param:
        quantity.append(r'$R$')
        unit.append(r'$R_\mathregular{Jup}}$')

    if 'distance' in param:
        quantity.append(r'$d$')
        unit.append('pc')

    if 'mass' in param:
        quantity.append(r'$M$')
        unit.append(r'$M_\mathregular{Jup}$')

    if 'luminosity' in param:
        quantity.append(r'$L$')
        unit.append(r'$L_\odot$')

    return quantity, unit
