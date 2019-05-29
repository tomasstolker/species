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
        if item[0:2] in ('M0', 'M1', 'M2',
                         'M3', 'M4'):

            spt_disc[i] = 0.5

        elif item[0:2] in ('M5', 'M6', 'M7',
                           'M8', 'M9'):

            spt_disc[i] = 1.5

        elif item[0:2] in ('L0', 'L1', 'L2',
                           'L3', 'L4'):

            spt_disc[i] = 2.5

        elif item[0:2] in ('L5', 'L6', 'L7',
                           'L8', 'L9'):

            spt_disc[i] = 3.5

        elif item[0:2] in ('T0', 'T1', 'T2',
                           'T3', 'T4'):

            spt_disc[i] = 4.5

        elif item[0:2] in ('T5', 'T6', 'T7',
                           'T8', 'T9'):

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
        if str(item)[0] == 'O':
            spt_disc[i] = 0.5

        elif str(item)[0] == 'B':
            spt_disc[i] = 1.5

        elif str(item)[0] == 'A':
            spt_disc[i] = 2.5

        elif str(item)[0] == 'F':
            spt_disc[i] = 3.5

        elif str(item)[0] == 'G':
            spt_disc[i] = 4.5

        elif str(item)[0] == 'K':
            spt_disc[i] = 5.5

        elif str(item)[0] == 'M':
            spt_disc[i] = 6.5

        elif str(item)[0] == 'L':
            spt_disc[i] = 7.5

        elif str(item)[0] == 'T':
            spt_disc[i] = 8.5

        elif str(item)[0] == 'Y':
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

    elif key == 'bt-nextgen':
        name = 'BT-NextGen'

    return name


def quantity_unit(param):
    """
    Parameters
    ----------
    param : list

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
