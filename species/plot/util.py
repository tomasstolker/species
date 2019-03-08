'''
Text.
'''

import numpy as np


def sptype_discrete(sptype,
                    shape):
    '''
    :param sptype:
    :type sptype:
    :param shape:
    :type shape:

    :return::
    :rtype: numpy.ndarray
    '''

    spt_disc = np.zeros(shape)

    for i, item in enumerate(sptype):
        sp = item[0:2]

        if sp in (np.string_('M0'), np.string_('M1'), np.string_('M2'), np.string_('M3'), np.string_('M4')):
            spt_disc[i] = 0

        elif sp in (np.string_('M5'), np.string_('M6'), np.string_('M7'), np.string_('M8'), np.string_('M9')):
            spt_disc[i] = 1

        elif sp in (np.string_('L0'), np.string_('L1'), np.string_('L2'), np.string_('L3'), np.string_('L4')):
            spt_disc[i] = 2

        elif sp in (np.string_('L5'), np.string_('L6'), np.string_('L7'), np.string_('L8'), np.string_('L9')):
            spt_disc[i] = 3

        elif sp in (np.string_('T0'), np.string_('T1'), np.string_('T2'), np.string_('T3'), np.string_('T4')):
            spt_disc[i] = 4

        elif sp in (np.string_('T5'), np.string_('T6'), np.string_('T7'), np.string_('T8'), np.string_('T9')):
            spt_disc[i] = 5

        elif np.string_('Y') in item:
            spt_disc[i] = 6

        else:
            spt_disc[i] = np.nan
            continue

    return spt_disc


def update_labels(param):
    '''
    :param param:
    :type param: list

    :return:
    :rtype: list
    '''

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
    '''
    :param key:
    :type key: str

    :return:
    :rtype: str
    '''

    if key == 'drift-phoenix':
        name = 'DRIFT-PHOENIX'

    elif key == 'bt-nextgen':
        name = 'BT-NextGen'

    return name


def quantity_unit(param):
    '''
    :param param:
    :type param: list

    :return:
    :rtype: list, list
    '''

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
