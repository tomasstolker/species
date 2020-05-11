"""
Utility functions for plotting data.
"""

import os
import configparser

from typing import Optional, Tuple, List

import h5py
import PyMieScatt
import numpy as np

from typeguard import typechecked
from scipy.interpolate import interp1d

from species.data import database
from species.read import read_filter


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
    Function for updating the fitted parameters to labels used in plots.

    Parameters
    ----------
    param : list
        List with parameter names.

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

    if 'feh' in param:
        index = param.index('feh')
        param[index] = r'[Fe/H]'

    if 'fsed' in param:
        index = param.index('fsed')
        param[index] = r'f$_\mathregular{sed}$'

    if 'co' in param:
        index = param.index('co')
        param[index] = r'C/O'

    if 'radius' in param:
        index = param.index('radius')
        param[index] = r'$R$ ($\mathregular{R_{J}}$)'

    if 'luminosity' in param:
        index = param.index('luminosity')
        param[index] = r'$\log\,L$/L$_\odot$'

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
def model_name(key) -> str:
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

    if 'feh' in param:
        quantity.append('feh')
        unit.append(None)
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
            unit.append(rf'$R_\mathregular{{J}}$')
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
def dust_cross_section(wavelengths: np.ndarray,
                       n_index: np.ndarray,
                       k_index: np.ndarray,
                       radius: float) -> np.ndarray:
    """
    Function for calculating the extinction cross section of a dust grain.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelengths (um).
    n_index : np.ndarray
        Real part of the refractive index.
    k_index : np.ndarray
        Imaginary part of the refractive index.
    radius : np.ndarray
        Radius of the dust grain (um).

    Returns
    -------
    np.ndarray
        Extinction cross section (um2)
    """

    sigma = np.zeros(wavelengths.shape)

    for i, item in enumerate(wavelengths):
        # PyMieScatt units are in nm and the grain size is provided as the diameter
        mie = PyMieScatt.MieQ(complex(n_index[i], k_index[i]), item*1e3, 2.*radius*1e3,
                              asDict=True, asCrossSection=True)

        if 'Cext' in mie:
            sigma[i] = mie['Cext']  # (nm2)

    return sigma*1e-6  # (um2)


@typechecked
def calc_reddening(filters_color: Tuple[str, str],
                   extinction: Tuple[str, float],
                   composition: str = 'MgSiO3',
                   structure: str = 'crystalline',
                   radius: float = 1.) -> Tuple[float, float]:
    """
    Function for calculating the reddening of a color given the extinction for a given filter.

    Parameters
    ----------
    filters_color : tuple(str, str)
        Filter names for which the extinction is calculated.
    extinction : str
        Filter name and extinction (mag).
    composition : str
        Dust composition ('MgSiO3' or 'Fe').
    structure : str
        Grain structure ('crystalline' or 'amorphous').
    radius : float
        Radius of the dust grain (um).

    Returns
    -------
    float
        Extinction (mag) for ``filters_color[0]``.
    float
        Extinction (mag) for ``filters_color[1]``.
    """

    config_file = os.path.join(os.getcwd(), 'species_config.ini')

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    database_path = config['species']['database']

    h5_file = h5py.File(database_path, 'r')

    try:
        h5_file['dust']

    except KeyError:
        h5_file.close()
        species_db = database.Database()
        species_db.add_dust()
        h5_file = h5py.File(database_path, 'r')

    if composition == 'MgSiO3' and structure == 'crystalline':
        for i in range(3):
            data = h5_file[f'dust/mgsio3/crystalline/axis_{i+1}']

            # Average cross section of the three axes
            if i == 0:
                sigma = dust_cross_section(data[:, 0], data[:, 1], data[:, 2], radius) / 3.

            else:
                sigma += dust_cross_section(data[:, 0], data[:, 1], data[:, 2], radius) / 3.

    else:
        if composition == 'MgSiO3' and structure == 'amorphous':
            data = h5_file[f'dust/mgsio3/amorphous/']

        elif composition == 'Fe' and structure == 'crystalline':
            data = h5_file[f'dust/fe/crystalline/']

        elif composition == 'Fe' and structure == 'amorphous':
            data = h5_file[f'dust/fe/amorphous/']

        sigma = dust_cross_section(data[:, 0], data[:, 1], data[:, 2], radius)

    interp_sigma = interp1d(data[:, 0], sigma, kind='linear')

    h5_file.close()

    read_filt = read_filter.ReadFilter(extinction[0])
    transmission = read_filt.get_filter()

    # Weighted average of the cross section for extinction[0]
    sigma_mag = np.trapz(interp_sigma(transmission[:, 0])*transmission[:, 1],
                         transmission[:, 0]) / np.trapz(transmission[:, 1], transmission[:, 0])

    read_filt = read_filter.ReadFilter(filters_color[0])
    transmission = read_filt.get_filter()

    # Weighted average of the cross section for filters_color[0]
    sigma_color_0 = np.trapz(interp_sigma(transmission[:, 0])*transmission[:, 1],
                             transmission[:, 0]) / np.trapz(transmission[:, 1], transmission[:, 0])

    read_filt = read_filter.ReadFilter(filters_color[1])
    transmission = read_filt.get_filter()

    # Weighted average of the cross section for filters_color[1]
    sigma_color_1 = np.trapz(interp_sigma(transmission[:, 0])*transmission[:, 1],
                             transmission[:, 0]) / np.trapz(transmission[:, 1], transmission[:, 0])

    density = extinction[1]/sigma_mag/2.5/np.log10(np.exp(1.))

    return 2.5 * np.log10(np.exp(1.)) * sigma_color_0 * density, \
        2.5 * np.log10(np.exp(1.)) * sigma_color_1 * density
