"""
Module for Exo-REM atmospheric model spectra.
"""

import os
import zipfile
import warnings
import urllib.request

import spectres
import numpy as np

from species.core import constants
from species.util import data_util


def add_exo_rem(input_path,
                database,
                data_folder,
                wavel_range=None,
                teff_range=None,
                spec_res=1000.):
    """
    Function for adding the Exo-REM atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    data_folder : str
        Path with input data.
    wavel_range : tuple(float, float), None
        Wavelength range (um). The original wavelength points are used if set to None.
    teff_range : tuple(float, float), None
        Effective temperature range (K). All temperatures are selected if set to None.
    spec_res : float, None
        Spectral resolution. Not used if ``wavel_range`` is set to None.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    param_file = os.path.join(data_folder, 'input_data_CO2.txt')

    par_teff, par_gravity, par_feh, par_co = np.loadtxt(param_file, unpack=True)

    par_logg = np.log10(par_gravity)  # log10(cm s-2)

    teff = []
    logg = []
    feh = []
    co_ratio = []
    flux = []

    if wavel_range is not None:
        wavelength = [wavel_range[0]]

        while wavelength[-1] <= wavel_range[1]:
            wavelength.append(wavelength[-1] + wavelength[-1]/spec_res)

        wavelength = np.asarray(wavelength[:-1])

    else:
        wavelength = None

    for _, _, files in os.walk(data_folder):
        for filename in files:
            if filename[:8] == 'spectre_':
                param_index = int(filename[8:].split('.')[0]) - 1

                teff_val = par_teff[param_index]
                logg_val = par_logg[param_index]
                feh_val = np.log10(par_feh[param_index])
                co_val = par_co[param_index]

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                print_message = f'Adding Exo-REM model spectra... {filename}'
                print(f'\r{print_message:<50}', end='')

                data = np.loadtxt(os.path.join(data_folder, filename))

                if data.shape[0] == 34979:
                    data = data[:-1, :]

                # change the order because of the conversion from wavenumber to wavelength
                data = data[::-1, :]

                teff.append(teff_val)
                logg.append(logg_val)
                feh.append(feh_val)
                co_ratio.append(co_val)

                if wavel_range is None:
                    if wavelength is None:
                        # (cm-1) -> (um)
                        wavelength = 1e4/data[:, 0]

                    if np.all(np.diff(wavelength) < 0):
                        raise ValueError('The wavelengths are not all sorted by increasing value.')

                    # (erg s-1 cm-2 cm) -> (W m-2 um-1) and include a factor pi
                    flux.append(np.pi*data[:, 1]*1e-7*1e8/wavelength**2)

                else:
                    # (cm-1) -> (um)
                    data_wavel = 1e4/data[:, 0]

                    # (erg s-1 cm-2 cm) -> (W m-2 um-1) and include a factor pi
                    data_flux = np.pi*data[:, 1]*1e-7*1e8/data_wavel**2

                    try:
                        flux.append(spectres.spectres(wavelength, data_wavel, data_flux))
                    except ValueError:
                        flux.append(np.zeros(wavelength.shape[0]))

                        warnings.warn('The wavelength range should fall within the range of the '
                                      'original wavelength sampling. Storing zeros instead.')

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      np.asarray(co_ratio),
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('exo-rem',
                         ['teff', 'logg', 'feh', 'co'],
                         database,
                         data_sorted)

    print_message = 'Adding Exo-REM model spectra... [DONE]'
    print(f'\r{print_message:<50}')
