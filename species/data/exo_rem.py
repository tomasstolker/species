"""
Module for Exo-REM atmospheric model spectra.
"""

import os
import tarfile
import urllib.request

from typing import Optional, Tuple

import h5py
import spectres
import numpy as np

from typeguard import typechecked

from species.util import data_util, read_util


@typechecked
def add_exo_rem(input_path: str,
                database: h5py._hl.files.File,
                wavel_range: Optional[Tuple[float, float]] = None,
                teff_range: Optional[Tuple[float, float]] = None,
                spec_res: Optional[float] = None) -> None:
    """
    Function for adding the Exo-REM atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wavel_range : tuple(float, float), None
        Wavelength range (um). The original wavelength points with a spectral resolution of 5000
        are used if set to ``None``.
    teff_range : tuple(float, float), None
        Effective temperature range (K). All temperatures are selected if set to ``None``.
    spec_res : float, None
        Spectral resolution. Not used if ``wavel_range`` is set to ``None``.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    input_file = 'exo-rem.tgz'
    url = 'https://home.strw.leidenuniv.nl/~stolker/species/exo-rem.tgz'

    data_folder = os.path.join(input_path, 'exo-rem/')
    data_file = os.path.join(input_path, input_file)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.isfile(data_file):
        print('Downloading Exo-REM model spectra (706 MB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Unpacking Exo-REM model spectra (706 MB)...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    teff = []
    logg = []
    feh = []
    co_ratio = []
    flux = []

    if wavel_range is not None and spec_res is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    for _, _, files in os.walk(data_folder):
        for filename in files:
            if filename[:8] == 'exo-rem_':
                file_split = filename.split('_')

                teff_val = float(file_split[2])
                logg_val = float(file_split[4])
                feh_val = float(file_split[6])
                co_val = float(file_split[8])

                if logg_val == 5.:
                    continue

                if co_val in [0.8, 0.85]:
                    continue

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                print_message = f'Adding Exo-REM model spectra... {filename}'
                print(f'\r{print_message:<84}', end='')

                data_wavel, data_flux = np.loadtxt(os.path.join(data_folder, filename), unpack=True)

                teff.append(teff_val)
                logg.append(logg_val)
                feh.append(feh_val)
                co_ratio.append(co_val)

                if wavel_range is None or spec_res is None:
                    if wavelength is None:
                        wavelength = np.copy(data_wavel)  # (um)

                    if np.all(np.diff(wavelength) < 0):
                        raise ValueError('The wavelengths are not all sorted by increasing value.')

                    flux.append(data_flux)  # (W m-2 um-1)

                else:
                    flux_resample = spectres.spectres(wavelength,
                                                      data_wavel,
                                                      data_flux,
                                                      spec_errs=None,
                                                      fill=np.nan,
                                                      verbose=False)

                    if np.isnan(np.sum(flux_resample)):
                        raise ValueError(f'Resampling is only possible if the new wavelength '
                                         f'range ({wavelength[0]} - {wavelength[-1]} um) falls '
                                         f'sufficiently far within the wavelength range '
                                         f'({data_wavel[0]} - {data_wavel[-1]} um) of the input '
                                         f'spectra.')

                    flux.append(flux_resample)  # (W m-2 um-1)

    print_message = 'Adding Exo-REM model spectra... [DONE]'
    print(f'\r{print_message:<84}')

    print('Grid points with the following parameters have been excluded:')
    print('   - log(g) = 5')
    print('   - C/O = 0.8')
    print('   - C/O = 0.85')

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
