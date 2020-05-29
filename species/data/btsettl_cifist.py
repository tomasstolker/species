"""
Module for BT-Settl atmospheric model spectra.
"""

import os
import tarfile
import warnings
import urllib.request

from typing import Optional, Tuple

import h5py
import spectres
import numpy as np

from typeguard import typechecked

from species.util import data_util, read_util


@typechecked
def add_btsettl(input_path: str,
                database: h5py._hl.files.File,
                wavel_range: Optional[Tuple[float, float]],
                teff_range: Optional[Tuple[float, float]],
                spec_res: Optional[float]):
    """
    Function for adding the BT-Settl-CIFIST atmospheric models (solar metallicity) to the database.
    The spectra had been downloaded from the Theoretical spectra web server
    (http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-settl-cifist) and resampled
    to a spectral resolution of 5000 from 0.1 to 100 um.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wavel_range : tuple(float, float), None
        Wavelength range (um). The original wavelength points are used if set to ``None``.
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

    input_file = 'bt-settl-cifist.tgz'
    label = '(578 MB)'

    data_folder = os.path.join(input_path, 'bt-settl-cifist/')
    data_file = os.path.join(data_folder, input_file)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    url = 'https://people.phys.ethz.ch/~ipa/tstolker/bt-settl-cifist.tgz'

    if not os.path.isfile(data_file):
        print(f'Downloading Bt-Settl model spectra {label}...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print(f'Unpacking BT-Settl model spectra {label}...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    teff = []
    logg = []
    flux = []

    if wavel_range is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):
            if filename[:16] == 'bt-settl-cifist_':
                file_split = filename.split('_')

                teff_val = float(file_split[2])
                logg_val = float(file_split[4])

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                print_message = f'Adding BT-Settl model spectra... {filename}'
                print(f'\r{print_message:<76}', end='')

                data_wavel, data_flux = np.loadtxt(os.path.join(data_folder, filename), unpack=True)

                teff.append(teff_val)
                logg.append(logg_val)

                if wavel_range is None:
                    if wavelength is None:
                        wavelength = np.copy(data_wavel)  # (um)

                    if np.all(np.diff(wavelength) < 0):
                        raise ValueError('The wavelengths are not all sorted by increasing value.')

                    flux.append(data_flux)  # (W m-2 um-1)

                else:
                    try:
                        flux_resample = spectres.spectres(wavelength, data_wavel, data_flux)
                        flux.append(flux_resample)  # (W m-2 um-1)

                    except ValueError:
                        flux.append(np.zeros(wavelength.shape[0]))  # (um)

                        warnings.warn('The wavelength range should fall within the range of the '
                                      'original wavelength sampling. Storing zeros instead.')

    print_message = 'Adding BT-Settl model spectra... [DONE]'
    print(f'\r{print_message:<76}')

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      None,
                                      None,
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('bt-settl-cifist',
                         ['teff', 'logg'],
                         database,
                         data_sorted)
