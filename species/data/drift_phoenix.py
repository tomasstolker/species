"""
Module for DRIFT-PHOENIX atmospheric model spectra.
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
def add_drift_phoenix(input_path: str,
                      database: h5py._hl.files.File,
                      wavel_range: Optional[Tuple[float, float]] = None,
                      teff_range: Optional[Tuple[float, float]] = None,
                      spec_res: Optional[float] = 1000.) -> None:
    """
    Function for adding the DRIFT-PHOENIX atmospheric models to the database.

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

    data_file = os.path.join(input_path, 'drift-phoenix.tgz')
    data_folder = os.path.join(input_path, 'drift-phoenix/')

    url = 'https://people.phys.ethz.ch/~ipa/tstolker/drift-phoenix.tgz'

    if not os.path.isfile(data_file):
        print('Downloading DRIFT-PHOENIX model spectra (151 MB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Unpacking DRIFT-PHOENIX model spectra...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(input_path)
    tar.close()
    print(' [DONE]')

    teff = []
    logg = []
    feh = []
    flux = []

    if wavel_range is not None and spec_res is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte_'):
                teff_val = float(filename[4:8])
                logg_val = float(filename[9:12])
                feh_val = float(filename[12:16])

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                print_message = f'Adding DRIFT-PHOENIX model spectra... {filename}'
                print(f'\r{print_message:<65}', end='')

                data = np.loadtxt(data_folder+filename)

                teff.append(teff_val)
                logg.append(logg_val)
                feh.append(feh_val)

                if wavel_range is None:
                    if wavelength is None:
                        # (Angstrom) -> (um)
                        wavelength = data[:, 0]*1e-4

                    if np.all(np.diff(wavelength) < 0):
                        raise ValueError('The wavelengths are not all sorted by increasing value.')

                    # (erg s-1 cm-2 Angstrom-1) -> (W m-2 um-1)
                    flux.append(data[:, 1]*1e-7*1e4*1e4)

                else:
                    # (Angstrom) -> (um)
                    data_wavel = data[:, 0]*1e-4

                    # (erg s-1 cm-2 Angstrom-1) -> (W m-2 um-1)
                    data_flux = data[:, 1]*1e-7*1e4*1e4

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

    print_message = 'Adding DRIFT-PHOENIX model spectra... [DONE]'
    print(f'\r{print_message:<65}')

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      None,
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('drift-phoenix',
                         ['teff', 'logg', 'feh'],
                         database,
                         data_sorted)
