"""
Module for petitCODE atmospheric model spectra.
"""

import os
import zipfile
import warnings
import urllib.request

import spectres
import numpy as np

from species.core import constants
from species.util import data_util, read_util


def add_petitcode_cool_clear(input_path,
                             database,
                             wavel_range=None,
                             teff_range=None,
                             spec_res=1000.):
    """
    Function for adding the petitCODE cool clear atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
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

    data_folder = os.path.join(input_path, 'linder_molliere_grid/clear/specs/')

    url = 'http://mpia.de/~molliere/online_data/linder_molliere_grid.zip'

    data_file = os.path.join(input_path, 'linder_molliere_grid.zip')

    if not os.path.isfile(data_file):
        print('Downloading petitCODE cool clear model spectra (3.7 GB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Unpacking petitCODE cool clear model spectra...', end='', flush=True)

    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(input_path)

    print(' [DONE]')

    teff = []
    logg = []
    feh = []
    flux = []

    if wavel_range is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    for _, _, files in os.walk(data_folder):
        for filename in files:
            file_split = filename.split('_')

            teff_val = float(file_split[2])
            logg_val = float(file_split[4])
            feh_val = float(file_split[6])

            if teff_range is not None:
                if teff_val < teff_range[0] or teff_val > teff_range[1]:
                    continue

            print_message = f'Adding petitCODE cool clear model spectra... {filename}'
            print(f'\r{print_message:<87}', end='')

            data = np.loadtxt(os.path.join(data_folder, filename))

            teff.append(teff_val)
            logg.append(logg_val)
            feh.append(feh_val)

            if wavel_range is None:
                if wavelength is None:
                    # (cm) -> (um)
                    wavelength = data[:, 0]*1e4

                if np.all(np.diff(wavelength) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

            else:
                # (cm) -> (um)
                data_wavel = data[:, 0]*1e4

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                data_flux = data[:, 1]*1e-9*constants.LIGHT/(data_wavel*1e-6)**2

                try:
                    flux.append(spectres.spectres(wavelength, data_wavel, data_flux))
                except ValueError:
                    flux.append(np.zeros(wavelength.shape[0]))

                    warnings.warn('The wavelength range should fall within the range of the '
                                  'original wavelength sampling. Storing zeros instead.')

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      None,
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('petitcode-cool-clear',
                         ['teff', 'logg', 'feh'],
                         database,
                         data_sorted)

    print_message = 'Adding petitCODE cool clear model spectra... [DONE]'
    print(f'\r{print_message:<87}')


def add_petitcode_cool_cloudy(input_path,
                              database,
                              wavel_range=None,
                              teff_range=None,
                              spec_res=1000.):
    """
    Function for adding the petitCODE cool cloudy atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
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

    data_folder = os.path.join(input_path, 'linder_molliere_grid/cloudy/specs/')

    url = 'http://mpia.de/~molliere/online_data/linder_molliere_grid.zip'

    data_file = os.path.join(input_path, 'linder_molliere_grid.zip')

    if not os.path.isfile(data_file):
        print('Downloading petitCODE cool cloudy model spectra (3.7 GB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Unpacking petitCODE cool cloudy model spectra...', end='', flush=True)

    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(input_path)

    print(' [DONE]')

    teff = []
    logg = []
    feh = []
    fsed = []
    flux = []

    if wavel_range is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    for _, _, files in os.walk(data_folder):
        for filename in files:
            file_split = filename.split('_')

            teff_val = float(file_split[2])
            logg_val = float(file_split[4])
            feh_val = float(file_split[6])
            fsed_val = float(file_split[8])

            if teff_range is not None:
                if teff_val < teff_range[0] or teff_val > teff_range[1]:
                    continue

            print_message = f'Adding petitCODE cool cloudy model spectra... {filename}'
            print(f'\r{print_message:<106}', end='')

            data = np.loadtxt(os.path.join(data_folder, filename))

            teff.append(teff_val)
            logg.append(logg_val)
            feh.append(feh_val)
            fsed.append(fsed_val)

            if wavel_range is None:
                if wavelength is None:
                    # (cm) -> (um)
                    wavelength = data[:, 0]*1e4

                if np.all(np.diff(wavelength) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

            else:
                # (cm) -> (um)
                data_wavel = data[:, 0]*1e4

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                data_flux = data[:, 1]*1e-9*constants.LIGHT/(data_wavel*1e-6)**2

                try:
                    flux.append(spectres.spectres(wavelength, data_wavel, data_flux))
                except ValueError:
                    flux.append(np.zeros(wavelength.shape[0]))

                    warnings.warn('The wavelength range should fall within the range of the '
                                  'original wavelength sampling. Storing zeros instead.')

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      None,
                                      np.asarray(fsed),
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('petitcode-cool-cloudy',
                         ['teff', 'logg', 'feh', 'fsed'],
                         database,
                         data_sorted)

    print_message = 'Adding petitCODE cool cloudy model spectra... [DONE]'
    print(f'\r{print_message:<106}')


def add_petitcode_hot_clear(input_path,
                            database,
                            data_folder,
                            wavel_range=None,
                            teff_range=None,
                            spec_res=1000.):
    """
    Function for adding the petitCODE hot clear atmospheric models to the database.

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

    teff = []
    logg = []
    feh = []
    co_ratio = []
    flux = []

    if wavel_range is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    for _, _, files in os.walk(data_folder):
        for filename in files:
            teff_val = float(filename[9:13])
            logg_val = float(filename[19:23])
            feh_val = float(filename[28:32])
            co_ratio_val = float(filename[36:40])

            if teff_range is not None:
                if teff_val < teff_range[0] or teff_val > teff_range[1]:
                    continue

            print_message = f'Adding petitCODE hot clear model spectra... {filename}'
            print(f'\r{print_message:<100}', end='')

            data = np.loadtxt(os.path.join(data_folder, filename))

            teff.append(teff_val)
            logg.append(logg_val)
            feh.append(feh_val)
            co_ratio.append(co_ratio_val)

            if wavel_range is None:
                if wavelength is None:
                    # (cm) -> (um)
                    wavelength = data[:, 0]*1e4

                if np.all(np.diff(wavelength) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

            else:
                # (cm) -> (um)
                data_wavel = data[:, 0]*1e4

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                data_flux = data[:, 1]*1e-9*constants.LIGHT/(data_wavel*1e-6)**2

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

    data_util.write_data('petitcode-hot-clear',
                         ['teff', 'logg', 'feh', 'co'],
                         database,
                         data_sorted)

    print_message = 'Adding petitCODE hot clear model spectra... [DONE]'
    print(f'\r{print_message:<100}')


def add_petitcode_hot_cloudy(input_path,
                             database,
                             data_folder,
                             wavel_range=None,
                             teff_range=None,
                             spec_res=1000.):
    """
    Function for adding the petitCODE hot cloudy atmospheric models to the database.

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

    teff = []
    logg = []
    feh = []
    co_ratio = []
    fsed = []
    flux = []

    if wavel_range is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    for _, _, files in os.walk(data_folder):
        for filename in files:
            teff_val = float(filename[9:13])
            logg_val = float(filename[19:23])
            feh_val = float(filename[28:32])
            co_ratio_val = float(filename[36:40])
            fsed_val = float(filename[46:50])

            if teff_range is not None:
                if teff_val < teff_range[0] or teff_val > teff_range[1]:
                    continue

            print_message = f'Adding petitCODE hot cloudy model spectra... {filename}'
            print(f'\r{print_message:<112}', end='')

            data = np.loadtxt(os.path.join(data_folder, filename))

            teff.append(teff_val)
            logg.append(logg_val)
            feh.append(feh_val)
            co_ratio.append(co_ratio_val)
            fsed.append(fsed_val)

            if wavel_range is None:
                if wavelength is None:
                    # (cm) -> (um)
                    wavelength = data[:, 0]*1e4

                if np.all(np.diff(wavelength) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

            else:
                # (cm) -> (um)
                data_wavel = data[:, 0]*1e4

                # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                data_flux = data[:, 1]*1e-9*constants.LIGHT/(data_wavel*1e-6)**2

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
                                      np.asarray(fsed),
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('petitcode-hot-cloudy',
                         ['teff', 'logg', 'feh', 'co', 'fsed'],
                         database,
                         data_sorted)

    print_message = 'Adding petitCODE hot cloudy model spectra... [DONE]'
    print(f'\r{print_message:<112}')
