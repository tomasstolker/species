"""
Module for petitCODE atmospheric models.
"""

import os
import zipfile

import wget
import numpy as np

from species.core import constants
from species.util import data_util


def add_petitcode_cool_clear(input_path,
                             database,
                             wl_bound,
                             teff_bound,
                             specres):
    """
    Function for adding the petitCODE cool clear atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wl_bound : tuple(float, float)
        Wavelength range (micron).
    teff_bound : tuple(float, float), None
        Effective temperature range (K).
    specres : float
        Spectral resolution.

    Returns
    -------
    NoneType
        None
    """

    if not wl_bound:
        wl_bound = (1e-2, 1e2)

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    data_folder = os.path.join(input_path, 'linder_molliere_grid/clear/specs/')

    input_file = 'linder_molliere_grid.zip'

    url = 'http://mpia.de/~molliere/online_data/linder_molliere_grid.zip'

    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print('Downloading petitCODE cool clear model spectra...', end='')
        wget.download(url, out=data_file, bar=None)
        print(' [DONE]')

    print('Unpacking petitCODE cool clear model spectra...', end='')

    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(input_path)

    print(' [DONE]')

    teff = []
    logg = []
    feh = []
    wavelength = None
    flux = []

    for _, _, files in os.walk(data_folder):
        for filename in files:
            print_message = f'Adding petitCODE cool clear model spectra... {filename}'
            print(f'\r{print_message:<110}', end='')

            file_split = filename.split('_')

            teff.append(float(file_split[2]))
            logg.append(float(file_split[4]))
            feh.append(float(file_split[6]))

            data = np.loadtxt(os.path.join(data_folder, filename))

            if wavelength is None:
                # [cm] -> [micron]
                wavelength = data[:, 0]*1e4

            if np.all(np.diff(wavelength) < 0):
                raise ValueError('The wavelengths are not all sorted by increasing value.')

            # [erg s-1 cm-2 Hz-1] -> [W m-2 micron-1]
            flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      None,
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('petitcode-cool-clear', ('teff', 'logg', 'feh'), database, data_sorted)

    print_message = 'Adding petitCODE cool clear model spectra... [DONE]'
    print(f'\r{print_message:<110}')


def add_petitcode_cool_cloudy(input_path,
                              database,
                              wl_bound,
                              teff_bound,
                              specres):
    """
    Function for adding the petitCODE cool cloudy atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wl_bound : tuple(float, float)
        Wavelength range (micron).
    teff_bound : tuple(float, float), None
        Effective temperature range (K).
    specres : float
        Spectral resolution.

    Returns
    -------
    NoneType
        None
    """

    if not wl_bound:
        wl_bound = (1e-2, 1e2)

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    data_folder = os.path.join(input_path, 'linder_molliere_grid/cloudy/specs/')

    input_file = 'linder_molliere_grid.zip'

    url = 'http://mpia.de/~molliere/online_data/linder_molliere_grid.zip'

    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print('Downloading petitCODE cool cloudy model spectra...', end='')
        wget.download(url, out=data_file, bar=None)
        print(' [DONE]')

    print('Unpacking petitCODE cool cloudy model spectra...', end='')

    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(input_path)

    print(' [DONE]')

    teff = []
    logg = []
    feh = []
    fsed = []
    wavelength = None
    flux = []

    for _, _, files in os.walk(data_folder):
        for filename in files:
            print_message = f'Adding petitCODE cool cloudy model spectra... {filename}'
            print(f'\r{print_message:<110}', end='')

            file_split = filename.split('_')

            teff.append(float(file_split[2]))
            logg.append(float(file_split[4]))
            feh.append(float(file_split[6]))
            fsed.append(float(file_split[8]))

            data = np.loadtxt(os.path.join(data_folder, filename))

            if wavelength is None:
                # [cm] -> [micron]
                wavelength = data[:, 0]*1e4

            if np.all(np.diff(wavelength) < 0):
                raise ValueError('The wavelengths are not all sorted by increasing value.')

            # [erg s-1 cm-2 Hz-1] -> [W m-2 micron-1]
            flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      None,
                                      np.asarray(fsed),
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('petitcode-cool-cloudy', ('teff', 'logg', 'feh', 'fsed'), database, data_sorted)

    print_message = 'Adding petitCODE cool cloudy model spectra... [DONE]'
    print(f'\r{print_message:<110}')


def add_petitcode_hot_clear(input_path,
                            database,
                            wl_bound,
                            teff_bound,
                            specres,
                            data_folder):
    """
    Function for adding the petitCODE hot clear atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wl_bound : tuple(float, float)
        Wavelength range (micron).
    teff_bound : tuple(float, float), None
        Effective temperature range (K).
    specres : float
        Spectral resolution.
    data_folder : str
        Path with input data.

    Returns
    -------
    NoneType
        None
    """

    if not wl_bound:
        wl_bound = (1e-2, 1e2)

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    teff = []
    logg = []
    feh = []
    co_ratio = []
    wavelength = None
    flux = []

    for _, _, files in os.walk(data_folder):
        for filename in files:
            print_message = f'Adding petitCODE hot clear model spectra... {filename}'
            print(f'\r{print_message:<110}', end='')

            teff.append(float(filename[9:13]))
            logg.append(float(filename[19:23]))
            feh.append(float(filename[28:32]))
            co_ratio.append(float(filename[36:40]))

            data = np.loadtxt(os.path.join(data_folder, filename))

            if wavelength is None:
                # [cm] -> [micron]
                wavelength = data[:, 0]*1e4

            if np.all(np.diff(wavelength) < 0):
                raise ValueError('The wavelengths are not all sorted by increasing value.')

            # [erg s-1 cm-2 Hz-1] -> [W m-2 micron-1]
            flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      np.asarray(co_ratio),
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('petitcode-hot-clear', ('teff', 'logg', 'feh', 'co'), database, data_sorted)

    print_message = 'Adding petitCODE hot clear model spectra... [DONE]'
    print(f'\r{print_message:<110}')


def add_petitcode_hot_cloudy(input_path,
                             database,
                             wl_bound,
                             teff_bound,
                             specres,
                             data_folder):
    """
    Function for adding the petitCODE hot cloudy atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wl_bound : tuple(float, float)
        Wavelength range (micron).
    teff_bound : tuple(float, float), None
        Effective temperature range (K).
    specres : float
        Spectral resolution.
    data_folder : str
        Path with input data.

    Returns
    -------
    NoneType
        None
    """

    if not wl_bound:
        wl_bound = (1e-2, 1e2)

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    teff = []
    logg = []
    feh = []
    co_ratio = []
    fsed = []
    wavelength = None
    flux = []

    for _, _, files in os.walk(data_folder):
        for filename in files:
            print_message = f'Adding petitCODE hot cloudy model spectra... {filename}'
            print(f'\r{print_message:<110}', end='')

            teff.append(float(filename[9:13]))
            logg.append(float(filename[19:23]))
            feh.append(float(filename[28:32]))
            co_ratio.append(float(filename[36:40]))
            fsed.append(float(filename[46:50]))

            data = np.loadtxt(os.path.join(data_folder, filename))

            if wavelength is None:
                # [cm] -> [micron]
                wavelength = data[:, 0]*1e4

            if np.all(np.diff(wavelength) < 0):
                raise ValueError('The wavelengths are not all sorted by increasing value.')

            # [erg s-1 cm-2 Hz-1] -> [W m-2 micron-1]
            flux.append(data[:, 1]*1e-9*constants.LIGHT/(wavelength*1e-6)**2)

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      np.asarray(co_ratio),
                                      np.asarray(fsed),
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('petitcode-hot-cloudy', ('teff', 'logg', 'feh', 'co', 'fsed'), database, data_sorted)

    print_message = 'Adding petitCODE hot cloudy model spectra... [DONE]'
    print(f'\r{print_message:<110}')
