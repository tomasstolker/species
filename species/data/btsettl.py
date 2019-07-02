"""
Module for BT-Settl atmospheric models.
"""

import os
import sys
import tarfile

from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from species.util import data_util


def add_btsettl(input_path,
                database,
                wl_bound,
                teff_bound,
                specres):
    """
    Function for adding the BT-Settl atmospheric models to the database.

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

    data_folder = os.path.join(input_path, 'bt-settl/')

    input_file = 'BT-Settl_M-0.0_a+0.0.tar'
    label = '(5.8 GB)'

    url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011/SPECTRA/BT-Settl_M-0.0_a+0.0.tar'

    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        sys.stdout.write(f'Downloading BT-Settl model spectra {label}...')
        sys.stdout.flush()

        urlretrieve(url, data_file)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    sys.stdout.write(f'Unpacking BT-Settl model spectra {label}...')
    sys.stdout.flush()

    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    teff = []
    logg = []
    flux = []

    wavelength = [wl_bound[0]]

    while wavelength[-1] <= wl_bound[1]:
        wavelength.append(wavelength[-1] + wavelength[-1]/specres)

    wavelength = np.asarray(wavelength[:-1])

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte') and filename.endswith('.7.bz2'):
                sys.stdout.write('\rAdding BT-Settl model spectra... '+filename+'  ')
                sys.stdout.flush()

                if len(filename) == 39:
                    teff_val = float(filename[3:6])*100.
                    logg_val = float(filename[7:10])
                    feh_val = float(filename[11:14])

                elif len(filename) == 41:
                    teff_val = float(filename[3:8])*100.
                    logg_val = float(filename[9:12])
                    feh_val = float(filename[13:16])

                if teff_bound is not None:
                    if teff_val < teff_bound[0] or teff_val > teff_bound[1]:
                        continue

                if feh_val != 0.:
                    continue

                dataf = pd.pandas.read_csv(data_folder+filename,
                                           usecols=[0, 1],
                                           names=['wavelength', 'flux'],
                                           header=None,
                                           dtype={'wavelength': str, 'flux': str},
                                           delim_whitespace=True,
                                           compression='bz2')

                dataf['wavelength'] = dataf['wavelength'].str.replace('D', 'E')
                dataf['flux'] = dataf['flux'].str.replace('D', 'E')

                dataf = dataf.apply(pd.to_numeric)
                data = dataf.values

                # [Angstrom] -> [micron]
                data_wavel = data[:, 0]*1e-4

                # See https://phoenix.ens-lyon.fr/Grids/FORMAT
                data_flux = 10.**(data[:, 1]-8.)  # [erg s-1 cm-2 Angstrom-1]

                # [erg s-1 cm-2 Angstrom-1] -> [W m-2 micron-1]
                data_flux = data_flux*1e-7*1e4*1e4

                data = np.stack((data_wavel, data_flux), axis=1)

                index_sort = np.argsort(data[:, 0])
                data = data[index_sort, :]

                if np.all(np.diff(data[:, 0]) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                indices = np.where((data[:, 0] >= wl_bound[0]) &
                                   (data[:, 0] <= wl_bound[1]))[0]

                if indices.size > 0:
                    teff.append(teff_val)
                    logg.append(logg_val)

                    data = data[indices, :]

                    flux_interp = interp1d(data[:, 0],
                                           data[:, 1],
                                           kind='linear',
                                           bounds_error=False,
                                           fill_value=1e-100)

                    flux.append(flux_interp(wavelength))

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('bt-settl', ('teff', 'logg'), database, data_sorted)

    sys.stdout.write('\rAdding BT-Settl model spectra... [DONE]'
                     '                                  \n')

    sys.stdout.flush()
