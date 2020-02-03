"""
Module for BT-NextGen atmospheric model spectra.
"""

import os
import tarfile
import urllib.request

import spectres
import numpy as np
import pandas as pd

from species.util import data_util


def add_btnextgen(input_path,
                  database,
                  wavel_range,
                  teff_range,
                  spec_res):
    """
    Function for adding the BT-NextGen atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wavel_range : tuple(float, float)
        Wavelength range (micron).
    teff_range : tuple(float, float), None
        Effective temperature range (K).
    spec_res : float
        Spectral resolution.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    data_folder = os.path.join(input_path, 'bt-nextgen/')

    files = ['BT-NextGen_M-0.0_a+0.0_hot.tar',
             'BT-NextGen_M+0.3_a+0.0_hot.tar',
             'BT-NextGen_M+0.5_a+0.0_hot.tar']

    urls = ['https://phoenix.ens-lyon.fr/Grids/BT-NextGen/SPECTRA/BT-NextGen_M-0.0_a+0.0_hot.tar',
            'https://phoenix.ens-lyon.fr/Grids/BT-NextGen/SPECTRA/BT-NextGen_M+0.3_a+0.0_hot.tar',
            'https://phoenix.ens-lyon.fr/Grids/BT-NextGen/SPECTRA/BT-NextGen_M+0.5_a+0.0_hot.tar']

    labels = ['[Fe/H]=0.0 (5.9 GB)',
              '[Fe/H]=0.3 (6.2 GB)',
              '[Fe/H]=0.5 (6.4 GB)']

    for i, item in enumerate(files):
        data_file = os.path.join(input_path, item)

        if not os.path.isfile(data_file):
            print(f'Downloading BT-NextGen model spectra {labels[i]}...', end='', flush=True)
            urllib.request.urlretrieve(urls[i], data_file)
            print(' [DONE]')

        print(f'Unpacking BT-NextGen model spectra {labels[i]}...', end='', flush=True)
        tar = tarfile.open(data_file)
        tar.extractall(data_folder)
        tar.close()
        print(' [DONE]')

    teff = []
    logg = []
    feh = []
    flux = []

    wavelength = [wavel_range[0]]

    while wavelength[-1] <= wavel_range[1]:
        wavelength.append(wavelength[-1] + wavelength[-1]/spec_res)

    wavelength = np.asarray(wavelength[:-1])

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte') and filename.endswith('.7.bz2'):
                teff_val = float(filename[3:6])*100.
                logg_val = float(filename[7:9])
                feh_val = float(filename[11:14])

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                print_message = f'Adding BT-NextGen model spectra... {filename}'
                print(f'\r{print_message:<80}', end='')

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

                data = np.stack((data_wavel, data_flux), axis=0)

                index_sort = np.argsort(data[0, :])
                data = data[:, index_sort]

                if np.all(np.diff(data[0, :]) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                teff.append(teff_val)
                logg.append(logg_val)
                feh.append(feh_val)

                try:
                    flux.append(spectres.spectres(wavelength, data[:, 0], data[:, 1]))
                except ValueError:
                    flux.append(np.zeros(wavelength.shape[0]))

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      None,
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('bt-nextgen', ('teff', 'logg', 'feh'), database, data_sorted)

    print_message = 'Adding BT-NextGen model spectra... [DONE]'
    print(f'\r{print_message:<80}')
