"""
Module for BT-NextGen atmospheric models.
"""

import os
import sys
import tarfile

from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from species.util import data_util


def add_btnextgen(input_path,
                  database,
                  wl_bound,
                  teff_bound,
                  specres):
    """
    Function for adding the BT-NextGen atmospheric models to the database.

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
            sys.stdout.write('Downloading BT-NextGen model spectra '+labels[i]+'...')
            sys.stdout.flush()

            urlretrieve(urls[i], data_file)

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

        sys.stdout.write('Unpacking BT-NextGen model spectra '+labels[i]+'...')
        sys.stdout.flush()

        tar = tarfile.open(data_file)
        tar.extractall(data_folder)
        tar.close()

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    teff = []
    logg = []
    feh = []
    flux = []

    wavelength = [wl_bound[0]]

    while wavelength[-1] <= wl_bound[1]:
        wavelength.append(wavelength[-1] + wavelength[-1]/specres)

    wavelength = np.asarray(wavelength[:-1])

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte') and filename.endswith('.7.bz2'):
                sys.stdout.write('\rAdding BT-NextGen model spectra... '+filename)
                sys.stdout.flush()

                teff_val = float(filename[3:6])*100.

                if teff_bound is not None:

                    if teff_bound[0] <= teff_val <= teff_bound[1]:
                        teff.append(teff_val)
                        logg.append(float(filename[7:9]))
                        feh.append(float(filename[11:14]))

                    else:
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

                data = np.stack((data_wavel, data_flux), axis=0)

                index_sort = np.argsort(data[0, :])
                data = data[:, index_sort]

                if np.all(np.diff(data[0, :]) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                indices = np.where((data[0, :] >= wl_bound[0]) &
                                   (data[0, :] <= wl_bound[1]))[0]

                data = data[:, indices]

                flux_interp = interp1d(data[0, :],
                                       data[1, :],
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value=1e-100)

                flux.append(flux_interp(wavelength))

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('bt-nextgen', ('teff', 'logg', 'feh'), database, data_sorted)

    sys.stdout.write('\rAdding BT-NextGen model spectra... [DONE]'
                     '                              \n')

    sys.stdout.flush()
