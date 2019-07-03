"""
Module for AMES-Dusty atmospheric models.
"""

import os
import sys
import gzip
import tarfile

from urllib.request import urlretrieve

import numpy as np

from scipy.interpolate import interp1d

from species.util import data_util


def add_ames_dusty(input_path,
                   database,
                   wl_bound,
                   teff_bound,
                   specres):
    """
    Function for adding the AMES-Dusty atmospheric models to the database.

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

    data_folder = os.path.join(input_path, 'ames-dusty/')

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    input_file = 'SPECTRA.tar'
    label = '[Fe/H]=0.0 (106 MB)'

    url = 'https://phoenix.ens-lyon.fr/Grids/AMES-Dusty/SPECTRA.tar'

    data_file = os.path.join(data_folder, input_file)

    if not os.path.isfile(data_file):
        sys.stdout.write(f'Downloading AMES-Dusty model spectra {label}...')
        sys.stdout.flush()

        urlretrieve(url, data_file)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    sys.stdout.write(f'Unpacking AMES-Dusty model spectra {label}...')
    sys.stdout.flush()

    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    data_folder += 'SPECTRA/'

    teff = []
    logg = []
    flux = []

    wavelength = [wl_bound[0]]

    while wavelength[-1] <= wl_bound[1]:
        wavelength.append(wavelength[-1] + wavelength[-1]/specres)

    wavelength = np.asarray(wavelength[:-1])

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte') and filename.endswith('.7.gz'):
                sys.stdout.write('\rAdding AMES-Dusty model spectra... '+filename+'   ')
                sys.stdout.flush()

                teff_val = float(filename[3:5])*100.
                logg_val = float(filename[6:9])
                feh_val = float(filename[10:13])

                if feh_val != 0.:
                    continue

                if teff_bound is not None:
                    if teff_val < teff_bound[0] or teff_val > teff_bound[1]:
                        continue

                data_wavel = []
                data_flux = []

                with gzip.open(data_folder+filename, 'rt') as gz_file:
                    for line in gz_file:
                        line_split = line.split()

                        if len(line_split) > 1:
                            tmp_wavel = line_split[0].strip()
                            tmp_flux = line_split[1].strip()

                            if len(tmp_wavel) == 21 and tmp_wavel[-4] == 'D' \
                                    and tmp_flux[-4] == 'D':
                                data_wavel.append(float(line[1:23].replace('D', 'E')))
                                data_flux.append(float(line[25:35].replace('D', 'E')))

                # See https://phoenix.ens-lyon.fr/Grids/FORMAT
                data_wavel = np.asarray(data_wavel)*1e-4  # [Angstrom] -> [micron]
                data_flux = 10.**(np.asarray(data_flux)-8.)  # [erg s-1 cm-2 Angstrom-1]

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

    data_util.write_data('ames-dusty', ('teff', 'logg'), database, data_sorted)

    sys.stdout.write('\rAdding AMES-Dusty model spectra... [DONE]'
                     '                                  \n')

    sys.stdout.flush()
