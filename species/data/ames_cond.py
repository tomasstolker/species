"""
Module for AMES-Cond atmospheric model spectra.
"""

import os
import gzip
import tarfile
import urllib.request

import spectres
import numpy as np

from species.util import data_util


def add_ames_cond(input_path,
                  database,
                  wavel_range,
                  teff_range,
                  spec_res):
    """
    Function for adding the AMES-Cond atmospheric models to the database.

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

    data_folder = os.path.join(input_path, 'ames-cond/')

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    input_file = 'SPECTRA.tar'
    label = '(823 MB)'

    url = 'https://phoenix.ens-lyon.fr/Grids/AMES-Cond/SPECTRA.tar'

    data_file = os.path.join(data_folder, input_file)

    if not os.path.isfile(data_file):
        print(f'Downloading AMES-Cond model spectra {label}...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print(f'Unpacking AMES-Cond model spectra {label}...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    data_folder += 'SPECTRA/'

    teff = []
    logg = []
    flux = []

    wavelength = [wavel_range[0]]

    while wavelength[-1] <= wavel_range[1]:
        wavelength.append(wavelength[-1] + wavelength[-1]/spec_res)

    wavelength = np.asarray(wavelength[:-1])

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte') and filename.endswith('.gz'):
                teff_val = float(filename[3:5])*100.
                logg_val = float(filename[6:9])
                feh_val = float(filename[10:13])

                if feh_val != 0.:
                    continue

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                print_message = f'Adding AMES-Cond model spectra... {filename}'
                print(f'\r{print_message:<71}', end='')

                data_wavel = []
                data_flux = []

                with gzip.open(data_folder+filename, 'rt') as gz_file:
                    if filename.endswith('.7.gz'):

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
                        data_flux = 10.**(np.asarray(data_flux)-8.)  # [erg s-1 cm-2 Angstrom-1]

                    elif filename.endswith('.spec.gz'):

                        read_wavel = True
                        read_end = False

                        for i, line in enumerate(gz_file):
                            if read_end:
                                break

                            if i == 1:
                                wl_points = int(line.split()[0])

                            elif i > 1:
                                for item in line.split():
                                    if read_wavel:
                                        data_wavel.append(float(item))

                                        if len(data_wavel) == wl_points:
                                            read_wavel = False
                                            break

                                    else:
                                        data_flux.append(float(item))

                                        if len(data_flux) == wl_points:
                                            read_end = True
                                            break

                        # See https://phoenix.ens-lyon.fr/Grids/FORMAT
                        data_flux = np.asarray(data_flux)*10.**-8.  # [erg s-1 cm-2 Angstrom-1]

                # [Angstrom] -> [micron]
                data_wavel = np.asarray(data_wavel)*1e-4

                # [erg s-1 cm-2 Angstrom-1] -> [W m-2 micron-1]
                data_flux = data_flux*1e-7*1e4*1e4

                data = np.stack((data_wavel, data_flux), axis=1)

                index_sort = np.argsort(data[:, 0])
                data = data[index_sort, :]

                if np.all(np.diff(data[:, 0]) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                teff.append(teff_val)
                logg.append(logg_val)

                try:
                    flux.append(spectres.spectres(wavelength, data[:, 0], data[:, 1]))
                except ValueError:
                    flux.append(np.zeros(wavelength.shape[0]))

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      None,
                                      None,
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('ames-cond', ['teff', 'logg'], database, data_sorted)

    print_message = 'Adding AMES-Cond model spectra... [DONE]'
    print(f'\r{print_message:<71}')
