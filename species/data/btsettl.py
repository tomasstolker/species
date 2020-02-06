"""
Module for BT-Settl atmospheric model spectra.
"""

import os
import lzma
import tarfile
import urllib.request

import spectres
import numpy as np

from species.util import data_util


def add_btsettl(input_path,
                database,
                wavel_range,
                teff_range,
                spec_res):
    """
    Function for adding the BT-Settl atmospheric models (solar metallicity) to the database.
    The spectra are read line-by-line because the wavelength and flux are not separated in the
    input data.

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

    data_folder = os.path.join(input_path, 'bt-settl/')

    input_file = 'SPECTRA.tar'

    url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011c/SPECTRA.tar'

    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print('Downloading BT-Settl model spectra (8.1 GB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Unpacking BT-Settl model spectra (8.1 GB)...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    data_folder = os.path.join(data_folder, 'SPECTRA')

    teff = []
    logg = []
    flux = []

    wavelength = [wavel_range[0]]

    while wavelength[-1] <= wavel_range[1]:
        wavelength.append(wavelength[-1] + wavelength[-1]/spec_res)

    wavelength = np.asarray(wavelength[:-1])

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte') and filename.endswith('.7.xz'):
                if len(filename) == 38:
                    teff_val = float(filename[3:6])*100.
                    logg_val = float(filename[7:10])
                    feh_val = float(filename[11:14])

                elif len(filename) == 40:
                    teff_val = float(filename[3:8])*100.
                    logg_val = float(filename[9:12])
                    feh_val = float(filename[13:16])

                else:
                    raise ValueError('The length of the filename is not compatible for reading '
                                     'the parameter values.')

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                if feh_val != 0.:
                    continue

                print_message = f'Adding BT-Settl model spectra... {filename}'
                print(f'\r{print_message:<80}', end='')

                data_wavel = []
                data_flux = []

                with lzma.open(os.path.join(data_folder, filename), mode='rt') as xz_file:
                    for line in xz_file:
                        line = line[:line.find('D')+4].split()

                        if len(line) == 1:
                            line = line[0]
                            wavel_tmp = line[:line.find('-')]
                            flux_tmp = line[line.find('-'):]

                        elif len(line) == 2:
                            wavel_tmp = line[0]
                            flux_tmp = line[1]

                        # [Angstrom] -> [micron]
                        data_wavel.append(float(wavel_tmp)*1e-4)

                        # See https://phoenix.ens-lyon.fr/Grids/FORMAT
                        flux_cgs = 10.**(float(flux_tmp.replace('D', 'E'))-8.)

                        # [erg s-1 cm-2 Angstrom-1] -> [W m-2 micron-1]
                        data_flux.append(flux_cgs*1e-7*1e4*1e4)

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

    data_util.write_data('bt-settl', ['teff', 'logg'], database, data_sorted)

    print_message = 'Adding BT-Settl model spectra... [DONE]'
    print(f'\r{print_message:<80}')
