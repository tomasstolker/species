"""
Module for DRIFT-PHOENIX atmospheric model spectra.
"""

import os
import tarfile
import urllib.request

import numpy as np

from species.util import data_util


def add_drift_phoenix(input_path,
                      database):
    """
    Function for adding the DRIFT-PHOENIX atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    data_file = os.path.join(input_path, 'drift-phoenix.tgz')
    data_folder = os.path.join(input_path, 'drift-phoenix/')

    url = 'https://people.phys.ethz.ch/~stolkert/species/drift-phoenix.tgz'

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
    wavelength = None
    flux = []

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte_'):
                print_message = f'Adding DRIFT-PHOENIX model spectra... {filename}'
                print(f'\r{print_message:<65}', end='')

                teff.append(float(filename[4:8]))
                logg.append(float(filename[9:12]))
                feh.append(float(filename[12:16]))

                data = np.loadtxt(data_folder+filename)

                if wavelength is None:
                    # [Angstrom] -> [micron]
                    wavelength = data[:, 0]*1e-4

                if np.all(np.diff(wavelength) < 0):
                    raise ValueError('The wavelengths are not all sorted by increasing value.')

                # [erg s-1 cm-2 Angstrom-1] -> [W m-2 micron-1]
                flux.append(data[:, 1]*1e-7*1e4*1e4)

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      np.asarray(feh),
                                      None,
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('drift-phoenix', ('teff', 'logg', 'feh'), database, data_sorted)

    print_message = 'Adding DRIFT-PHOENIX model spectra... [DONE]'
    print(f'\r{print_message:<65}')
