"""
Module for DRIFT-PHOENIX atmospheric models.
"""

import os
import sys
import tarfile

from urllib.request import urlretrieve

import numpy as np

from species.util import data_util


def add_drift_phoenix(input_path,
                      database):
    """
    Function for adding the DRIFT-PHOENIX atmospheric models to the database.

    Parameters
    ----------
    input_path : str
    database : h5py._hl.files.File

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
        sys.stdout.write('Downloading DRIFT-PHOENIX model spectra (151 MB)...')
        sys.stdout.flush()

        urlretrieve(url, data_file)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    sys.stdout.write('Unpacking DRIFT-PHOENIX model spectra...')
    sys.stdout.flush()

    tar = tarfile.open(data_file)
    tar.extractall(input_path)
    tar.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    teff = []
    logg = []
    feh = []
    wavelength = None
    flux = []

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte_'):
                sys.stdout.write('\rAdding DRIFT-PHOENIX model spectra... '+filename)
                sys.stdout.flush()

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
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('drift-phoenix', ('teff', 'logg', 'feh'), database, data_sorted)

    sys.stdout.write('\rAdding DRIFT-PHOENIX model spectra... [DONE]                    \n')
    sys.stdout.flush()
