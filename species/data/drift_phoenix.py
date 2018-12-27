"""
Module for DRIFT-PHOENIX atmospheric models.
"""

import os
import sys
import tarfile
import urllib.request

import numpy as np


def add_drift_phoenix(input_path,
                      database):
    """
    Function for adding the DRIFT-PHOENIX atmospheric models to the database.

    :param input_path:
    :type input_path: str
    :param database:
    :type database: h5py._hl.files.File

    :return: None
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    data_file = os.path.join(input_path, 'drift-phoenix.tgz')
    data_folder = os.path.join(input_path, 'drift-phoenix/')

    url = 'https://people.phys.ethz.ch/~stolkert/species/drift-phoenix.tgz'

    if not os.path.isfile(data_file):
        sys.stdout.write('Downloading DRIFT-PHOENIX model spectra (151 MB)...')
        sys.stdout.flush()

        urllib.request.urlretrieve(url, data_file)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    sys.stdout.write('Unpacking DRIFT-PHOENIX model spectra...')
    sys.stdout.flush()

    tar = tarfile.open(data_file)
    tar.extractall(input_path)
    tar.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    files = []
    teff = []
    logg = []
    feh = []
    wavelength = None
    flux = []

    for root, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte_'):
                files.append(filename)

                sys.stdout.write('\rAdding DRIFT-PHOENIX model spectra... '+filename)
                sys.stdout.flush()

                teff.append(float(filename[4:8]))
                logg.append(float(filename[9:12]))
                feh.append(float(filename[12:16]))

                data = np.loadtxt(root+filename)

                if wavelength is None:
                    # [Angstrom] -> [micron]
                    wavelength = data[:, 0]*1e-4

                # [erg s-1 cm-2 Angstrom-1] -> [W m-2 micron-1]
                flux.append(data[:, 1]*1e-7*1e4*1e4)

    data_sorted = sort_data(np.asarray(teff),
                            np.asarray(logg),
                            np.asarray(feh),
                            wavelength,
                            np.asarray(flux))

    write_data(database, data_sorted)

    sys.stdout.write('\rAdding DRIFT-PHOENIX model spectra... [DONE]                    \n')
    sys.stdout.flush()


def sort_data(teff,
              logg,
              feh,
              wavelength,
              flux):
    """
    :param teff:
    :type teff: numpy.ndarray
    :param logg:
    :type logg: numpy.ndarray
    :param feh:
    :type feh: numpy.ndarray
    :param wavelength:
    :type wavelength: numpy.ndarray
    :param flux:
    :type flux: numpy.ndarray

    :return:
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    teff_unique = np.unique(teff)
    logg_unique = np.unique(logg)
    feh_unique = np.unique(feh)

    spectrum = np.zeros((teff_unique.shape[0],
                         logg_unique.shape[0],
                         feh_unique.shape[0],
                         wavelength.shape[0]))

    for i in range(teff.shape[0]):
        index_teff = np.argwhere(teff_unique == teff[i])[0]
        index_logg = np.argwhere(logg_unique == logg[i])[0]
        index_feh = np.argwhere(feh_unique == feh[i])[0]

        spectrum[index_teff, index_logg, index_feh, :] = flux[i]

    return (teff_unique, logg_unique, feh_unique, wavelength, spectrum)


def write_data(database,
               data_sorted):
    """
    :param database:
    :type database: h5py._hl.files.File

    :return: None
    """

    if 'models/drift-phoenix' in database:
        del database['models/drift-phoenix']

    database.create_group('models/drift-phoenix')

    database.create_dataset('models/drift-phoenix/teff',
                            data=data_sorted[0],
                            dtype='f')

    database.create_dataset('models/drift-phoenix/logg',
                            data=data_sorted[1],
                            dtype='f')

    database.create_dataset('models/drift-phoenix/feh',
                            data=data_sorted[2],
                            dtype='f')

    database.create_dataset('models/drift-phoenix/wavelength',
                            data=data_sorted[3],
                            dtype='f')

    database.create_dataset('models/drift-phoenix/flux',
                            data=data_sorted[4],
                            dtype='f')


def add_missing(database):
    """
    :param database:
    :type database: h5py._hl.files.File

    :return: None
    """

    teff = np.asarray(database['models/drift-phoenix/teff'])
    logg = np.asarray(database['models/drift-phoenix/logg'])
    feh = np.asarray(database['models/drift-phoenix/feh'])
    flux = np.asarray(database['models/drift-phoenix/flux'])

    for i in range(teff.shape[0]):
        for j in range(logg.shape[0]):
            for k in range(feh.shape[0]):
                if np.count_nonzero(flux[i, j, k]) == 0:
                    scaling = (teff[i+1]-teff[i])/(teff[i+1]-teff[i-1])
                    flux[i, j, k] = scaling*flux[i+1, j, k] + (1.-scaling)*flux[i-1, j, k]

    del database['models/drift-phoenix/flux']

    database.create_dataset('models/drift-phoenix/flux',
                            data=flux,
                            dtype='f')
