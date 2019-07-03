"""
Module for isochrone data of evolutionary models.
"""

import sys

import h5py
import numpy as np

from species.core import constants


def add_baraffe(database,
                tag,
                filename):
    """
    Function for adding the Baraffe et al. isochrone data to the database. Any of the isochrones
    from  https://phoenix.ens-lyon.fr/Grids/ can be used as input.

    https://ui.adsabs.harvard.edu/abs/2003A%26A...402..701B/

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    filename : str
        Filename with the isochrones data.

    Returns
    -------
    NoneType
        None
    """

    # read in all the data, ignoring empty lines or lines with '---'
    data = []
    with open(filename) as data_file:
        for line in data_file:
            if '---' in line or line == '\n':
                continue
            else:
                data.append(list(filter(None, line.rstrip().split(' '))))

    isochrones = []

    for line in data:
        if '(Gyr)' in line:
            age = line[-1]

        elif 'lg(g)' in line:
            header = ['M/Ms', 'Teff(K)'] + line[1:]

        else:
            line.insert(0, age)
            isochrones.append(line)

    header = np.asarray(header, dtype=bytes)
    isochrones = np.asarray(isochrones, dtype=float)

    isochrones[:, 0] *= 1e3  # [Myr]
    isochrones[:, 1] *= constants.M_SUN/constants.M_JUP  # [Mjup]

    index_sort = np.argsort(isochrones[:, 0])
    isochrones = isochrones[index_sort, :]

    sys.stdout.write('Adding isochrones: '+tag+'...')
    sys.stdout.flush()

    bytes_type = h5py.special_dtype(vlen=bytes)

    database.create_dataset('isochrones/'+tag+'/filters',
                            data=header[7:],
                            dtype=bytes_type)

    database.create_dataset('isochrones/'+tag+'/magnitudes',
                            data=isochrones[:, 8:],
                            dtype='f')

    dset = database.create_dataset('isochrones/'+tag+'/evolution',
                                   data=isochrones[:, 0:8],
                                   dtype='f')

    dset.attrs['model'] = 'baraffe'

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()


def add_marleau(database,
                tag,
                filename):
    """
    Function for adding the Marleau et al. isochrone data to the database. The isochrone data can
    be requested from Gabriel Marleau.

    https://ui.adsabs.harvard.edu/abs/2019A%26A...624A..20M/abstract

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    filename : str
        Filename with the isochrones data.

    Returns
    -------
    NoneType
        None
    """

    # M      age     S_0             L          S(t)            R        Teff
    # (M_J)  (Gyr)   (k_B/baryon)    (L_sol)    (k_B/baryon)    (R_J)    (K)
    mass, age, _, luminosity, _, radius, teff = np.loadtxt(filename, unpack=True)

    age *= 1e3  # [Myr]
    luminosity = np.log10(luminosity)

    mass_cgs = 1e3 * mass * constants.M_JUP  # [g]
    radius_cgs = 1e2 * radius * constants.R_JUP  # [cm]

    logg = np.log10(1e3 * constants.GRAVITY * mass_cgs / radius_cgs**2)

    sys.stdout.write('Adding isochrones: '+tag+'...')
    sys.stdout.flush()

    isochrones = np.vstack((age, mass, teff, luminosity, logg))
    isochrones = np.transpose(isochrones)

    index_sort = np.argsort(isochrones[:, 0])
    isochrones = isochrones[index_sort, :]

    dset = database.create_dataset('isochrones/'+tag+'/evolution',
                                   data=isochrones,
                                   dtype='f')

    dset.attrs['model'] = 'marleau'

    sys.stdout.write(' [DONE]\n')
