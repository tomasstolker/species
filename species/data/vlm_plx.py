"""
Module for the photometric data and parallaxes from the Database of Ultracool Parallaxes.
"""

import os
import urllib.request

import h5py
import numpy as np

from astropy.io import fits

from species.util import data_util


def add_vlm_plx(input_path,
                database):
    """
    Function for adding the Database of Ultracool Parallaxes to the database.

    Parameters
    ----------
    input_path : str
        Data folder.
    database : h5py._hl.files.File
        HDF5 database.

    Returns
    -------
    NoneType
        None
    """

    data_file = os.path.join(input_path, 'vlm-plx-all.fits')

    url = 'http://www.as.utexas.edu/~tdupuy/plx/' \
          'Database_of_Ultracool_Parallaxes_files/vlm-plx-all.fits'

    if not os.path.isfile(data_file):
        print('Downloading Database of Ultracool Parallaxes (307 kB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Adding Database of Ultracool Parallaxes...', end='', flush=True)

    database.create_group('photometry/vlm-plx')

    with fits.open(data_file) as hdu_list:
        phot_data = hdu_list[1].data

    parallax = phot_data['PLX']  # (mas)
    parallax_error = phot_data['EPLX']  # (mas)
    distance = 1./(parallax*1e-3)  # (pc)

    distance_minus = distance - 1./((parallax+parallax_error)*1e-3)  # (pc)
    distance_plus = 1./((parallax-parallax_error)*1e-3) - distance  # (pc)
    distance_error = (distance_plus+distance_minus)/2.  # (pc)

    name = phot_data['NAME']
    name = np.core.defchararray.strip(name)

    sptype = phot_data['OSPTSTR']
    sptype = np.core.defchararray.strip(sptype)

    sptype_nir = phot_data['ISPTSTR']
    sptype_nir = np.core.defchararray.strip(sptype_nir)

    for i, item in enumerate(sptype):
        if item == 'null':
            sptype[i] = sptype_nir[i]

    flag = phot_data['FLAG']
    flag = np.core.defchararray.strip(flag)

    sptype = data_util.update_sptype(sptype)

    dtype = h5py.special_dtype(vlen=str)

    dset = database.create_dataset('photometry/vlm-plx/name', (np.size(name), ), dtype=dtype)
    dset[...] = name

    dset = database.create_dataset('photometry/vlm-plx/sptype', (np.size(sptype), ), dtype=dtype)
    dset[...] = sptype

    dset = database.create_dataset('photometry/vlm-plx/flag', (np.size(flag), ), dtype=dtype)
    dset[...] = flag

    database.create_dataset('photometry/vlm-plx/ra', data=phot_data['RA'])  # (deg)
    database.create_dataset('photometry/vlm-plx/dec', data=phot_data['DEC'])  # (deg)
    database.create_dataset('photometry/vlm-plx/parallax', data=parallax)
    database.create_dataset('photometry/vlm-plx/parallax_error', data=parallax_error)
    database.create_dataset('photometry/vlm-plx/distance', data=distance)
    database.create_dataset('photometry/vlm-plx/distance_error', data=distance_error)
    database.create_dataset('photometry/vlm-plx/MKO/NSFCam.Y', data=phot_data['YMAG'])
    database.create_dataset('photometry/vlm-plx/MKO/NSFCam.J', data=phot_data['JMAG'])
    database.create_dataset('photometry/vlm-plx/MKO/NSFCam.H', data=phot_data['HMAG'])
    database.create_dataset('photometry/vlm-plx/MKO/NSFCam.K', data=phot_data['KMAG'])
    database.create_dataset('photometry/vlm-plx/MKO/NSFCam.Lp', data=phot_data['LMAG'])
    database.create_dataset('photometry/vlm-plx/MKO/NSFCam.Mp', data=phot_data['MMAG'])
    database.create_dataset('photometry/vlm-plx/2MASS/2MASS.J', data=phot_data['J2MAG'])
    database.create_dataset('photometry/vlm-plx/2MASS/2MASS.H', data=phot_data['H2MAG'])
    database.create_dataset('photometry/vlm-plx/2MASS/2MASS.Ks', data=phot_data['K2MAG'])

    print(' [DONE]')

    database.close()
