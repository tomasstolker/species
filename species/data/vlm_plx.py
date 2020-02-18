"""
Text
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
    database : h5py._hl.files.File

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

    group = 'photometry/vlm-plx'

    database.create_group(group)

    with fits.open(data_file) as hdulist:
        header = hdulist[1].header
        photdata = hdulist[1].data

    parallax = photdata['PLX']  # [mas]
    parallax_error = photdata['EPLX']  # [mas]
    distance = 1./(parallax*1e-3)  # [pc]

    distance_minus = distance - 1./((parallax+parallax_error)*1e-3)  # [pc]
    distance_plus = 1./((parallax-parallax_error)*1e-3) - distance  # [pc]
    distance_error = (distance_plus+distance_minus)/2.  # [pc]

    name = photdata['NAME']
    name = np.core.defchararray.strip(name)

    sptype = photdata['ISPTSTR']
    sptype = np.core.defchararray.strip(sptype)

    sptype_op = photdata['OSPTSTR']
    sptype_op = np.core.defchararray.strip(sptype_op)

    for i, item in enumerate(sptype):
        if item == 'null':
            sptype[i] = sptype_op[i]

    flag = photdata['FLAG']
    flag = np.core.defchararray.strip(flag)

    sptype = data_util.update_sptype(sptype)

    dtype = h5py.special_dtype(vlen=str)

    dset = database.create_dataset(group+'/name', (np.size(name), ), dtype=dtype)
    dset[...] = name

    dset = database.create_dataset(group+'/sptype', (np.size(sptype), ), dtype=dtype)
    dset[...] = sptype

    dset = database.create_dataset(group+'/flag', (np.size(flag), ), dtype=dtype)
    dset[...] = flag

    database.create_dataset(group+'/ra', data=photdata['RA'])  # [deg]
    database.create_dataset(group+'/dec', data=photdata['DEC'])  # [deg]
    database.create_dataset(group+'/distance', data=distance)
    database.create_dataset(group+'/distance_error', data=distance_error)
    database.create_dataset(group+'/MKO/NSFCam.Y', data=photdata['YMAG'])
    database.create_dataset(group+'/MKO/NSFCam.J', data=photdata['JMAG'])
    database.create_dataset(group+'/MKO/NSFCam.H', data=photdata['HMAG'])
    database.create_dataset(group+'/MKO/NSFCam.K', data=photdata['KMAG'])
    database.create_dataset(group+'/MKO/NSFCam.Lp', data=photdata['LMAG'])
    database.create_dataset(group+'/MKO/NSFCam.Mp', data=photdata['MMAG'])
    database.create_dataset(group+'/2MASS/2MASS.J', data=photdata['J2MAG'])
    database.create_dataset(group+'/2MASS/2MASS.H', data=photdata['H2MAG'])
    database.create_dataset(group+'/2MASS/2MASS.Ks', data=photdata['K2MAG'])

    print(' [DONE]')

    database.close()
