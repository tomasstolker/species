"""
Module for optical constants of dust grains.
"""

import os
import zipfile
import urllib.request

import h5py
import numpy as np

from astropy.io import fits
from typeguard import typechecked


@typechecked
def add_optical_constants(input_path: str,
                          database: h5py._hl.files.File) -> None:
    """
    Function for adding the optical constants of crystalline and amorphous MgSiO3 and Fe to the
    database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    None
        NoneType
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url = 'https://home.strw.leidenuniv.nl/~stolker/species/optical_constants.zip'

    data_file = os.path.join(input_path, 'optical_constants.zip')

    if not os.path.isfile(data_file):
        print('Downloading optical constants (87 kB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Unpacking optical constants...', end='', flush=True)

    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(input_path)

    print(' [DONE]')

    print('Adding optical constants of MgSiO3...', end='')

    nk_file = os.path.join(input_path, 'optical_constants/mgsio3/crystalline/'
                           'mgsio3_jaeger_98_scott_96_axis1.dat')

    data = np.loadtxt(nk_file)
    database.create_dataset('dust/mgsio3/crystalline/axis_1/', data=data)

    nk_file = os.path.join(input_path, 'optical_constants/mgsio3/crystalline/'
                           'mgsio3_jaeger_98_scott_96_axis2.dat')

    data = np.loadtxt(nk_file)
    database.create_dataset('dust/mgsio3/crystalline/axis_2/', data=data)

    nk_file = os.path.join(input_path, 'optical_constants/mgsio3/crystalline/'
                           'mgsio3_jaeger_98_scott_96_axis3.dat')

    data = np.loadtxt(nk_file)
    database.create_dataset('dust/mgsio3/crystalline/axis_3/', data=data)

    nk_file = os.path.join(input_path, 'optical_constants/mgsio3/amorphous/'
                           'mgsio3_jaeger_2003_reformat.dat')

    data = np.loadtxt(nk_file)
    database.create_dataset('dust/mgsio3/amorphous', data=data)

    print(' [DONE]')

    print('Adding optical constants of Fe...', end='')

    nk_file = os.path.join(input_path, 'optical_constants/fe/crystalline/fe_henning_1996.dat')
    data = np.loadtxt(nk_file)
    database.create_dataset('dust/fe/crystalline', data=data)

    nk_file = os.path.join(input_path, 'optical_constants/fe/amorphous/fe_pollack_1994.dat')
    data = np.loadtxt(nk_file)
    database.create_dataset('dust/fe/amorphous', data=data)

    print(' [DONE]')


@typechecked
def add_cross_sections(input_path: str,
                       database: h5py._hl.files.File) -> None:
    """
    Function for adding the extinction cross section of crystalline MgSiO3 for a log-normal and
    power-law size distribution to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    None
        NoneType
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url = 'https://home.strw.leidenuniv.nl/~stolker/species/lognorm_mgsio3_c_ext.fits'

    data_file = os.path.join(input_path, 'lognorm_mgsio3_c_ext.fits')

    print('Downloading log-normal dust cross sections (231 kB)...', end='', flush=True)
    urllib.request.urlretrieve(url, data_file)
    print(' [DONE]')

    print('Adding log-normal dust cross sections:')

    with fits.open(os.path.join(input_path, 'lognorm_mgsio3_c_ext.fits')) as hdu_list:
        database.create_dataset('dust/lognorm/mgsio3/crystalline/cross_section/',
                                data=hdu_list[0].data)

        print(f'   - Data shape (n_wavelength, n_radius, n_sigma): {hdu_list[0].data.shape}')

        database.create_dataset('dust/lognorm/mgsio3/crystalline/wavelength/',
                                data=hdu_list[1].data)

        data_range = f'{np.amin(hdu_list[1].data)} - {np.amax(hdu_list[1].data)}'
        print(f'   - Wavelength range: {data_range} um')

        database.create_dataset('dust/lognorm/mgsio3/crystalline/radius_g/',
                                data=hdu_list[2].data)

        data_range = f'{np.amin(hdu_list[2].data)} - {np.amax(hdu_list[2].data)}'
        print(f'   - Mean geometric radius range: {data_range} um')

        database.create_dataset('dust/lognorm/mgsio3/crystalline/sigma_g/',
                                data=hdu_list[3].data)

        data_range = f'{np.amin(hdu_list[3].data)} - {np.amax(hdu_list[3].data)}'
        print(f'   - Geometric standard deviation range: {data_range}')

    url = 'https://home.strw.leidenuniv.nl/~stolker/species/powerlaw_mgsio3_c_ext.fits'

    data_file = os.path.join(input_path, 'powerlaw_mgsio3_c_ext.fits')

    print('Downloading power-law dust cross sections (231 kB)...', end='', flush=True)
    urllib.request.urlretrieve(url, data_file)
    print(' [DONE]')

    print('Adding power-law dust cross sections')

    with fits.open(os.path.join(input_path, 'powerlaw_mgsio3_c_ext.fits')) as hdu_list:
        database.create_dataset('dust/powerlaw/mgsio3/crystalline/cross_section/',
                                data=hdu_list[0].data)

        print(f'   - Data shape (n_wavelength, n_radius, n_exponent): {hdu_list[0].data.shape}')

        database.create_dataset('dust/powerlaw/mgsio3/crystalline/wavelength/',
                                data=hdu_list[1].data)

        data_range = f'{np.amin(hdu_list[1].data)} - {np.amax(hdu_list[1].data)}'
        print(f'   - Wavelength range: {data_range} um')

        database.create_dataset('dust/powerlaw/mgsio3/crystalline/radius_max/',
                                data=hdu_list[2].data)

        data_range = f'{np.amin(hdu_list[2].data)} - {np.amax(hdu_list[2].data)}'
        print(f'   - Maximum grain radius range: {data_range} um')

        database.create_dataset('dust/powerlaw/mgsio3/crystalline/exponent/',
                                data=hdu_list[3].data)

        data_range = f'{np.amin(hdu_list[3].data)} - {np.amax(hdu_list[3].data)}'
        print(f'   - Power-law exponent range: {data_range}')
