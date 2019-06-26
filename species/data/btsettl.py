"""
Module for BT-Settl atmospheric models.
"""

import os
import sys
import lzma
import tarfile

from urllib.request import urlretrieve

import numpy as np

from scipy.interpolate import interp1d

from species.util import data_util


def add_btsettl(input_path,
                database,
                wl_bound,
                teff_bound,
                specres):
    """
    Function for adding the BT-Settl atmospheric models to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wl_bound : tuple(float, float)
        Wavelength range (micron).
    teff_bound : tuple(float, float)
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

    data_folder = os.path.join(input_path, 'bt-settl/')

    input_file = 'BT-Settl_M-0.0a+0.0.tar'
    label = '[Fe/H]=0.0 (6.0 GB)'

    url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/' \
          'SPECTRA/BT-Settl_M-0.0a+0.0.tar'

    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        sys.stdout.write(f'Downloading BT-Settl model spectra {label}...')
        sys.stdout.flush()

        urlretrieve(url, data_file)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    sys.stdout.write(f'Unpacking BT-Settl model spectra {label}...')
    sys.stdout.flush()

    # tar = tarfile.open(data_file)
    # tar.extractall(data_folder)
    # tar.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    teff = []
    logg = []
    flux = []

    wavelength = [wl_bound[0]]

    while wavelength[-1] <= wl_bound[1]:
        wavelength.append(wavelength[-1] + wavelength[-1]/specres)

    wavelength = np.asarray(wavelength[:-1])

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename.startswith('lte') and filename.endswith('.7.xz'):
                sys.stdout.write('\rAdding BT-Settl model spectra... '+filename)
                sys.stdout.flush()

                teff_val = float(filename[3:6])*100.

                if teff_bound[0] <= teff_val <= teff_bound[1]:
                    teff.append(teff_val)
                    logg.append(float(filename[9:12]))

                else:
                    continue

                data_wavel = []
                data_flux = []

                with lzma.open(data_folder+filename) as xz_file:
                    for line in xz_file:
                        # See https://phoenix.ens-lyon.fr/Grids/FORMAT

                        # [Angstrom] -> [micron]
                        data_wavel.append(float(line[1:13])*1e-4)

                        # [erg s-1 cm-2 Angstrom-1]
                        flux_tmp = 10.**(float(line[14:25].replace(b'D', b'E'))-8.)

                        # [erg s-1 cm-2 Angstrom-1] -> [W m-2 micron-1]
                        data_flux.append(flux_tmp*1e-7*1e4*1e4)

                data_wavel = np.asarray(data_wavel)
                data_flux = np.asarray(data_flux)

                indices = np.where((data_wavel >= wl_bound[0]) &
                                   (data_wavel <= wl_bound[1]))[0]

                flux_interp = interp1d(data_wavel[indices],
                                       data_flux[indices],
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value=float('nan'))

                flux.append(flux_interp(wavelength))

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      None,
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('bt-settl', ('teff', 'logg'), database, data_sorted)

    sys.stdout.write('\rAdding BT-Settl model spectra... [DONE]'
                     '                                  \n')

    sys.stdout.flush()
