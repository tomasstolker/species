"""
Text
"""

import os
import urllib.request

import numpy as np

from astropy.io import fits


def add_vega(input_path, database):
    """
    Function for adding a Vega spectrum to the database.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    NoneType
        None
    """

    data_file = os.path.join(input_path, 'alpha_lyr_stis_008.fits')
    url = 'http://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_008.fits'

    if not os.path.isfile(data_file):
        print('Downloading Vega spectrum (270 kB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    if 'spectra/calibration' not in database:
        database.create_group('spectra/calibration')

    if 'spectra/calibration/vega' in database:
        del database['spectra/calibration/vega']

    hdu = fits.open(data_file)
    data = hdu[1].data
    wavelength = data['WAVELENGTH']  # [Angstrom]
    flux = data['FLUX']  # [erg s-1 cm-2 A-1]
    error_stat = data['STATERROR']  # [erg s-1 cm-2 A-1]
    error_sys = data['SYSERROR']  # [erg s-1 cm-2 A-1]
    hdu.close()

    wavelength *= 1e-4  # [Angstrom] -> [micron]
    flux *= 1.e-3*1e4  # [erg s-1 cm-2 A-1] -> [W m-2 micron-1]
    error_stat *= 1.e-3*1e4  # [erg s-1 cm-2 A-1] -> [W m-2 micron-1]
    error_sys *= 1.e-3*1e4  # [erg s-1 cm-2 A-1] -> [W m-2 micron-1]

    print('Adding Vega spectrum...', end='', flush=True)

    database.create_dataset('spectra/calibration/vega',
                            data=np.vstack((wavelength, flux, error_stat)))

    print(' [DONE]')
