"""
Module for downloading filter data from the SVO website.
"""

import os

import wget
import numpy as np


def download_filter(filter_id):
    """
    Parameters
    ----------
    filter_id : str
        Filter ID.

    Returns
    -------
    numpy.ndarray
        Wavelength (micron).
    numpy.ndarray
        Transmission.
    """

    if filter_id == 'LCO/VisAO.Ys':
        url = 'https://xwcl.science/magao/visao/VisAO_Ys_filter_curve.dat'

        wget.download(url, out='VisAO_Ys_filter_curve.dat', bar=None)

        wavelength, transmission, _, _ = np.loadtxt('VisAO_Ys_filter_curve.dat', unpack=True)

        wavelength = wavelength[:-7]
        transmission = transmission[:-7]

        os.remove('VisAO_Ys_filter_curve.dat')

    else:
        url = 'http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii&id='+filter_id

        wget.download(url, out='filter.dat', bar=None)

        if os.stat('filter.dat').st_size == 0:
            raise ValueError(f'Filter {filter_id} not found on the SVO Filter Profile Service.')

        else:
            wavelength, transmission = np.loadtxt('filter.dat', unpack=True)
            wavelength *= 1e-4  # [micron]

        os.remove('filter.dat')

    if wavelength is not None:
        indices = []

        for i in range(transmission.size):
            if i == 0 and transmission[i] == 0. and transmission[i+1] == 0.:
                indices.append(i)

            elif i == transmission.size-1 and transmission[i-1] == 0. and transmission[i] == 0.:
                indices.append(i)

            elif transmission[i-1] == 0. and transmission[i] == 0. and transmission[i+1] == 0.:
                indices.append(i)

        wavelength = np.delete(wavelength, indices)
        transmission = np.delete(transmission, indices)

    return wavelength, transmission
