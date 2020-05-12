"""
Module for downloading filter data from the SVO website.
"""

import os
import warnings
import urllib.request

from typing import Optional, Tuple
from astropy.io.votable import parse_single_table

import numpy as np

from typeguard import typechecked


@typechecked
def download_filter(filter_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Function for downloading filter transmission data from the SVO Filter Profile Service.

    Parameters
    ----------
    filter_id : str
        Filter name as listed on the SVO website.

    Returns
    -------
    np.ndarray
        Wavelength (um).
    np.ndarray
        Transmission.
    """

    if filter_id == 'LCO/VisAO.Ys':
        url = 'https://xwcl.science/magao/visao/VisAO_Ys_filter_curve.dat'
        urllib.request.urlretrieve(url, 'VisAO_Ys_filter_curve.dat')

        wavelength, transmission, _, _ = np.loadtxt('VisAO_Ys_filter_curve.dat', unpack=True)

        wavelength = wavelength[:-7]
        transmission = transmission[:-7]

        os.remove('VisAO_Ys_filter_curve.dat')

    else:
        url = 'http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID='+filter_id
        urllib.request.urlretrieve(url, 'filter.xml')

        try:
            table = parse_single_table('filter.xml')

            wavelength = table.array['Wavelength']
            transmission = table.array['Transmission']

        except IndexError:
            wavelength = None
            transmission = None

            warnings.warn(f'Filter \'{filter_id}\' is not available on the SVO Filter Profile '
                          f'Service.')

        except:
            os.remove('filter.xml')

            raise ValueError(f'The filter data of \'{filter_id}\' could not be downloaded. '
                             f'Perhaps the website of the SVO Filter Profile Service '
                             f'(http://svo2.cab.inta-csic.es/svo/theory/fps/) is not '
                             f'available?')

        if transmission is not None:
            try:
                det_type = table.get_field_by_id('DetectorType').value
                det_type = det_type.decode('utf-8')

                if int(det_type) == 1:
                    det_type = 'photon'

            except KeyError:
                det_type = 'energy'

            if det_type == 'photon':
                raise ValueError(f'The detector of the {filter_id} filter is a photon counter, '
                                 f'therefore the transmission profile has to be multiplied with '
                                 f'the wavelength when calculating average fluxes. This is '
                                 f'currently not implemented in species. Please open an issue on '
                                 f'Github if needed.')

            wavelength *= 1e-4  # (um)

        os.remove('filter.xml')

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

        if np.amin(transmission) < 0.:
            raise ValueError('The minimum transmission value is smaller than zero.')

        if np.amax(transmission) > 1.:
            raise ValueError('The maximum transmission value is larger than one.')

    return wavelength, transmission
