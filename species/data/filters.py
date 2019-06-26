"""
Module for downloading filter data from the SVO website.
"""

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

import numpy as np


requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


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
        url = 'https://visao.as.arizona.edu/software_files/visao/html/VisAO_Ys_filter_curve.dat'

        session = requests.Session()
        response = session.get(url, verify=False)
        data = response.content

        wavelength = []
        transmission = []
        for line in data.splitlines():
            if not line.startswith(b'#'):
                split = line.split()

                wavelength.append(float(split[0]))  # [micron]
                transmission.append(float(split[1]))

        wavelength = np.asarray(wavelength)
        transmission = np.asarray(transmission)

        wavelength = wavelength[:-7]
        transmission = transmission[:-7]

    else:
        url = 'http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii&id='+filter_id

        session = requests.Session()
        response = session.get(url)
        data = response.content

        wavelength = []
        transmission = []
        for line in data.splitlines():
            if not line.startswith(b'#'):
                split = line.split(b' ')

                wavelength.append(float(split[0])*1e-4)  # [micron]
                transmission.append(float(split[1]))

        wavelength = np.asarray(wavelength)
        transmission = np.asarray(transmission)

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
