"""
Text
"""

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

import numpy as np


requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def download_filter(filter_id):
    """
    :param filter_id: Filter ID.
    :type filter_id: str

    :return:
    :rtype: numpy.ndarray, numpy.ndarray
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

                wavelength.append(float(split[0])) # [micron]
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

                wavelength.append(float(split[0])*1e-4) # [micron]
                transmission.append(float(split[1]))

        wavelength = np.asarray(wavelength)
        transmission = np.asarray(transmission)

    return wavelength, transmission
