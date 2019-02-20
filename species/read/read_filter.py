"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from .. data import database


class ReadFilter:
    """
    Text
    """

    def __init__(self, filter_name):
        """
        :param filter_name: Filter name.
        :type filter_name: str

        :return: None
        """

        self.filter_name = filter_name

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_filter(self):
        """
        :return: Filter data.
        :rtype: numpy.ndarray
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['filters/'+self.filter_name]

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_filter(self.filter_name)
            h5_file = h5py.File(self.database, 'r')

        data = h5_file['filters/'+self.filter_name]
        data = np.asarray(data)

        h5_file.close()

        return data

    def wavelength_range(self):
        """
        :return: Minimum and maximum wavelength (micron).
        :rtype: float, float
        """

        data = self.get_filter()

        return np.amin(data[0, ]), np.amax(data[0, ])

    def mean_wavelength(self):
        """
        :return: Mean wavelength (micron).
        :rtype: float
        """

        data = self.get_filter()

        return np.trapz(data[0, ]*data[1, ], data[0, ]) / np.trapz(data[1, ], data[0, ])

    def interpolate(self):
        """
        :return: Interpolated filter.
        :rtype: scipy.interpolate.interpolate.interp1d
        """

        data = self.get_filter()

        filter_interp = interp1d(data[0, ],
                                 data[1, ],
                                 kind='cubic',
                                 bounds_error=False,
                                 fill_value=float('nan'))

        return filter_interp

    def filter_fwhm(self):
        """
        :return: Filter full width at half maximum (micron).
        :rtype: float
        """

        data = self.get_filter()

        spline = InterpolatedUnivariateSpline(data[0, :], data[1, :]-np.max(data[1, :])/2.)
        root = spline.roots()

        diff = root-self.mean_wavelength()

        root1 = np.amax(diff[diff < 0.])
        root2 = np.amin(diff[diff > 0.])

        return root2-root1
