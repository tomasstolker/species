"""
Module for reading filter data from the database.
"""

import os
import configparser

import h5py
import numpy as np

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from species.data import database


class ReadFilter:
    """
    Reading filter data and information from the database.
    """

    def __init__(self,
                 filter_name):
        """
        Parameters
        ----------
        filter_name : str
            Filter ID.

        Returns
        -------
        None
        """

        self.filter_name = filter_name

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_filter(self):
        """
        Returns
        -------
        numpy.ndarray
            Filter data.
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

    def interpolate(self):
        """
        Returns
        -------
        scipy.interpolate.interpolate.interp1d
            Linearly interpolated filter.
        """

        data = self.get_filter()

        return interp1d(data[0, ],
                        data[1, ],
                        kind='linear',
                        bounds_error=False,
                        fill_value=float('nan'))

    def wavelength_range(self):
        """
        Returns
        -------
        float
            Minimum wavelength (micron).
        float
            Maximum wavelength (micron).
        """

        data = self.get_filter()

        return np.amin(data[0, ]), np.amax(data[0, ])

    def mean_wavelength(self):
        """
        Returns
        -------
        float
            Mean wavelength (micron).
        """

        data = self.get_filter()

        return np.trapz(data[0, ]*data[1, ], data[0, ]) / np.trapz(data[1, ], data[0, ])

    def filter_fwhm(self):
        """
        Returns
        -------
        float
            Filter full width at half maximum (micron).
        """

        data = self.get_filter()

        spline = InterpolatedUnivariateSpline(data[0, :], data[1, :]-np.max(data[1, :])/2.)
        root = spline.roots()

        diff = root-self.mean_wavelength()

        root1 = np.amax(diff[diff < 0.])
        root2 = np.amin(diff[diff > 0.])

        return root2-root1
