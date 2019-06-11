"""
Module for reading isochrones data from the database.
"""

import os
import configparser

import h5py
import numpy as np

from scipy.interpolate import griddata

from species.core import box, constants
from species.data import database


class ReadIsochrone:
    """
    Reading filter data and information from the database.
    """

    def __init__(self,
                 tag,
                 filters_color,
                 filter_mag):
        """
        Parameters
        ----------
        tag : str
            Database tag.
        filters_color : tuple(str, str)
            Filter IDs for the color as listed in the file with the isochrone data.
        filter_mag : str
            Filter ID for the absolute magnitude as listed in the file with the isochrone data.

        Returns
        -------
        NoneType
            None
        """

        self.tag = tag
        self.filters_color = filters_color
        self.filter_mag = filter_mag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_isochrone(self,
                      age):
        """
        Parameters
        ----------
        age : str
            Age (Myr) that is used to interpolate the isochrone data.
        filter_id : str
            Filter ID for which the magnitudes are interpolated. Should be the same as in the
            isochrone data file.

        Returns
        -------
        NoneType
            None
        """

        with  h5py.File(self.database, 'r') as h5_file:
            filters = list(h5_file['isochrones/'+self.tag+'/filters'])
            isochrones = np.asarray(h5_file['isochrones/'+self.tag+'/magnitudes'])

        for i, item in enumerate(filters):
            filters[i] = item.decode()

        # skip 8 columns which contain the age, mass, Teff, etc.
        index_color_1 = 8 + filters.index(self.filters_color[0])
        index_color_2 = 8 + filters.index(self.filters_color[1])
        index_mag = 8 + filters.index(self.filter_mag)

        masses = np.unique(isochrones[:, 1]) # [Mjup]
        ages = np.repeat(age, masses.shape[0]) # [Myr]

        mag_color_1 = griddata(points=isochrones[:, 0:2],
                               values=isochrones[:, index_color_1],
                               xi=np.stack((ages, masses), axis=1),
                               method='linear',
                               fill_value='nan',
                               rescale=False)

        mag_color_2 = griddata(points=isochrones[:, 0:2],
                               values=isochrones[:, index_color_2],
                               xi=np.stack((ages, masses), axis=1),
                               method='linear',
                               fill_value='nan',
                               rescale=False)

        mag_abs = griddata(points=isochrones[:, 0:2],
                           values=isochrones[:, index_mag],
                           xi=np.stack((ages, masses), axis=1),
                           method='linear',
                           fill_value='nan',
                           rescale=False)

        return box.create_box(boxtype='isochrone',
                              model=self.tag,
                              filters_color=self.filters_color,
                              filter_mag=self.filter_mag,
                              color=mag_color_1-mag_color_2,
                              magnitude=mag_abs)
