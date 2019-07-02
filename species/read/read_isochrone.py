"""
Module for reading isochrones data from the database.
"""

import os
import configparser

import h5py
import numpy as np

from scipy.interpolate import griddata

from species.core import box
from species.read import read_model


class ReadIsochrone:
    """
    Reading filter data and information from the database.
    """

    def __init__(self,
                 tag):
        """
        Parameters
        ----------
        tag : str
            Database tag.

        Returns
        -------
        NoneType
            None
        """

        self.tag = tag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_isochrone(self,
                      age,
                      mass,
                      filters_color,
                      filter_mag):
        """
        Parameters
        ----------
        age : str
            Age (Myr) that is used to interpolate the isochrone data.
        mass : numpy.ndarray
            Masses (Mjup) for which the isochrone data is interpolated.
        filters_color : tuple(str, str), None
            Filter IDs for the color as listed in the file with the isochrone data. Not selected if
            set to None or if only evolutionary tracks are available.
        filter_mag : str, None
            Filter ID for the absolute magnitude as listed in the file with the isochrone data. Not
            selected if set to None or if only evolutionary tracks are available.

        Returns
        -------
        species.core.box.IsochroneBox
            Box with the isochrone data.
        """

        age_points = np.repeat(age, mass.shape[0])  # [Myr]

        color = None
        mag_abs = None

        index_teff = 2
        index_logg = 4

        with h5py.File(self.database, 'r') as h5_file:
            model = h5_file['isochrones/'+self.tag+'/evolution'].attrs['model']
            evolution = np.asarray(h5_file['isochrones/'+self.tag+'/evolution'])

            if model == 'baraffe':
                filters = list(h5_file['isochrones/'+self.tag+'/filters'])
                magnitudes = np.asarray(h5_file['isochrones/'+self.tag+'/magnitudes'])

        if model == 'baraffe':
            for i, item in enumerate(filters):
                filters[i] = item.decode()

            if filters_color is not None:
                index_color_1 = filters.index(filters_color[0])
                index_color_2 = filters.index(filters_color[1])

            if filter_mag is not None:
                index_mag = filters.index(filter_mag)

            if filters_color is not None:
                mag_color_1 = griddata(points=evolution[:, 0:2],
                                       values=magnitudes[:, index_color_1],
                                       xi=np.stack((age_points, mass), axis=1),
                                       method='linear',
                                       fill_value='nan',
                                       rescale=False)

                mag_color_2 = griddata(points=evolution[:, 0:2],
                                       values=magnitudes[:, index_color_2],
                                       xi=np.stack((age_points, mass), axis=1),
                                       method='linear',
                                       fill_value='nan',
                                       rescale=False)

                color = mag_color_1-mag_color_2

            if filter_mag is not None:
                mag_abs = griddata(points=evolution[:, 0:2],
                                   values=magnitudes[:, index_mag],
                                   xi=np.stack((age_points, mass), axis=1),
                                   method='linear',
                                   fill_value='nan',
                                   rescale=False)

        teff = griddata(points=evolution[:, 0:2],
                        values=evolution[:, index_teff],
                        xi=np.stack((age_points, mass), axis=1),
                        method='linear',
                        fill_value='nan',
                        rescale=False)

        logg = griddata(points=evolution[:, 0:2],
                        values=evolution[:, index_logg],
                        xi=np.stack((age_points, mass), axis=1),
                        method='linear',
                        fill_value='nan',
                        rescale=False)

        return box.create_box(boxtype='isochrone',
                              model=self.tag,
                              filters_color=filters_color,
                              filter_mag=filter_mag,
                              color=color,
                              magnitude=mag_abs,
                              teff=teff,
                              logg=logg,
                              mass=mass)

    def get_color_magnitude(self,
                            age,
                            mass,
                            model,
                            filters_color,
                            filter_mag):
        """
        Parameters
        ----------
        age : str
            Age (Myr) that is used to interpolate the isochrone data.
        mass : numpy.ndarray
            Masses (Mjup) for which the isochrone data is interpolated.
        model : str
            Atmospheric model used to compute the synthetic photometry.
        filters_color : tuple(str, str), None
            Filter IDs for the color as listed in the file with the isochrone data. Not selected if
            set to None or if only evolutionary tracks are available.
        filter_mag : str, None
            Filter ID for the absolute magnitude as listed in the file with the isochrone data. Not
            selected if set to None or if only evolutionary tracks are available.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the isochrone data.
        """

        isochrone = self.get_isochrone(age=age,
                                       mass=mass,
                                       filters_color=None,
                                       filter_mag=None)

        model1 = read_model.ReadModel(model=model, wavelength=filters_color[0])
        model2 = read_model.ReadModel(model=model, wavelength=filters_color[1])

        mag1 = np.zeros(isochrone.mass.shape[0])
        mag2 = np.zeros(isochrone.mass.shape[0])

        for i, item in enumerate(isochrone.mass):
            model_par = {'teff': isochrone.teff[i],
                         'logg': isochrone.logg[i],
                         'feh': 0.,
                         'mass': item,
                         'distance': 10.}

            mag1[i], _ = model1.get_magnitude(model_par=model_par)
            mag2[i], _ = model2.get_magnitude(model_par=model_par)

        if filter_mag == filters_color[0]:
            abs_mag = mag1

        elif filter_mag == filters_color[1]:
            abs_mag = mag2

        else:
            raise ValueError('The filter_mag argument should be equal to one of the two filter '
                             'values of filters_color.')

        return box.create_box(boxtype='colormag',
                              library=model,
                              object_type='temperature',
                              filters_color=filters_color,
                              filter_mag=filter_mag,
                              color=mag1-mag2,
                              magnitude=abs_mag,
                              sptype=isochrone.teff)
