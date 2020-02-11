"""
Module with reading functionalities for isochrones.
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
    Class for reading an isochrone from the database.
    """

    def __init__(self,
                 tag):
        """
        Parameters
        ----------
        tag : str
            Database tag of the isochrone data.

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
                      masses,
                      filters_color,
                      filter_mag):
        """
        Function for selecting an isochrone.

        Parameters
        ----------
        age : str
            Age (Myr) at which the isochrone data is interpolated.
        masses : numpy.ndarray
            Masses (Mjup) at which the isochrone data is interpolated.
        filters_color : tuple(str, str), None
            Filter IDs for the color as listed in the file with the isochrone data. Not selected if
            set to None or if only evolutionary tracks are available.
        filter_mag : str, None
            Filter ID for the absolute magnitude as listed in the file with the isochrone data. Not
            selected if set to None or if only evolutionary tracks are available.

        Returns
        -------
        species.core.box.IsochroneBox
            Box with the isochrone.
        """

        age_points = np.repeat(age, masses.shape[0])  # [Myr]

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
                                       xi=np.stack((age_points, masses), axis=1),
                                       method='linear',
                                       fill_value='nan',
                                       rescale=False)

                mag_color_2 = griddata(points=evolution[:, 0:2],
                                       values=magnitudes[:, index_color_2],
                                       xi=np.stack((age_points, masses), axis=1),
                                       method='linear',
                                       fill_value='nan',
                                       rescale=False)

                color = mag_color_1-mag_color_2

            if filter_mag is not None:
                mag_abs = griddata(points=evolution[:, 0:2],
                                   values=magnitudes[:, index_mag],
                                   xi=np.stack((age_points, masses), axis=1),
                                   method='linear',
                                   fill_value='nan',
                                   rescale=False)

        teff = griddata(points=evolution[:, 0:2],
                        values=evolution[:, index_teff],
                        xi=np.stack((age_points, masses), axis=1),
                        method='linear',
                        fill_value='nan',
                        rescale=False)

        logg = griddata(points=evolution[:, 0:2],
                        values=evolution[:, index_logg],
                        xi=np.stack((age_points, masses), axis=1),
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
                              masses=masses)

    def get_color_magnitude(self,
                            age,
                            masses,
                            model,
                            filters_color,
                            filter_mag):
        """
        Function for calculating color-magnitude combinations from a selected isochrone.

        Parameters
        ----------
        age : str
            Age (Myr) at which the isochrone data is interpolated.
        masses : numpy.ndarray
            Masses (Mjup) at which the isochrone data is interpolated.
        model : str
            Atmospheric model used to compute the synthetic photometry.
        filters_color : tuple(str, str)
            Filter IDs for the color as listed in the file with the isochrone data. The filter IDs
            should be provided in the format of the SVO Filter Profile Service.
        filter_mag : str
            Filter ID for the absolute magnitude as listed in the file with the isochrone data. The
            value should be equal to one of the ``filters_color`` values.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the color-magnitude data.
        """

        isochrone = self.get_isochrone(age=age,
                                       masses=masses,
                                       filters_color=None,
                                       filter_mag=None)

        model1 = read_model.ReadModel(model=model, filter_name=filters_color[0])
        model2 = read_model.ReadModel(model=model, filter_name=filters_color[1])

        if model1.get_parameters() != ['teff', 'logg']:
            raise ValueError('Creating synthetic colors and magnitudes from isochrones is '
                             'currently only implemented for models with only Teff and log(g) '
                             'as free parameters. Please contact Tomas Stolker if additional '
                             'functionalities are required.')

        mag1 = np.zeros(isochrone.masses.shape[0])
        mag2 = np.zeros(isochrone.masses.shape[0])

        for i, mass_item in enumerate(isochrone.masses):
            model_param = {'teff': isochrone.teff[i],
                           'logg': isochrone.logg[i],
                           'mass': mass_item,
                           'distance': 10.}

            if np.isnan(isochrone.teff[i]):
                mag1[i] = np.nan
                mag2[i] = np.nan

            else:
                for item_bounds in model1.get_bounds():
                    if model_param[item_bounds] <= model1.get_bounds()[item_bounds][0]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan

                    elif model_param[item_bounds] >= model1.get_bounds()[item_bounds][1]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan

                if not np.isnan(mag1[i]):
                    mag1[i], _ = model1.get_magnitude(model_param=model_param)
                    mag2[i], _ = model2.get_magnitude(model_param=model_param)

        if filter_mag == filters_color[0]:
            abs_mag = mag1

        elif filter_mag == filters_color[1]:
            abs_mag = mag2

        else:
            raise ValueError('The filter_mag argument should be equal to one of the two filter '
                             'values of filters_color.')

        return box.create_box(boxtype='colormag',
                              library=model,
                              object_type='model',
                              filters_color=filters_color,
                              filter_mag=filter_mag,
                              color=mag1-mag2,
                              magnitude=abs_mag,
                              sptype=masses)

    def get_color_color(self,
                        age,
                        masses,
                        model,
                        filters):
        """
        Function for calculating color-magnitude combinations from a selected isochrone.

        Parameters
        ----------
        age : str
            Age (Myr) at which the isochrone data is interpolated.
        masses : numpy.ndarray
            Masses (Mjup) at which the isochrone data is interpolated.
        model : str
            Atmospheric model used to compute the synthetic photometry.
        filters : tuple(tuple(str, str), tuple(str, str))
            Filter IDs for the colors as listed in the file with the isochrone data. The filter IDs
            should be provided in the format of the SVO Filter Profile Service.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the color-color data.
        """

        isochrone = self.get_isochrone(age=age,
                                       masses=masses,
                                       filters_color=None,
                                       filter_mag=None)

        model1 = read_model.ReadModel(model=model, filter_name=filters[0][0])
        model2 = read_model.ReadModel(model=model, filter_name=filters[0][1])
        model3 = read_model.ReadModel(model=model, filter_name=filters[1][0])
        model4 = read_model.ReadModel(model=model, filter_name=filters[1][1])

        if model1.get_parameters() != ['teff', 'logg']:
            raise ValueError('Creating synthetic colors and magnitudes from isochrones is '
                             'currently only implemented for models with only Teff and log(g) '
                             'as free parameters. Please contact Tomas Stolker if additional '
                             'functionalities are required.')

        mag1 = np.zeros(isochrone.masses.shape[0])
        mag2 = np.zeros(isochrone.masses.shape[0])
        mag3 = np.zeros(isochrone.masses.shape[0])
        mag4 = np.zeros(isochrone.masses.shape[0])

        for i, mass_item in enumerate(isochrone.masses):
            model_param = {'teff': isochrone.teff[i],
                           'logg': isochrone.logg[i],
                           'mass': mass_item,
                           'distance': 10.}

            if np.isnan(isochrone.teff[i]):
                mag1[i] = np.nan
                mag2[i] = np.nan

            else:
                for item_bounds in model1.get_bounds():
                    if model_param[item_bounds] <= model1.get_bounds()[item_bounds][0]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan

                    elif model_param[item_bounds] >= model1.get_bounds()[item_bounds][1]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan

                if not np.isnan(mag1[i]):
                    mag1[i], _ = model1.get_magnitude(model_param=model_param)
                    mag2[i], _ = model2.get_magnitude(model_param=model_param)
                    mag3[i], _ = model3.get_magnitude(model_param=model_param)
                    mag4[i], _ = model4.get_magnitude(model_param=model_param)

        return box.create_box(boxtype='colorcolor',
                              library=model,
                              object_type='model',
                              filters=filters,
                              color1=mag1-mag2,
                              color2=mag3-mag4,
                              sptype=masses)
