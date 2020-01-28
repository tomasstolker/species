"""
Module with reading functionalities for spectral libraries.
"""

import os
import configparser

import h5py
import numpy as np

from species.analysis import photometry
from species.core import box
from species.data import database
from species.read import read_filter


class ReadSpectrum:
    """
    Class for reading spectral library data from the database.
    """

    def __init__(self,
                 spec_library,
                 filter_name=None):
        """
        Parameters
        ----------
        spec_library : str
            Name of the spectral library ('irtf', 'spex') or other type of spectrum ('vega').
        filter_name : str, None
            Filter ID for the wavelength range. Full spectra are read if set to None.

        Returns
        -------
        NoneType
            None
        """

        self.spec_library = spec_library
        self.filter_name = filter_name

        if filter_name is None:
            self.wavel_range = None

        else:
            transmission = read_filter.ReadFilter(filter_name)
            self.wavel_range = transmission.wavelength_range()

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_spectrum(self,
                     sptypes=None,
                     exclude_nan=True):
        """
        Function for selecting spectra from the database.

        Parameters
        ----------
        sptypes : list('str', )
            Spectral types to select from a library. The spectral types should be indicated with
            two characters (e.g. 'M5', 'L2', 'T3'). All spectra are selected if set to None.
        exclude_nan : bool
            Exclude wavelength points for which the flux is NaN.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the spectra.
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['spectra/'+self.spec_library]

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectrum(self.spec_library, sptypes)
            h5_file = h5py.File(self.database, 'r')

        list_wavelength = []
        list_flux = []
        list_name = []
        list_simbad = []
        list_sptype = []
        list_distance = []

        for item in h5_file['spectra/'+self.spec_library]:
            data = h5_file['spectra/'+self.spec_library+'/'+item]

            wavelength = data[0, :]  # [micron]
            flux = data[1, :]  # [W m-2 micron-1]

            if data.shape[0] == 3:
                error = data[2, :]  # [W m-2 micron-1]

            if exclude_nan:
                indices = np.isnan(flux)
                indices = np.logical_not(indices)
                indices = np.where(indices)[0]

                wavelength = wavelength[indices]
                flux = flux[indices]

                if data.shape[0] == 3:
                    error = error[indices]

            if self.wavel_range is None:
                wl_index = np.arange(0, len(wavelength), 1)

            else:
                wl_index = (flux > 0.) & (wavelength > self.wavel_range[0]) & \
                           (wavelength < self.wavel_range[1])

            count = np.count_nonzero(wl_index)

            if count > 0:
                index = np.where(wl_index)[0]

                if index[0] > 0:
                    wl_index[index[0] - 1] = True

                if index[-1] < len(wl_index)-1:
                    wl_index[index[-1] + 1] = True

                list_wavelength.append(wavelength[wl_index])
                list_flux.append(flux[wl_index])

                attrs = data.attrs
                if 'name' in attrs:
                    list_name.append(data.attrs['name'])

                if 'simbad' in attrs:
                    list_simbad.append(data.attrs['simbad'])

                if 'sptype' in attrs:
                    list_sptype.append(data.attrs['sptype'])

                if 'distance' in attrs:
                    list_distance.append(data.attrs['distance'])

        specbox = box.SpectrumBox()

        specbox.spec_library = self.spec_library
        specbox.wavelength = np.asarray(list_wavelength)
        specbox.flux = np.asarray(list_flux)

        if list_name:
            specbox.name = np.asarray(list_name)

        if list_simbad:
            specbox.simbad = np.asarray(list_simbad)

        if list_sptype:
            specbox.sptype = np.asarray(list_sptype)

        if list_distance:
            specbox.distance = np.asarray(list_distance)

        if sptypes is not None:
            indices = None

            for item in sptypes:
                if indices is None:
                    indices = np.where(np.chararray.startswith(specbox.sptype, item.encode()))[0]

                else:
                    ind_tmp = np.where(np.chararray.startswith(specbox.sptype, item.encode()))[0]
                    indices = np.append(indices, ind_tmp)

            specbox.wavelength = specbox.wavelength[indices]
            specbox.flux = specbox.flux[indices]
            specbox.name = specbox.name[indices]
            specbox.simbad = specbox.simbad[indices]
            specbox.sptype = specbox.sptype[indices]
            specbox.distance = specbox.distance[indices]

        return specbox

    def get_flux(self,
                 sptypes=None):
        """
        Function for calculating the average flux density for the ``filter_name``.

        Parameters
        ----------
        sptypes : list('str', )
            Spectral types to select from a library. The spectral types should be indicated with
            two characters (e.g. 'M5', 'L2', 'T3'). All spectra are selected if set to None.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        specbox = self.get_spectrum(sptypes=sptypes,
                                    exclude_nan=True)

        filter_profile = read_filter.ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()

        synphot = photometry.SyntheticPhotometry(filter_name=self.filter_name)

        phot_wavel = []
        phot_flux = []

        for i, _ in enumerate(specbox.wavelength):
            flux = synphot.spectrum_to_flux(wavelength=specbox.wavelength[i],
                                            flux=specbox.flux[i])

            phot_wavel.append(mean_wavel)
            phot_flux.append(flux)

        return box.create_box(boxtype='photometry',
                              name=specbox.name,
                              sptype=specbox.sptype,
                              wavelength=phot_wavel,
                              flux=phot_flux,
                              app_mag=None,
                              abs_mag=None)

    def get_magnitude(self,
                      sptypes=None):
        """
        Function for calculating the apparent magnitude for the ``filter_name``.

        Parameters
        ----------
        sptypes : list('str', )
            Spectral types to select from a library. The spectral types should be indicated with
            two characters (e.g. 'M5', 'L2', 'T3'). All spectra are selected if set to None.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        specbox = self.get_spectrum(sptypes=sptypes,
                                    exclude_nan=True)

        filter_profile = read_filter.ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()
        wavelength = np.full(specbox.wavelength.shape[0], mean_wavel)

        synphot = photometry.SyntheticPhotometry(filter_name=self.filter_name)

        app_mag, abs_mag = synphot.spectrum_to_magnitude(wavelength=specbox.wavelength,
                                                         flux=specbox.flux,
                                                         distance=specbox.distance)

        return box.create_box(boxtype='photometry',
                              name=specbox.name,
                              sptype=specbox.sptype,
                              wavelength=wavelength,
                              flux=None,
                              app_mag=app_mag,
                              abs_mag=abs_mag)
