"""
Module with reading functionalities for spectral libraries.
"""

import configparser
import os

from typing import List, Optional

import h5py
import numpy as np

from typeguard import typechecked

from species.analysis import photometry
from species.core import box
from species.data import database
from species.read import read_filter


class ReadSpectrum:
    """
    Class for reading spectral library data from the database.
    """

    @typechecked
    def __init__(self, spec_library: str, filter_name: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        spec_library : str
            Name of the spectral library ('irtf', 'spex', 'kesseli+2017',
            'bonnefoy+2014', 'allers+2013', or 'vega').
        filter_name : str, None
            Filter name for the wavelength range. Full spectra
            are read if the argument is set to ``None``.

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

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

    @typechecked
    def get_spectrum(
        self, sptypes: Optional[List[str]] = None, exclude_nan: bool = True
    ) -> box.SpectrumBox:
        """
        Function for selecting spectra from the database.

        Parameters
        ----------
        sptypes : list(str), None
            Spectral types to select from a library. The spectral types
            should be indicated with two characters (e.g. 'M5', 'L2',
            'T3'). All spectra are selected if set to ``None``. For
            each object in the ``spec_library``, the requested
            ``sptypes`` are first compared with the optical spectral
            type and, if not available, secondly the near-infrared
            spectral type.
        exclude_nan : bool
            Exclude wavelength points for which the flux is NaN.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the spectra.
        """

        h5_file = h5py.File(self.database, "r")

        if self.spec_library not in h5_file[f"spectra"]:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectra(self.spec_library, sptypes)
            h5_file = h5py.File(self.database, "r")

        list_wavelength = []
        list_flux = []
        list_error = []
        list_name = []
        list_simbad = []
        list_sptype = []
        list_parallax = []
        list_spec_res = []

        for item in h5_file[f"spectra/{self.spec_library}"]:
            dset = h5_file[f"spectra/{self.spec_library}/{item}"]

            wavelength = dset[:, 0]  # (um)
            flux = dset[:, 1]  # (W m-2 um-1)
            error = dset[:, 2]  # (W m-2 um-1)

            if exclude_nan:
                nan_index = np.isnan(flux)

                wavelength = wavelength[~nan_index]
                flux = flux[~nan_index]
                error = error[~nan_index]

            if self.wavel_range is None:
                wl_index = np.arange(0, len(wavelength), 1)

            else:
                wl_index = (
                    (flux > 0.0)
                    & (wavelength > self.wavel_range[0])
                    & (wavelength < self.wavel_range[1])
                )

            count = np.count_nonzero(wl_index)

            if count > 0:
                index = np.where(wl_index)[0]

                if index[0] > 0:
                    wl_index[index[0] - 1] = True

                if index[-1] < len(wl_index) - 1:
                    wl_index[index[-1] + 1] = True

                list_wavelength.append(wavelength[wl_index])
                list_flux.append(flux[wl_index])
                list_error.append(error[wl_index])

                attrs = dset.attrs

                if "name" in attrs:
                    if isinstance(dset.attrs["name"], str):
                        list_name.append(dset.attrs["name"])
                    else:
                        list_name.append(dset.attrs["name"].decode("utf-8"))
                else:
                    list_name.append("")

                if "simbad" in attrs:
                    if isinstance(dset.attrs["simbad"], str):
                        list_simbad.append(dset.attrs["simbad"])
                    else:
                        list_simbad.append(dset.attrs["simbad"].decode("utf-8"))
                else:
                    list_simbad.append("")

                if "sptype" in attrs:
                    if isinstance(dset.attrs["sptype"], str):
                        list_sptype.append(dset.attrs["sptype"])
                    else:
                        list_sptype.append(dset.attrs["sptype"].decode("utf-8"))
                else:
                    list_sptype.append("None")

                if "parallax" in attrs:
                    list_parallax.append(
                        (dset.attrs["parallax"], dset.attrs["parallax_error"])
                    )
                else:
                    list_parallax.append((np.nan, np.nan))

                if "spec_res" in attrs:
                    list_spec_res.append(dset.attrs["spec_res"])
                else:
                    list_spec_res.append(np.nan)

            else:
                list_wavelength.append(np.array([]))
                list_flux.append(np.array([]))
                list_error.append(np.array([]))
                list_name.append("")
                list_simbad.append("")
                list_sptype.append("None")
                list_parallax.append((np.nan, np.nan))
                list_spec_res.append(np.nan)

        spec_box = box.SpectrumBox()
        spec_box.spec_library = self.spec_library

        if sptypes is not None:
            spec_box.wavelength = []
            spec_box.flux = []
            spec_box.error = []
            spec_box.name = []
            spec_box.simbad = []
            spec_box.sptype = []
            spec_box.parallax = []
            spec_box.spec_res = []

            for item in sptypes:

                for i, spec_item in enumerate(list_sptype):
                    if item == spec_item[:2]:
                        spec_box.wavelength.append(list_wavelength[i])
                        spec_box.flux.append(list_flux[i])
                        spec_box.error.append(list_error[i])
                        spec_box.name.append(list_name[i])
                        spec_box.simbad.append(list_simbad[i])
                        spec_box.sptype.append(list_sptype[i])
                        spec_box.parallax.append(list_parallax[i])
                        spec_box.spec_res.append(list_spec_res[i])

        else:
            spec_box.wavelength = list_wavelength
            spec_box.flux = list_flux
            spec_box.error = list_error
            spec_box.name = list_name
            spec_box.simbad = list_simbad
            spec_box.sptype = list_sptype
            spec_box.parallax = list_parallax
            spec_box.spec_res = list_spec_res

        return spec_box

    @typechecked
    def get_flux(self, sptypes: Optional[List[str]] = None) -> box.PhotometryBox:
        """
        Function for calculating the average flux density for the
        ``filter_name``.

        Parameters
        ----------
        sptypes : list(str), None
            Spectral types to select from a library. The spectral types
            should be indicated with two characters (e.g. 'M5', 'L2',
            'T3'). All spectra are selected if set to ``None``.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        spec_box = self.get_spectrum(sptypes=sptypes, exclude_nan=True)

        n_spectra = len(spec_box.wavelength)

        filter_profile = read_filter.ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()

        wavelengths = np.full(n_spectra, mean_wavel)
        filters = np.full(n_spectra, self.filter_name)

        synphot = photometry.SyntheticPhotometry(filter_name=self.filter_name)

        phot_flux = []

        for i in range(n_spectra):
            flux = synphot.spectrum_to_flux(
                wavelength=spec_box.wavelength[i],
                flux=spec_box.flux[i],
                error=spec_box.error[i],
            )

            phot_flux.append(flux)

        phot_flux = np.asarray(phot_flux)

        return box.create_box(
            boxtype="photometry",
            name=spec_box.name,
            sptype=spec_box.sptype,
            wavelength=wavelengths,
            flux=phot_flux,
            app_mag=None,
            abs_mag=None,
            filter_name=filters,
        )

    @typechecked
    def get_magnitude(self, sptypes: Optional[List[str]] = None) -> box.PhotometryBox:
        """
        Function for calculating the apparent magnitude for the
        specified ``filter_name``.

        Parameters
        ----------
        sptypes : list(str), None
            Spectral types to select from the library. The spectral
            types should be indicated with two characters (e.g. 'M5',
            'L2', 'T3'). All spectra are selected if set to ``None``.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        spec_box = self.get_spectrum(sptypes=sptypes, exclude_nan=True)

        n_spectra = len(spec_box.wavelength)

        filter_profile = read_filter.ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()

        wavelengths = np.full(n_spectra, mean_wavel)
        filters = np.full(n_spectra, self.filter_name)

        synphot = photometry.SyntheticPhotometry(filter_name=self.filter_name)

        app_mag = []
        abs_mag = []

        for i in range(n_spectra):

            if np.isnan(spec_box.parallax[i][0]):
                app_tmp = (np.nan, np.nan)
                abs_tmp = (np.nan, np.nan)

            else:

                app_tmp, abs_tmp = synphot.spectrum_to_magnitude(
                    spec_box.wavelength[i],
                    spec_box.flux[i],
                    error=spec_box.error[i],
                    parallax=(
                        float(spec_box.parallax[i][0]),
                        float(spec_box.parallax[i][1]),
                    ),
                )

            app_mag.append(app_tmp)
            abs_mag.append(abs_tmp)

        return box.create_box(
            boxtype="photometry",
            name=spec_box.name,
            sptype=spec_box.sptype,
            wavelength=wavelengths,
            flux=None,
            app_mag=np.asarray(app_mag),
            abs_mag=np.asarray(abs_mag),
            filter_name=filters,
        )
