"""
Module with reading functionalities for spectral libraries.
"""

import os

from configparser import ConfigParser
from typing import List, Optional, Union

import h5py
import numpy as np

from typeguard import typechecked

from species.core.box import PhotometryBox, SpectrumBox, create_box
from species.data.spec_data.add_spec_data import add_spec_library
from species.phot.syn_phot import SyntheticPhotometry
from species.read.read_filter import ReadFilter
from species.util.dust_util import ism_extinction


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
            transmission = ReadFilter(filter_name)
            self.wavel_range = transmission.wavelength_range()

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.data_folder = config["species"]["data_folder"]

    @typechecked
    def get_spectrum(
        self,
        object_name: Optional[Union[str, List[str]]] = None,
        sptypes: Optional[List[str]] = None,
        exclude_nan: bool = True,
        av_ext: Optional[float] = None,
    ) -> SpectrumBox:
        """
        Function for selecting spectra from the database.

        Parameters
        ----------
        object_name : str, list(str), None
            Object name of which the spectrum will be selected from the
            spectral library. Either a single object or a list of object
            names. By setting the argument to ``None``, the ``sptype``
            parameter will be used.
        sptypes : list(str), None
            Spectral types to select from a library. The spectral types
            should be indicated with two characters (e.g. 'M5', 'L2',
            'T3'). All spectra are selected if set to ``None``. For
            each object in the ``spec_library``, the requested
            ``sptypes`` are first compared with the optical spectral
            type and, if not available, secondly the near-infrared
            spectral type. To use the ``sptypes`` parameter, the
            argument of ``object_name`` should be set to ``None``
        exclude_nan : bool
            Exclude wavelength points for which the flux is NaN.
        av_ext : float, None
            Visual extinction, :math:`A_V`, applied to the spectra.
            The extinction is calculated with the empirical relation
            from `Cardelli et al. (1989)
            <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/
            abstract>`_. The extinction is not applied if the argument
            is set to ``None``.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the spectra.
        """

        with h5py.File(self.database, "r") as hdf5_file:
            # Check if the spectral library is found in
            # 'r' mode because the 'a' mode is not possible
            # when using multiprocessing
            spec_found = f"spectra/{self.spec_library}" in hdf5_file

        if not spec_found:
            with h5py.File(self.database, "a") as hdf5_file:
                add_spec_library(
                    self.data_folder, hdf5_file, self.spec_library, sptypes
                )

        list_wavelength = []
        list_flux = []
        list_error = []
        list_name = []
        list_simbad = []
        list_sptype = []
        list_parallax = []
        list_spec_res = []

        if object_name is not None:
            if isinstance(object_name, str):
                object_name = [object_name]

            with h5py.File(self.database, "r") as hdf5_file:
                for object_item in object_name:
                    if object_item not in hdf5_file[f"spectra/{self.spec_library}"]:
                        object_list = list(hdf5_file[f"spectra/{self.spec_library}"])

                        raise ValueError(
                            f"The selected 'object_name' (='{object_item}') "
                            "is found in the selected spectral library "
                            f"(='{self.spec_library}'). The following objects "
                            f"are available in the database: {object_list}"
                        )

        with h5py.File(self.database, "r") as hdf5_file:
            for spec_item in hdf5_file[f"spectra/{self.spec_library}"]:
                if object_name is not None and spec_item not in object_name:
                    continue

                dset = hdf5_file[f"spectra/{self.spec_library}/{spec_item}"]

                wavelength = dset[:, 0]  # (um)
                flux = dset[:, 1]  # (W m-2 um-1)
                error = dset[:, 2]  # (W m-2 um-1)

                if exclude_nan:
                    nan_index = np.isnan(flux)

                    wavelength = wavelength[~nan_index]
                    flux = flux[~nan_index]
                    error = error[~nan_index]

                if av_ext is not None:
                    ism_ext = ism_extinction(av_ext, 3.1, wavelength)
                    flux *= 10.0 ** (-0.4 * ism_ext)

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

        spec_box = SpectrumBox()
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

            for spt_item in sptypes:
                for i, spec_item in enumerate(list_sptype):
                    if spt_item == spec_item[:2]:
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
    def get_flux(
        self,
        object_name: Optional[Union[str, List[str]]] = None,
        sptypes: Optional[List[str]] = None,
        av_ext: Optional[float] = None,
    ) -> PhotometryBox:
        """
        Function for calculating the flux density for the filter
        that is set with ``filter_name``.

        Parameters
        ----------
        object_name : str, list(str), None
            Object name of which the spectrum will be selected from the
            spectral library. Either a single object or a list of object
            names. By setting the argument to ``None``, the ``sptype``
            parameter will be used.
        sptypes : list(str), None
            Spectral types to select from a library. The spectral types
            should be indicated with two characters (e.g. 'M5', 'L2',
            'T3'). All spectra are selected if set to ``None``. For
            each object in the ``spec_library``, the requested
            ``sptypes`` are first compared with the optical spectral
            type and, if not available, secondly the near-infrared
            spectral type. To use the ``sptypes`` parameter, the
            argument of ``object_name`` should be set to ``None``
        av_ext : float, None
            Visual extinction, :math:`A_V`, applied to the spectra.
            The extinction is calculated with the empirical relation
            from `Cardelli et al. (1989)
            <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/
            abstract>`_. The extinction is not applied if the argument
            is set to ``None``.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        spec_box = self.get_spectrum(
            object_name=object_name,
            sptypes=sptypes,
            exclude_nan=True,
            av_ext=av_ext,
        )

        n_spectra = len(spec_box.wavelength)

        filter_profile = ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()

        filter_wavel_list = np.full(n_spectra, mean_wavel)
        filter_name_list = np.full(n_spectra, self.filter_name)

        syn_phot = SyntheticPhotometry(filter_name=self.filter_name)

        flux_list = []

        for i in range(n_spectra):
            spec_wavel = spec_box.wavelength[i]
            spec_flux = spec_box.flux[i]
            spec_error = spec_box.error[i]

            if np.sum(np.isnan(spec_error)) == spec_error.size:
                spec_error = None

            phot_flux = syn_phot.spectrum_to_flux(
                wavelength=spec_wavel,
                flux=spec_flux,
                error=spec_error,
            )

            flux_list.append(phot_flux)

        return create_box(
            boxtype="photometry",
            name=spec_box.name,
            sptype=spec_box.sptype,
            wavelength=filter_wavel_list,
            flux=np.array(flux_list),
            app_mag=None,
            abs_mag=None,
            filter_name=filter_name_list,
        )

    @typechecked
    def get_magnitude(
        self,
        object_name: Optional[Union[str, List[str]]] = None,
        sptypes: Optional[List[str]] = None,
        av_ext: Optional[float] = None,
    ) -> PhotometryBox:
        """
        Function for calculating the magnitude for the filter
        that is set with ``filter_name``.

        Parameters
        ----------
        object_name : str, list(str), None
            Object name of which the spectrum will be selected from the
            spectral library. Either a single object or a list of object
            names. By setting the argument to ``None``, the ``sptype``
            parameter will be used.
        sptypes : list(str), None
            Spectral types to select from a library. The spectral types
            should be indicated with two characters (e.g. 'M5', 'L2',
            'T3'). All spectra are selected if set to ``None``. For
            each object in the ``spec_library``, the requested
            ``sptypes`` are first compared with the optical spectral
            type and, if not available, secondly the near-infrared
            spectral type. To use the ``sptypes`` parameter, the
            argument of ``object_name`` should be set to ``None``
        av_ext : float, None
            Visual extinction, :math:`A_V`, applied to the spectra.
            The extinction is calculated with the empirical relation
            from `Cardelli et al. (1989)
            <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/
            abstract>`_. The extinction is not applied if the argument
            is set to ``None``.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        spec_box = self.get_spectrum(
            object_name=object_name,
            sptypes=sptypes,
            exclude_nan=True,
            av_ext=av_ext,
        )

        n_spectra = len(spec_box.wavelength)

        filter_profile = ReadFilter(filter_name=self.filter_name)
        mean_wavel = filter_profile.mean_wavelength()

        filter_wavel_list = np.full(n_spectra, mean_wavel)
        filter_name_list = np.full(n_spectra, self.filter_name)

        syn_phot = SyntheticPhotometry(filter_name=self.filter_name)

        app_mag_list = []
        abs_mag_list = []

        for i in range(n_spectra):
            spec_wavel = spec_box.wavelength[i]
            spec_flux = spec_box.flux[i]
            spec_error = spec_box.error[i]

            if np.sum(np.isnan(spec_error)) == spec_error.size:
                spec_error = None

            if np.isnan(spec_box.parallax[i][0]):
                parallax = None

            else:
                parallax = (
                    float(spec_box.parallax[i][0]),
                    float(spec_box.parallax[i][1]),
                )

            app_mag, abs_mag = syn_phot.spectrum_to_magnitude(
                spec_wavel,
                spec_flux,
                error=spec_error,
                parallax=parallax,
            )

            app_mag_list.append(app_mag)
            abs_mag_list.append(abs_mag)

        return create_box(
            boxtype="photometry",
            name=spec_box.name,
            sptype=spec_box.sptype,
            wavelength=filter_wavel_list,
            flux=None,
            app_mag=np.asarray(app_mag_list),
            abs_mag=np.asarray(abs_mag_list),
            filter_name=filter_name_list,
        )
