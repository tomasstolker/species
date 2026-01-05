"""
Module with functionalities for reading and writing of data.
"""

import json
import os
import warnings

from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import astropy.units as u
import h5py
import numpy as np
import pooch

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units.quantity import Quantity
from scipy.integrate import simpson
from tqdm.auto import tqdm
from typeguard import typechecked

from species.core import constants
from species.core.box import ObjectBox, ModelBox, SamplesBox, SpectrumBox, create_box
from species.util.core_util import print_section


class Database:
    """
    Class with reading and writing functionalities for the HDF5 database.
    """

    @typechecked
    def __init__(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.data_folder = config["species"]["data_folder"]

    @typechecked
    def list_content(self) -> None:
        """
        Function for listing the content of the HDF5 database. The
        database structure will be descended while printing the paths
        of all the groups and datasets, as well as the dataset
        attributes.

        Returns
        -------
        NoneType
            None
        """

        print_section("List database content")

        @typechecked
        def _descend(
            h5_object: Union[
                h5py._hl.files.File, h5py._hl.group.Group, h5py._hl.dataset.Dataset
            ],
            seperator: str = "",
        ) -> None:
            """
            Internal function for descending into an HDF5
            dataset and printing its content.

            Parameters
            ----------
            h5_object : h5py._hl.files.File, h5py._hl.group.Group, h5py._hl.dataset.Dataset
                The ``h5py`` object.
            separator : str
                Separator that is used between items.

            Returns
            -------
            NoneType
                None
            """

            if isinstance(h5_object, (h5py._hl.files.File, h5py._hl.group.Group)):
                for group_key in h5_object.keys():
                    print(
                        seperator + "- " + group_key + ": " + str(h5_object[group_key])
                    )

                    _descend(h5_object[group_key], seperator=seperator + "\t")

                    for attr_key in h5_object[group_key].attrs.keys():
                        print(
                            seperator
                            + "\t"
                            + "- "
                            + attr_key
                            + " = "
                            + str(h5_object[group_key].attrs[attr_key])
                        )

            elif isinstance(h5_object, h5py._hl.dataset.Dataset):
                for attr_key in h5_object.attrs.keys():
                    print(
                        seperator
                        + "- "
                        + attr_key
                        + ": "
                        + str(h5_object.attrs[attr_key])
                    )

        with h5py.File(self.database, "r") as hdf_file:
            _descend(hdf_file)

    @typechecked
    def list_companions(self, verbose: bool = False) -> List[str]:
        """
        Function for printing an overview of the companion data that
        are stored in the database. It will return a list with all
        the companion names. Each name can be used as input for
        :class:`~species.read.read_object.ReadObject`.

        Parameters
        ----------
        verbose : bool
            Print details on the companion data or list only the names
            of the companions for which data are available.

        Returns
        -------
        list(str)
            List with the object names that are stored in the database.
        """

        data_file = (
            Path(__file__).parent.resolve() / "companion_data/companion_data.json"
        )

        with open(data_file, "r", encoding="utf-8") as json_file:
            comp_data = json.load(json_file)

        spec_file = (
            Path(__file__).parent.resolve() / "companion_data/companion_spectra.json"
        )

        with open(spec_file, "r", encoding="utf-8") as json_file:
            comp_spec = json.load(json_file)

        print_section("Companions with available data")

        comp_names = []

        for planet_name, planet_dict in comp_data.items():
            comp_names.append(planet_name)

            if verbose:
                print(f"Object name = {planet_name}")

                if "parallax" in planet_dict:
                    parallax = planet_dict["parallax"]
                    print(f"Parallax (pc) = {parallax[0]} +/- {parallax[1]}")

                if "distance" in planet_dict:
                    distance = planet_dict["distance"]
                    print(f"Distance (pc) = {distance[0]} +/- {distance[1]}")

                app_mag = planet_dict["app_mag"]

                for mag_key, mag_value in app_mag.items():
                    if isinstance(mag_value[0], list) or isinstance(
                        mag_value[0], tuple
                    ):
                        for item in mag_value:
                            print(f"{mag_key} (mag) = {item[0]} +/- {item[1]}")
                    else:
                        print(f"{mag_key} (mag) = {mag_value[0]} +/- {mag_value[1]}")

                print("References:")
                for ref_item in planet_dict["references"]:
                    print(f"   - {ref_item}")

                if planet_name in comp_spec:
                    for key, value in comp_spec[planet_name].items():
                        print(f"{key} spectrum from {value[3]}")

                print()

            else:
                print(planet_name)

        return comp_names

    @typechecked
    def available_models(self, verbose: bool = True) -> Dict:
        """
        Function for printing an overview of the available model grids
        that can be downloaded and added to the database with
        :class:`~species.data.database.Database.add_model`.

        Parameters
        ----------
        verbose : bool
            Print details on the grids of model spectra or list only
            the names of the available model spectra.

        Returns
        -------
        dict
            Dictionary with the details on the model grids. The
            dictionary is created from the ``model_data.json``
            file in the ``species.data`` folder.
        """

        print_section("Available model spectra")

        data_file = Path(__file__).parent.resolve() / "model_data/model_data.json"

        with open(data_file, "r", encoding="utf-8") as json_file:
            model_data = json.load(json_file)

        for model_name, model_dict in model_data.items():
            if verbose:
                print(f"   - {model_dict['name']}:")
                print(f"      - Label = {model_name}")

                if "parameters" in model_dict:
                    print(f"      - Model parameters: {model_dict['parameters']}")

                if "teff range" in model_dict:
                    print(f"      - Teff range (K): {model_dict['teff range']}")

                if "wavelength range" in model_dict:
                    print(
                        f"      - Wavelength range (um): {model_dict['wavelength range']}"
                    )

                if "lambda/d_lambda" in model_dict:
                    print(
                        f"      - Sampling (lambda/d_lambda): {model_dict['lambda/d_lambda']}"
                    )

                if "information" in model_dict:
                    print(f"      - Extra details: {model_dict['information']}")

                if "file size" in model_dict:
                    print(f"      - File size: {model_dict['file size']}")

                if "reference" in model_dict:
                    print(f"      - Reference: {model_dict['reference']}")

                if "url" in model_dict:
                    print(f"      - URL: {model_dict['url']}")

                print()

            else:
                print(f"{model_dict['name']} -> label: {model_name}")

        return model_data

    @typechecked
    def delete_data(self, data_set: str) -> None:
        """
        Function for deleting a dataset from the HDF5 database.

        Parameters
        ----------
        data_set : str
            Group or dataset path in the HDF5 database. The content
            and structure of the database can be shown with
            :func:`~species.data.database.Database.list_content`. That
            could help to determine which argument should be provided
            as argument of ``data_set``. For example,
            ``data_set="models/drift-phoenix"`` will remove the
            model spectra of DRIFT-PHOENIX.

        Returns
        -------
        NoneType
            None
        """

        with h5py.File(self.database, "a") as hdf5_file:
            if data_set in hdf5_file:
                print(f"Deleting data: {data_set}...", end="", flush=True)
                del hdf5_file[data_set]
                print(" [DONE]")

            else:
                warnings.warn(
                    f"The dataset {data_set} is not found in {self.database}."
                )

    @typechecked
    def add_companion(
        self,
        name: Optional[Union[Optional[str], Optional[List[str]]]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Function for adding the magnitudes and spectra of
        directly imaged planets and brown dwarfs from
        `data/companion_data/companion_data.json` and
        :func:`~species.data.companion_data.companion_spectra` to the database.

        Parameters
        ----------
        name : str, list(str), None
            Name or list with names of the directly imaged planets
            and brown dwarfs (e.g. ``'HR 8799 b'`` or ``['HR 8799 b',
            '51 Eri b', 'PZ Tel B']``). All the available companion
            data are added if the argument is set to ``None``.
        verbose : bool
            Print details on the companion data that are
            added to the database.

        Returns
        -------
        NoneType
            None
        """

        from species.data.companion_data.companion_spectra import companion_spectra

        if isinstance(name, str):
            name = [
                name,
            ]

        data_file = (
            Path(__file__).parent.resolve() / "companion_data/companion_data.json"
        )

        with open(data_file, "r", encoding="utf-8") as json_file:
            comp_data = json.load(json_file)

        if name is None:
            name = list(comp_data.keys())

        if not verbose:
            print(f"Add companion: {name}")

        for item in name:
            spec_dict = companion_spectra(Path(self.data_folder), item, verbose=verbose)

            parallax = None

            # try:
            #     # Query SIMBAD to get the parallax
            #     simbad = Simbad()
            #     simbad.add_votable_fields("parallax")
            #     simbad_result = simbad.query_object(comp_data[item]["simbad"])
            #
            #     if simbad_result is not None:
            #         par_sim = (
            #             simbad_result["PLX_VALUE"][0],  # (mas)
            #             simbad_result["PLX_ERROR"][0],
            #         )  # (mas)
            #
            #         if not np.ma.is_masked(par_sim[0]) and not np.ma.is_masked(
            #             par_sim[1]
            #         ):
            #             parallax = (float(par_sim[0]), float(par_sim[1]))
            #
            # except urllib.error.URLError:
            #     parallax = tuple(comp_data[item]["parallax"])

            if parallax is None:
                parallax = tuple(comp_data[item]["parallax"])

            if "app_mag" in comp_data[item]:
                app_mag = comp_data[item]["app_mag"]

                for key, value in app_mag.items():
                    if isinstance(value[0], list):
                        mag_list = []
                        for mag_item in value:
                            mag_list.append(tuple(mag_item))

                        app_mag[key] = mag_list

                    else:
                        app_mag[key] = tuple(value)

            else:
                app_mag = None

            self.add_object(
                object_name=item,
                parallax=parallax,
                app_mag=app_mag,
                spectrum=spec_dict,
                verbose=verbose,
            )

    @typechecked
    def add_dust(self) -> None:
        """
        Function for adding optical constants of MgSiO3 and Fe, and
        MgSiO3 cross sections for a log-normal and power-law size
        distribution to the database. The optical constants have
        been compiled by Mollière et al. (2019) for petitRADTRANS
        from the following sources:

        - MgSiO3, crystalline
            - Scott & Duley (1996), ApJS, 105, 401
            - Jäger et al. (1998), A&A, 339, 904

        - MgSiO3, amorphous
            - Jäger et al. (2003), A&A, 408, 193

        - Fe, crystalline
            - Henning & Stognienko (1996), A&A, 311, 291

        - Fe, amorphous
            - Pollack et al. (1994), ApJ, 421, 615

        Returns
        -------
        NoneType
            None
        """

        from species.data.misc_data.dust_data import (
            add_cross_sections,
            add_optical_constants,
        )

        with h5py.File(self.database, "a") as hdf5_file:
            if "dust" in hdf5_file:
                del hdf5_file["dust"]

            add_optical_constants(self.data_folder, hdf5_file)
            add_cross_sections(self.data_folder, hdf5_file)

    @typechecked
    def add_accretion(self) -> None:
        """
        Function for adding the coefficients for converting line
        luminosities of hydrogen emission lines into accretion
        luminosities (see `Aoyama et al. (2021) <https://ui.
        adsabs.harvard.edu/abs/ 2021ApJ...917L..30A/abstract>`_
        and `Marleau & Aoyama (2022) <https://ui.adsabs.harvard.
        edu/abs/2022RNAAS...6..262M/abstract>`_ for details).
        The relation is used by
        :class:`~species.fit.emission_line.EmissionLine`
        for converting the fitted line luminosity.

        Returns
        -------
        NoneType
            None
        """

        from species.data.misc_data.accretion_data import add_accretion_relation

        with h5py.File(self.database, "a") as hdf5_file:
            if "accretion" in hdf5_file:
                del hdf5_file["accretion"]

            add_accretion_relation(self.data_folder, hdf5_file)

    @typechecked
    def add_filter(
        self,
        filter_name: str,
        filename: Optional[str] = None,
        detector_type: str = "photon",
        verbose: bool = True,
    ) -> None:
        """
        Function for adding a filter profile to the database, either
        from the SVO Filter profile Service or from an input file.
        Additional filters that are automatically added are
        Magellan/VisAO.rp, Magellan/VisAO.ip, Magellan/VisAO.zp,
        Magellan/VisAO.Ys, ALMA/band6, and ALMA/band7.

        Parameters
        ----------
        filter_name : str
            Filter name from the SVO Filter Profile Service (e.g.,
            'Paranal/NACO.Lp') or a user-defined name if a ``filename``
            is specified.
        filename : str, None
            Filename of the filter profile. The first column should
            contain the wavelength (um) and the second column the
            fractional transmission. The profile is downloaded from
            the SVO Filter Profile Service if the argument of
            ``filename`` is set to ``None``.
        detector_type : str
            The detector type ('photon' or 'energy'). The argument is
            only used if a ``filename`` is provided. Otherwise, for
            filters that are fetched from the SVO website, the detector
            type is read from the SVO data. The detector type determines
            if a wavelength factor is included in the integral for the
            synthetic photometry.
        verbose : bool
            Print details on the companion data that are added
            to the database.

        Returns
        -------
        NoneType
            None
        """

        if verbose:
            print(f"Adding filter: {filter_name}...", end="", flush=True)

        # filter_split = filter_name.split("/")

        if filename is not None:
            filter_profile = np.loadtxt(filename)

            wavelength = filter_profile[:, 0]
            transmission = filter_profile[:, 1]

            with h5py.File(self.database, "a") as hdf5_file:
                if f"filters/{filter_name}" in hdf5_file:
                    del hdf5_file[f"filters/{filter_name}"]

                dset = hdf5_file.create_dataset(
                    f"filters/{filter_name}",
                    data=np.column_stack((wavelength, transmission)),
                    dtype="f8",
                )

                dset.attrs["det_type"] = str(detector_type)

        else:
            from species.data.filter_data.filter_data import add_filter_profile

            with h5py.File(self.database, "a") as hdf5_file:
                if f"filters/{filter_name}" in hdf5_file:
                    del hdf5_file[f"filters/{filter_name}"]

                # if f"filters/{filter_split[0]}" not in hdf5_file:
                #     hdf5_file.create_group(f"filters/{filter_split[0]}")

                add_filter_profile(self.data_folder, hdf5_file, filter_name)

        if verbose:
            print(" [DONE]")

    @typechecked
    def add_isochrones(
        self,
        model: Optional[str] = None,
        filename: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        """
        Function for adding isochrone data to the database.

        Parameters
        ----------
        model : str, None
            Evolutionary model ('ames', 'atmo', 'atmo-chabrier2023',
            'baraffe2015', 'bt-settl', linder2019', 'marleau2014',
            'nextgen', 'parsec', 'saumon2008', 'sonora-bobcat',
            'sonora-diamondback', 'spiegel2012'). Isochrones will
            be automatically downloaded. Alternatively, the
            ``filename`` parameter can be used in combination
            with ``tag``.
        filename : str, None
            Filename with the isochrone data. The argument of
            ``model`` will be ignored by setting the argument
            of ``filename``. When using ``filename``, also
            the argument of ``tag`` should be set. Only files
            with isochrone data from
            https://phoenix.ens-lyon.fr/Grids/ and
            https://perso.ens-lyon.fr/isabelle.baraffe/ are
            supported. The parameter is ignored by setting
            the argument to ``None``.
        tag : str, None
            Database tag name where the isochrone that will be
            stored. Setting the argument is only required in
            combination with the ``filename`` parameter.
            Otherwise, the argument can be set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        from species.data.isochrone_data.add_isochrone import add_isochrone_grid

        print_section("Add isochrone grid")

        print(f"Evolutionary model: {model}")
        print(f"File name: {filename}")
        print(f"Database tag: {tag}")

        avail_models = [
            "ames",
            "atmo",
            "atmo-chabrier2023",
            "baraffe2015",
            "bt-settl",
            "linder2019",
            "marleau2014",
            "nextgen",
            "parsec",
            "saumon2008",
            "sonora-bobcat",
            "sonora-diamondback",
            "spiegel2012",
        ]

        if filename is None and model not in avail_models:
            raise ValueError(
                f"The selected 'model={model}' is not supported. Please "
                f"select one of the following models: {avail_models}"
            )

        with h5py.File(self.database, "a") as hdf5_file:

            if model == "ames":
                if "isochrones/ames-cond" in hdf5_file:
                    del hdf5_file["isochrones/ames-cond"]
                if "isochrones/ames-dusty" in hdf5_file:
                    del hdf5_file["isochrones/ames-dusty"]

            elif model == "atmo":
                if "isochrones/atmo-ceq" in hdf5_file:
                    del hdf5_file["isochrones/atmo-ceq"]
                if "isochrones/atmo-neq-weak" in hdf5_file:
                    del hdf5_file["isochrones/atmo-neq-weak"]
                if "isochrones/atmo-neq-strong" in hdf5_file:
                    del hdf5_file["isochrones/atmo-neq-strong"]

            elif model == "baraffe2015":
                if "isochrones/baraffe2015" in hdf5_file:
                    del hdf5_file["isochrones/baraffe2015"]

            elif model == "bt-settl":
                if "isochrones/bt-settl" in hdf5_file:
                    del hdf5_file["isochrones/bt-settl"]

            elif model == "atmo-chabrier2023":
                if "isochrones/atmo-ceq-chabrier2023" in hdf5_file:
                    del hdf5_file["isochrones/atmo-ceq-chabrier2023"]
                if "isochrones/atmo-neq-weak-chabrier2023" in hdf5_file:
                    del hdf5_file["isochrones/atmo-neq-weak-chabrier2023"]
                if "isochrones/atmo-neq-strong-chabrier2023" in hdf5_file:
                    del hdf5_file["isochrones/atmo-neq-strong-chabrier2023"]

            elif model == "linder2019":
                if "isochrones" in hdf5_file:
                    for iso_item in list(hdf5_file["isochrones"]):
                        if iso_item[:10] == "linder2019":
                            del hdf5_file[f"isochrones/{iso_item}"]

            elif model == "marleau2014":
                if "isochrones/marleau2014" in hdf5_file:
                    del hdf5_file["isochrones/marleau2014"]

            elif model == "nextgen":
                if "isochrones/nextgen" in hdf5_file:
                    del hdf5_file["isochrones/nextgen"]

            elif model == "parsec":
                if "isochrones/parsec" in hdf5_file:
                    del hdf5_file["isochrones/parsec"]

            elif model == "saumon2008":
                if "isochrones/saumon2008-nc_solar" in hdf5_file:
                    del hdf5_file["isochrones/saumon2008-nc_solar"]
                if "isochrones/saumon2008-nc_-0.3" in hdf5_file:
                    del hdf5_file["isochrones/saumon2008-nc_-0.3"]
                if "isochrones/saumon2008-nc_+0.3" in hdf5_file:
                    del hdf5_file["isochrones/saumon2008-nc_+0.3"]
                if "isochrones/saumon2008-f2_solar" in hdf5_file:
                    del hdf5_file["isochrones/saumon2008-f2_solar"]
                if "isochrones/saumon2008-hybrid_solar" in hdf5_file:
                    del hdf5_file["isochrones/saumon2008-hybrid_solar"]

            elif model == "sonora-bobcat":
                if "isochrones/sonora-bobcat+0.0" in hdf5_file:
                    del hdf5_file["isochrones/sonora-bobcat+0.0"]
                if "isochrones/sonora-bobcat+0.5" in hdf5_file:
                    del hdf5_file["isochrones/sonora-bobcat+0.5"]
                if "isochrones/sonora-bobcat-0.5" in hdf5_file:
                    del hdf5_file["isochrones/sonora-bobcat-0.5"]

            elif model == "sonora-diamondback":
                iso_tags = [
                    "nc-0.5",
                    "nc+0.0",
                    "nc+0.5",
                    "hybrid-0.5",
                    "hybrid+0.0",
                    "hybrid+0.5",
                    "hybrid-grav-0.5",
                    "hybrid-grav+0.0",
                    "hybrid-grav+0.5",
                ]

                for iso_item in iso_tags:
                    if f"isochrones/sonora-diamondback-{iso_item}" in hdf5_file:
                        del hdf5_file[f"isochrones/sonora-diamondback-{iso_item}"]

            elif model == "spiegel2012":
                if "isochrones/spiegel2012-hybrid+0.0" in hdf5_file:
                    del hdf5_file["isochrones/spiegel2012-hybrid+0.0"]

                if "isochrones/spiegel2012-hybrid+0.5" in hdf5_file:
                    del hdf5_file["isochrones/spiegel2012-hybrid+0.5"]

                if "isochrones/spiegel2012-cloudfree+0.0" in hdf5_file:
                    del hdf5_file["isochrones/spiegel2012-cloudfree+0.0"]

                if "isochrones/spiegel2012-cloudfree+0.5" in hdf5_file:
                    del hdf5_file["isochrones/spiegel2012-cloudfree+0.5"]

            else:
                if f"isochrones/{tag}" in hdf5_file:
                    del hdf5_file[f"isochrones/{tag}"]

            add_isochrone_grid(
                self.data_folder, hdf5_file, model, filename=filename, tag=tag
            )

    @typechecked
    def add_model(
        self,
        model: str,
        wavel_range: Optional[Tuple[float, float]] = None,
        wavel_sampling: Optional[float] = None,
        teff_range: Optional[Tuple[float, float]] = None,
        unpack_tar: bool = True,
        fit_from: Optional[float] = None,
        extend_from: Optional[float] = None,
    ) -> None:
        """
        Function for adding a grid of model spectra to the database.
        All spectra have been resampled to logarithmically-spaced
        wavelengths (see
        :func:`~species.data.database.Database.available_models`),
        typically at the order of several thousand. It should be
        noted that the original spectra were typically calculated
        with a constant step size in wavenumber, so the original
        wavelength sampling decreased from short to long wavelengths.
        When fitting medium/high- resolution spectra, it is best to
        carefully check the result to determine if the sampling of
        the input grid was sufficient for modeling the spectra at
        the considered wavelength regime. See also
        :func:`~species.data.database.Database.add_custom_model`
        for adding a custom grid to the database. Most grids
        provide spectra up to the near- or mid-infrared regime.
        To extend the spectra towards longer wavelengths, it is
        possible to use the ``fit_from`` and ``extend_from``
        parameters to extend the spectra with a Rayleigh-Jeans slope.

        Parameters
        ----------
        model : str
            Model name ('ames-cond', 'ames-dusty', 'atmo-ceq',
            'atmo-neq-weak', 'atmo-neq-strong', 'bt-settl',
            'bt-settl-cifist', 'bt-nextgen', 'drift-phoenix',
            'petitcode-cool-clear', 'petitcode-cool-cloudy',
            'petitcode-hot-clear', 'petitcode-hot-cloudy', 'exo-rem',
            'blackbody', bt-cond', 'bt-cond-feh, 'morley-2012',
            'sonora-cholla', 'sonora-bobcat', 'sonora-bobcat-co',
            'koester-wd', 'saumon2008-clear', 'saumon2008-cloudy',
            'petrus2023', 'sphinx').
        wavel_range : tuple(float, float), None
            Wavelength range (:math:`\\mu\\text{m}`) that will be
            stored in the database. The full wavelength range is
            used if the argument is set to ``None``.
        wavel_sampling : float, None
            Wavelength spacing :math:`\\lambda/\\Delta\\lambda` to which
            the spectra will be resampled. Typically this parameter is
            not needed so the argument can be set to ``None``. The only
            benefit of using this parameter is limiting the storage
            in the HDF5 database. The parameter should be used in
            combination with setting the ``wavel_range``.
        teff_range : tuple(float, float), None
            Range of effective temperatures (K) of which the spectra
            are extracted from the TAR file and added to the HDF5
            database. The full grid of spectra will be added if the
            argument is set to ``None``.
        unpack_tar : bool
            Unpack the TAR file with the model spectra in the
            ``data_folder``. By default, the argument is set to
            ``True`` such the TAR file with the model spectra
            will be unpacked after downloading. In case the TAR
            file had already been unpacked previously, the
            argument can be set to ``False`` such that the
            unpacking will be skipped. This can save some time
            with unpacking large TAR files.
        fit_from : float, None
            Extend the spectra with a Rayleigh-Jeans slope. To do so,
            the red end of the spectra will be fitted by setting
            ``fit_from`` to the minimum wavelength (in um) beyond
            which fluxes will be included in the least-squares fit.
            The spectra are not extended when setting the
            argument to ``None``.
        extend_from : float, None
            This parameter can be used in combination with
            ``fit_from``. The argument of ``extend_from`` is the
            minimum wavelength (in um) from which the spectra
            will be extended with the Rayleigh-Jeans slope.
            The spectra will be extended from the last available
            wavelength when setting the argument to ``None``.
            Typically, the value of ``fit_from`` will be smaller
            than the value of ``extend_from``.

        Returns
        -------
        NoneType
            None
        """

        from species.data.model_data.model_spectra import add_model_grid

        with h5py.File(self.database, "a") as hdf5_file:
            add_model_grid(
                model_tag=model,
                input_path=self.data_folder,
                database=hdf5_file,
                wavel_range=wavel_range,
                teff_range=teff_range,
                wavel_sampling=wavel_sampling,
                unpack_tar=unpack_tar,
                fit_from=fit_from,
                extend_from=extend_from,
            )

    @typechecked
    def add_custom_model(
        self,
        model: str,
        data_path: Union[str, Path],
        parameters: List[str],
        wavel_range: Optional[Tuple[float, float]] = None,
        wavel_sampling: Optional[float] = None,
        teff_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Function for adding a custom grid of model spectra to the
        database. The spectra are read from the ``data_path`` and
        should contain the ``model_name`` and ``parameters`` in
        the filenames in the following format example:
        `model-name_teff_1000_logg_4.0_feh_0.0_spec.dat` or
        `model-name_teff_1000_logg_4.0_feh_0.0_spec.npy`. The
        list with ``parameters`` should contain the same parameters
        as are included in the filenames. Each datafile should
        contain two columns with the wavelengths in
        :math:`\\mu\\text{m}` and the fluxes in
        :math:`\\text{W} \\text{m}^{-2} \\mu\\text{m}^{-1}`. Each file
        should contain the same number and values of wavelengths. The
        wavelengths should be logarithmically spaced, so at a constant
        :math:`\\lambda/\\Delta\\lambda`. If not, then the
        ``wavel_range`` and ``wavel_sampling`` parameters should be
        used such that the wavelengths are resampled.

        Parameters
        ----------
        model : str
            Name of the model grid. Should be identical to the model
            name that is used in the filenames.
        data_path : str, Path
            Path where the files with the model spectra are located.
            Either a relative or absolute path. Either a string or
            a ``Path`` object from ``pathlib``. The model spectra
            can be stored as NumPy binary files (.npy) or plain
            text files.
        parameters : list(str)
            List with the model parameters. The following parameters
            are supported: ``teff`` (for :math:`T_\\mathrm{eff}`),
            ``logg`` (for :math:`\\log\\,g`), ``feh`` (for [Fe/H]),
            ``co`` (for C/O), ``fsed`` (for :math:`f_\\mathrm{sed}`),
            ``logkzz`` (for :math:`\\log\\,K_\\mathrm{zz}`),
            ``adindex`` (for :math:`\\gamma_\\mathrm{ad}`), and
            ``log_co_iso`` (for
            :math:`\\log\\,^{12}\\mathrm{CO}/^{13}\\mathrm{CO}`).
            Please contact the code maintainer if support for
            other parameters should be added.
        wavel_range : tuple(float, float), None
            Wavelength range (:math:`\\mu\\text{m}`) that will be
            stored in the database. The full wavelength range is used
            if the argument is set to ``None``.
        wavel_sampling : float, None
            Wavelength spacing :math:`\\lambda/\\Delta\\lambda` to which
            the spectra will be resampled. Typically this parameter is
            not needed so the argument can be set to ``None``. The only
            benefit of using this parameter is limiting the storage
            in the HDF5 database. The parameter should be used in
            combination with setting the ``wavel_range``.
        teff_range : tuple(float, float), None
            Effective temperature range (K) for adding a subset of the
            model grid. The full parameter grid will be added if the
            argument is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        from species.data.model_data.custom_model import add_custom_model_grid

        with h5py.File(self.database, "a") as hdf5_file:
            add_custom_model_grid(
                model,
                data_path,
                parameters,
                hdf5_file,
                wavel_range,
                teff_range,
                wavel_sampling,
            )

    @typechecked
    def add_object(
        self,
        object_name: str,
        parallax: Optional[Tuple[float, float]] = None,
        app_mag: Optional[
            Dict[str, Union[Tuple[float, float], List[Tuple[float, float]]]]
        ] = None,
        flux_density: Optional[Dict[str, Tuple[float, float]]] = None,
        spectrum: Optional[
            Dict[
                str,
                Tuple[
                    Union[str, np.ndarray],
                    Optional[Union[str, np.ndarray]],
                    Optional[float],
                ],
            ]
        ] = None,
        deredden: Optional[Union[Dict[str, float], float]] = None,
        verbose: bool = True,
        units: Optional[Dict[str, Union[str, Tuple[str, str]]]] = None,
    ) -> None:
        """
        Function for adding the photometry and/or spectra
        of an object to the database.

        Parameters
        ----------
        object_name : str
            Object name that will be used as label in the database.
        parallax : tuple(float, float), None
            Parallax and uncertainty (mas). Not stored if the argument
            is set to ``None``.
        app_mag : dict, None
            Dictionary with the filter names, apparent magnitudes, and
            uncertainties. For example, ``{'Paranal/NACO.Lp': (15.,
            0.2), 'Paranal/NACO.Mp': (13., 0.3)}``. For the use of
            duplicate filter names, the magnitudes have to be provided
            in a list, for example ``{'Paranal/NACO.Lp': [(15., 0.2),
            (14.5, 0.5)], 'Paranal/NACO.Mp': (13., 0.3)}``. No
            photometry is stored if the argument is set to ``None``.
        flux_density : dict, None
            Dictionary with filter names, flux densities (W m-2 um-1),
            and uncertainties (W m-1 um-1), or setting the ``units``
            parameter when other flux units are used. For example,
            ``{'Paranal/NACO.Lp': (1e-15, 1e-16)}``. Currently, the use
            of duplicate filters is not implemented. The use of
            ``app_mag`` is preferred over ``flux_density`` because with
            ``flux_density`` only fluxes are stored while with
            ``app_mag`` both magnitudes and fluxes. However,
            ``flux_density`` can be used in case the magnitudes and/or
            filter profiles are not available. In that case, the fluxes
            can still be selected with ``inc_phot`` in
            :class:`~species.fit.fit_model.FitModel`. The
            ``flux_density`` can also be used for an upper limit on a
            photometric flux, by setting the flux to zero and the
            uncertainty to the 1:math:`\\sigma` upper limit. The
            argument of ``flux_density`` is ignored if set to ``None``.
        spectrum : dict, None
            Dictionary with the spectrum, optional covariance matrix,
            and resolving power of the instrument. The input data
            can either be a FITS or ASCII file, or as NumPy array
            directly. The spectra should have 3 columns, so the array
            shape should be (n_wavelengths, 3), with wavelength (um),
            flux (W m-2 um-1), and uncertainty (W m-2 um-1). The
            ``units`` parameter can be set for reading in data with
            different wavelength and/or flux units. The covariance
            matrix should be 2D with the same number of wavelength
            points as the spectrum, so the shape is (n_wavelengths,
            n_wavelengths). Like the spectrum, the covariance matrix
            can be provided as FITS or ASCII file, or as NumPy array
            directly. For example, when providing the input as files,
            ``{'SPHERE': ('spectrum.dat', 'covariance.fits', 50.)}``.
            No covariance data is stored if set to ``None``, for
            example, ``{'SPHERE': ('spectrum.dat', None, 50.)}``. The
            ``spectrum`` parameter is ignored if set to ``None``.
            For VLTI/GRAVITY data, the same FITS file can be provided
            as spectrum and covariance matrix.
        deredden : dict, float, None
            Dictionary with ``spectrum`` and ``app_mag`` names that
            will be dereddened with the provided :math:`A_V`. For
            example, ``deredden={'SPHERE': 1.5, 'Keck/NIRC2.J': 1.5}``
            will deredden the provided spectrum named 'SPHERE' and
            the Keck/NIRC2 J-band photometry with a visual extinction
            of 1.5. For photometric fluxes, the filter-averaged
            extinction is used for the dereddening.
        verbose : bool
            Print details on the object data that are added to
            the database.
        units : dict, None
            Dictionary with the units of the data provided with
            ``flux_density`` and ``spectrum``. Only required if
            the wavelength units are not :math:`\\mu\\text{m}`
            and/or the flux units are not provided as
            :math:`\\text{W} \\text{m}^{-2} \\mu\\text{m}^{-1}`.
            Otherwise, the argument of ``units`` can be set to
            ``None`` such that it will be ignored. The dictionary
            keys should be the filter names as provided with
            ``flux_density`` and the spectrum names as provided
            with ``spectrum``. Supported units can be found in
            the docstring of
            :func:`~species.util.data_util.convert_units`.

        Returns
        -------
        NoneType
            None
        """

        from species.read.read_filter import ReadFilter

        if verbose:
            print_section("Add object")
            print(f"Object name: {object_name}")
            print(f"Units: {units}")
            print(f"Deredden: {deredden}")

        # Set default units

        if units is None:
            units = {}

        # First add filters here because ReadFilter
        # will also open the HDF5 database

        if app_mag is not None:
            for mag_item in app_mag:
                read_filt = ReadFilter(mag_item)

        if flux_density is not None:
            for flux_item in flux_density:
                read_filt = ReadFilter(flux_item)

        if deredden is None:
            deredden = {}

        if app_mag is not None:
            from species.data.spec_data.spec_vega import add_vega

            with h5py.File(self.database, "a") as hdf5_file:
                if "spectra/calibration/vega" not in hdf5_file:
                    add_vega(self.data_folder, hdf5_file)

                for mag_item in app_mag:
                    if f"filters/{mag_item}" not in hdf5_file:
                        self.add_filter(mag_item, verbose=verbose)

        if flux_density is not None:
            from species.data.spec_data.spec_vega import add_vega

            with h5py.File(self.database, "a") as hdf5_file:
                if "spectra/calibration/vega" not in hdf5_file:
                    add_vega(self.data_folder, hdf5_file)

                for flux_item in flux_density:
                    if f"filters/{flux_item}" not in hdf5_file:
                        self.add_filter(flux_item, verbose=verbose)

        hdf5_file = h5py.File(self.database, "a")

        if "objects" not in hdf5_file:
            hdf5_file.create_group("objects")

        if f"objects/{object_name}" not in hdf5_file:
            hdf5_file.create_group(f"objects/{object_name}")

        if parallax is not None:
            if verbose:
                print(f"Parallax (mas) = {parallax[0]:.2f} +/- {parallax[1]:.2f}")

            if f"objects/{object_name}/parallax" in hdf5_file:
                del hdf5_file[f"objects/{object_name}/parallax"]

            hdf5_file.create_dataset(
                f"objects/{object_name}/parallax", data=parallax, dtype="f8"
            )  # (mas)

        flux = {}
        error = {}
        dered_phot = {}
        mean_wavel = {}

        if app_mag is not None:
            if verbose:
                print("\nMagnitudes:")

            for mag_item in app_mag:
                read_filt = ReadFilter(mag_item)
                mean_wavel[mag_item] = read_filt.mean_wavelength()

                if isinstance(deredden, float) or mag_item in deredden:
                    from species.util.dust_util import ism_extinction

                    filter_profile = read_filt.get_filter()

                    if isinstance(deredden, float):
                        ext_mag = ism_extinction(deredden, 3.1, filter_profile[:, 0])

                    else:
                        ext_mag = ism_extinction(
                            deredden[mag_item], 3.1, filter_profile[:, 0]
                        )

                    from species.phot.syn_phot import SyntheticPhotometry

                    synphot = SyntheticPhotometry(mag_item)

                    dered_phot[mag_item], _ = synphot.spectrum_to_flux(
                        filter_profile[:, 0], 10.0 ** (0.4 * ext_mag)
                    )

                else:
                    dered_phot[mag_item] = 1.0

                if isinstance(app_mag[mag_item], tuple):
                    try:
                        from species.phot.syn_phot import SyntheticPhotometry

                        synphot = SyntheticPhotometry(mag_item)

                        flux[mag_item], error[mag_item] = synphot.magnitude_to_flux(
                            app_mag[mag_item][0], app_mag[mag_item][1]
                        )

                        flux[mag_item] *= dered_phot[mag_item]

                    except KeyError:
                        warnings.warn(
                            f"Filter '{mag_item}' is not available on the SVO Filter "
                            f"Profile Service so a flux calibration cannot be done. "
                            f"Please add the filter manually with the 'add_filter' "
                            f"function. For now, only the '{mag_item}' magnitude of "
                            f"'{object_name}' is stored."
                        )

                        # Write NaNs if the filter is not available
                        flux[mag_item], error[mag_item] = np.nan, np.nan

                elif isinstance(app_mag[mag_item], list):
                    flux_list = []
                    error_list = []

                    for i, dupl_item in enumerate(app_mag[mag_item]):
                        try:
                            from species.phot.syn_phot import SyntheticPhotometry

                            synphot = SyntheticPhotometry(mag_item)

                            flux_dupl, error_dupl = synphot.magnitude_to_flux(
                                dupl_item[0], dupl_item[1]
                            )

                            flux_dupl *= dered_phot[mag_item]

                        except KeyError:
                            warnings.warn(
                                f"Filter '{mag_item}' is not available on the SVO "
                                f"Filter Profile Service so a flux calibration cannot "
                                f"be done. Please add the filter manually with the "
                                f"'add_filter' function. For now, only the "
                                f"'{mag_item}' magnitude of '{object_name}' is "
                                f"stored."
                            )

                            # Write NaNs if the filter is not available
                            flux_dupl, error_dupl = np.nan, np.nan

                        flux_list.append(flux_dupl)
                        error_list.append(error_dupl)

                    flux[mag_item] = flux_list
                    error[mag_item] = error_list

                else:
                    raise ValueError(
                        "The values in the dictionary with magnitudes "
                        "should be tuples or a list with tuples (in "
                        "case duplicate filter names are required)."
                    )

            for mag_item in app_mag:
                if f"objects/{object_name}/{mag_item}" in hdf5_file:
                    del hdf5_file[f"objects/{object_name}/{mag_item}"]

                if isinstance(app_mag[mag_item], tuple):
                    n_phot = 1

                    app_mag[mag_item] = (
                        app_mag[mag_item][0] - 2.5 * np.log10(dered_phot[mag_item]),
                        app_mag[mag_item][1],
                    )

                    if verbose:
                        print(f"   - {mag_item}:")

                        print(
                            f"      - Mean wavelength (um) = {mean_wavel[mag_item]:.4e}"
                        )

                        print(
                            f"      - Apparent magnitude = {app_mag[mag_item][0]:.2f} +/- "
                            f"{app_mag[mag_item][1]:.2f}"
                        )

                        print(
                            f"      - Flux (W m-2 um-1) = {flux[mag_item]:.2e} +/- "
                            f"{error[mag_item]:.2e}"
                        )

                        if isinstance(deredden, float):
                            print(f"      - Dereddening A_V: {deredden}")

                        elif mag_item in deredden:
                            print(f"      - Dereddening A_V: {deredden[mag_item]}")

                    data = np.asarray(
                        [
                            app_mag[mag_item][0],
                            app_mag[mag_item][1],
                            flux[mag_item],
                            error[mag_item],
                        ]
                    )

                elif isinstance(app_mag[mag_item], list):
                    n_phot = len(app_mag[mag_item])

                    if verbose:
                        print(f"   - {mag_item} ({n_phot} values):")

                    mag_list = []
                    mag_err_list = []

                    for i, dupl_item in enumerate(app_mag[mag_item]):
                        dered_mag = app_mag[mag_item][i][0] - 2.5 * np.log10(
                            dered_phot[mag_item]
                        )
                        app_mag_item = (dered_mag, app_mag[mag_item][i][1])

                        if verbose:
                            print(
                                f"      - Mean wavelength (um) = {mean_wavel[mag_item]:.4e}"
                            )

                            print(
                                f"      - Apparent magnitude = {app_mag_item[0]:.2f} +/- "
                                f"{app_mag_item[1]:.2f}"
                            )

                            print(
                                f"      - Flux (W m-2 um-1) = {flux[mag_item][i]:.2e} +/- "
                                f"{error[mag_item][i]:.2e}"
                            )

                            if isinstance(deredden, float):
                                print(f"      - Dereddening A_V: {deredden}")

                            elif mag_item in deredden:
                                print(f"      - Dereddening A_V: {deredden[mag_item]}")

                        mag_list.append(app_mag_item[0])
                        mag_err_list.append(app_mag_item[1])

                    data = np.asarray(
                        [mag_list, mag_err_list, flux[mag_item], error[mag_item]]
                    )

                # (mag), (mag), (W m-2 um-1), (W m-2 um-1)
                dset = hdf5_file.create_dataset(
                    f"objects/{object_name}/{mag_item}", data=data, dtype="f8"
                )

                dset.attrs["n_phot"] = n_phot

        if flux_density is not None:
            if verbose:
                print("\nFlux densities:")

            for flux_item in flux_density:
                read_filt = ReadFilter(flux_item)
                mean_wavel[flux_item] = read_filt.mean_wavelength()

                if isinstance(deredden, float) or flux_item in deredden:
                    warnings.warn(
                        "The deredden parameter is not supported "
                        "by flux_density. Please use app_mag instead "
                        "and/or open an issue on Github. Ignoring "
                        f"the dereddening of {flux_item}."
                    )

                if f"objects/{object_name}/{flux_item}" in hdf5_file:
                    del hdf5_file[f"objects/{object_name}/{flux_item}"]

                if isinstance(flux_density[flux_item], tuple):
                    data = np.asarray(
                        [
                            np.nan,
                            np.nan,
                            flux_density[flux_item][0],
                            flux_density[flux_item][1],
                        ]
                    )

                    if flux_item in units:
                        from species.util.data_util import convert_units

                        flux_in = np.array([[mean_wavel[flux_item], data[2], data[3]]])
                        flux_out = convert_units(
                            flux_in, ("um", units[flux_item]), convert_from=True
                        )

                        data = [np.nan, np.nan, flux_out[0, 1], flux_out[0, 2]]

                    if verbose:
                        print(f"   - {flux_item}:")
                        print(
                            f"      - Mean wavelength (um) = {mean_wavel[flux_item]:.4e}"
                        )
                        print(
                            f"      - Flux (W m-2 um-1) = {data[2]:.2e} +/- {data[3]:.2e}"
                        )

                    # None, None, (W m-2 um-1), (W m-2 um-1)
                    dset = hdf5_file.create_dataset(
                        f"objects/{object_name}/{flux_item}", data=data, dtype="f8"
                    )

                    dset.attrs["n_phot"] = 1

        if spectrum is not None:
            if verbose:
                print("\nSpectra:")

            read_spec = {}
            read_cov = {}

            # Read spectra

            spec_nan = {}

            for spec_item, spec_value in spectrum.items():
                if "/" in spec_item:
                    raise ValueError(
                        "Spectrum names can't include a slash as "
                        f"character. Please adjust '{spec_item}'."
                    )

                if f"objects/{object_name}/spectrum/{spec_item}" in hdf5_file:
                    del hdf5_file[f"objects/{object_name}/spectrum/{spec_item}"]

                if isinstance(spec_value[0], str):
                    if spec_value[0].endswith(".fits") or spec_value[0].endswith(
                        ".fit"
                    ):
                        with fits.open(spec_value[0]) as hdulist:
                            if (
                                "INSTRU" in hdulist[0].header
                                and hdulist[0].header["INSTRU"] == "GRAVITY"
                            ):
                                # Read data from a FITS file with the GRAVITY format
                                if "OBJECT" in hdulist[0].header:
                                    gravity_object = hdulist[0].header["OBJECT"]
                                else:
                                    gravity_object = None

                                if verbose:
                                    print("   - GRAVITY spectrum:")
                                    print(f"      - Object: {gravity_object}")

                                # Wavelength (um)
                                wavelength = hdulist[1].data["WAVELENGTH"].byteswap()
                                wavelength = wavelength.view(
                                    wavelength.dtype.newbyteorder("=")
                                )

                                # Flux (W m-2 um-1)
                                flux = hdulist[1].data["FLUX"].byteswap()
                                flux = flux.view(flux.dtype.newbyteorder("="))

                                # Covariance (W m-2 um-1)^2
                                covariance = hdulist[1].data["COVARIANCE"].byteswap()
                                covariance = covariance.view(
                                    covariance.dtype.newbyteorder("=")
                                )

                                # Uncorrelated uncertainties (W m-2 um-1)
                                error = np.sqrt(np.diag(covariance))

                                read_spec[spec_item] = np.column_stack(
                                    [wavelength, flux, error]
                                )

                            else:
                                # Otherwise try to read a 2D dataset with 3 columns
                                if verbose:
                                    print("   - Spectrum:")

                                for i, hdu_item in enumerate(hdulist):
                                    data = hdu_item.data.byteswap()
                                    data = data.view(data.dtype.newbyteorder("="))

                                    if (
                                        data.ndim == 2
                                        and 3 in data.shape
                                        and spec_item not in read_spec
                                    ):
                                        if spec_item in units:
                                            from species.util.data_util import (
                                                convert_units,
                                            )

                                            data = convert_units(
                                                data,
                                                units[spec_item],
                                                convert_from=True,
                                            )

                                        read_spec[spec_item] = data

                                if spec_item not in read_spec:
                                    raise ValueError(
                                        f"The spectrum data from {spec_value[0]} cannot "
                                        f"be read. The data format should be 2D with "
                                        f"3 columns."
                                    )

                    else:
                        try:
                            data = np.loadtxt(spec_value[0])

                        except UnicodeDecodeError:
                            raise ValueError(
                                f"The spectrum data from {spec_value[0]} cannot "
                                "be read. Please provide a FITS or ASCII file."
                            )

                        if data.ndim != 2 or 3 not in data.shape:
                            raise ValueError(
                                f"The spectrum data from {spec_value[0]} "
                                "cannot be read. The data format "
                                "should be 2D with 3 columns."
                            )

                        if verbose:
                            print("   - Spectrum:")

                        if spec_item in units:
                            from species.util.data_util import convert_units

                            data = convert_units(
                                data, units[spec_item], convert_from=True
                            )

                        read_spec[spec_item] = data

                else:
                    read_spec[spec_item] = spec_value[0]

                if isinstance(deredden, float):
                    from species.util.dust_util import ism_extinction

                    ext_mag = ism_extinction(deredden, 3.1, read_spec[spec_item][:, 0])
                    read_spec[spec_item][:, 1] *= 10.0 ** (0.4 * ext_mag)

                elif spec_item in deredden:
                    from species.util.dust_util import ism_extinction

                    ext_mag = ism_extinction(
                        deredden[spec_item], 3.1, read_spec[spec_item][:, 0]
                    )
                    read_spec[spec_item][:, 1] *= 10.0 ** (0.4 * ext_mag)

                if (
                    read_spec[spec_item].shape[0] == 3
                    and read_spec[spec_item].shape[1] != 3
                ):
                    warnings.warn(
                        f"Transposing the data of {spec_item} because "
                        f"the first instead of the second axis "
                        f"has a length of 3."
                    )

                    read_spec[spec_item] = read_spec[spec_item].transpose()

                # Remove wavelengths with NaN flux and/or error

                nan_idx_flux = np.isnan(read_spec[spec_item][:, 1])
                nan_idx_error = np.isnan(read_spec[spec_item][:, 2])
                nan_idx = nan_idx_flux + nan_idx_error

                # Add NaN booleans to dictionary for adjusting
                # the covariance matrix later on
                spec_nan[spec_item] = nan_idx

                if np.sum(nan_idx) != 0:
                    read_spec[spec_item] = read_spec[spec_item][~nan_idx, :]

                    warnings.warn(
                        f"Found {np.sum(nan_idx)} fluxes with NaN in "
                        f"the data of {spec_item}. Removing the spectral "
                        f"fluxes that contain a NaN."
                    )

                wavelength = read_spec[spec_item][:, 0]
                flux = read_spec[spec_item][:, 1]
                error = read_spec[spec_item][:, 2]

                if verbose:
                    print(f"      - Database tag: {spec_item}")
                    if isinstance(spec_value[0], str):
                        print(f"      - Filename: {spec_value[0]}")
                    print(f"      - Data shape: {read_spec[spec_item].shape}")
                    print(
                        f"      - Wavelength range (um): {wavelength[0]:.2f} - {wavelength[-1]:.2f}"
                    )
                    print(f"      - Mean flux (W m-2 um-1): {np.nanmean(flux):.2e}")
                    print(f"      - Mean error (W m-2 um-1): {np.nanmean(error):.2e}")

                    if isinstance(deredden, float):
                        print(f"      - Dereddening A_V: {deredden}")

                    elif spec_item in deredden:
                        print(f"      - Dereddening A_V: {deredden[spec_item]}")

            # Read covariance matrix

            for spec_item, spec_value in spectrum.items():
                if spec_value[1] is None:
                    read_cov[spec_item] = None

                elif isinstance(spec_value[1], str):
                    if spec_value[1].endswith(".fits") or spec_value[1].endswith(
                        ".fit"
                    ):
                        with fits.open(spec_value[1]) as hdulist:
                            if (
                                "INSTRU" in hdulist[0].header
                                and hdulist[0].header["INSTRU"] == "GRAVITY"
                            ):
                                # Read data from a FITS file with the GRAVITY format
                                if "OBJECT" in hdulist[0].header:
                                    gravity_object = hdulist[0].header["OBJECT"]
                                else:
                                    gravity_object = None

                                if verbose:
                                    print("   - GRAVITY covariance matrix:")
                                    print(f"      - Object: {gravity_object}")

                                # (W m-2 um-1)^2
                                read_cov[spec_item] = (
                                    hdulist[1].data["COVARIANCE"].byteswap()
                                )
                                read_cov[spec_item] = read_cov[spec_item].view(
                                    read_cov[spec_item].dtype.newbyteorder("=")
                                )

                            else:
                                if spec_item in units:
                                    warnings.warn(
                                        "The unit conversion has not been "
                                        "implemented for covariance matrices. "
                                        "Please open an issue on the Github "
                                        "page if such functionality is required "
                                        "or provide the file with covariances "
                                        "in (W m-2 um-1)^2."
                                    )

                                # Otherwise try to read a square, 2D dataset
                                if verbose:
                                    print("   - Covariance matrix:")

                                for i, hdu_item in enumerate(hdulist):
                                    data = hdu_item.data.byteswap()
                                    data = data.view(data.dtype.newbyteorder("="))

                                    corr_warn = (
                                        f"The matrix from {spec_value[1]} contains "
                                        f"ones along the diagonal. Converting this "
                                        f"correlation matrix into a covariance matrix."
                                    )

                                    from species.util.data_util import (
                                        correlation_to_covariance,
                                    )

                                    if (
                                        data.ndim == 2
                                        and data.shape[0] == data.shape[1]
                                    ):
                                        if spec_item not in read_cov:
                                            if (
                                                data.shape[0]
                                                == read_spec[spec_item].shape[0]
                                            ):
                                                if np.all(np.diag(data) == 1.0):
                                                    warnings.warn(corr_warn)

                                                    read_cov[spec_item] = (
                                                        correlation_to_covariance(
                                                            data,
                                                            read_spec[spec_item][:, 2],
                                                        )
                                                    )

                                                else:
                                                    read_cov[spec_item] = data

                                if spec_item not in read_cov:
                                    raise ValueError(
                                        f"The covariance matrix from {spec_value[1]} cannot "
                                        f"be read. The data format should be 2D with the "
                                        f"same number of wavelength points as the "
                                        f"spectrum."
                                    )

                    else:
                        try:
                            data = np.loadtxt(spec_value[1])
                        except UnicodeDecodeError:
                            raise ValueError(
                                f"The covariance matrix from {spec_value[1]} "
                                f"cannot be read. Please provide a "
                                f"FITS or ASCII file."
                            )

                        if data.ndim != 2 or data.shape[0] != data.shape[1]:
                            raise ValueError(
                                f"The covariance matrix from {spec_value[1]} "
                                f"cannot be read. The data format "
                                f"should be 2D with the same number of "
                                f"wavelength points as the spectrum."
                            )

                        if verbose:
                            print("   - Covariance matrix:")

                        if np.all(np.diag(data) == 1.0):
                            warnings.warn(
                                f"The matrix from {spec_value[1]} contains "
                                f"ones on the diagonal. Converting this "
                                f" correlation matrix into a covariance "
                                f"matrix."
                            )

                            from species.util.data_util import correlation_to_covariance

                            read_cov[spec_item] = correlation_to_covariance(
                                data, read_spec[spec_item][:, 2]
                            )

                        else:
                            read_cov[spec_item] = data

                else:
                    read_cov[spec_item] = spec_value[1]

                if read_cov[spec_item] is not None:
                    # Remove the wavelengths for which the flux is NaN
                    read_cov[spec_item] = read_cov[spec_item][~spec_nan[spec_item], :]
                    read_cov[spec_item] = read_cov[spec_item][:, ~spec_nan[spec_item]]

                if verbose and read_cov[spec_item] is not None:
                    print(f"      - Database tag: {spec_item}")
                    if isinstance(spec_value[1], str):
                        print(f"      - Filename: {spec_value[1]}")
                    print(f"      - Data shape: {read_cov[spec_item].shape}")

            if verbose:
                print("   - Instrument resolution:")

            for spec_item, spec_value in spectrum.items():
                hdf5_file.create_dataset(
                    f"objects/{object_name}/spectrum/{spec_item}/spectrum",
                    data=read_spec[spec_item],
                    dtype="f8",
                )

                if read_cov[spec_item] is not None:
                    hdf5_file.create_dataset(
                        f"objects/{object_name}/spectrum/{spec_item}/covariance",
                        data=read_cov[spec_item],
                        dtype="f8",
                    )

                    hdf5_file.create_dataset(
                        f"objects/{object_name}/spectrum/{spec_item}/inv_covariance",
                        data=np.linalg.inv(read_cov[spec_item]),
                        dtype="f8",
                    )

                dset = hdf5_file[f"objects/{object_name}/spectrum/{spec_item}"]

                if spec_value[2] is None:
                    if verbose:
                        print(f"      - {spec_item}: None")
                    dset.attrs["specres"] = np.nan

                else:
                    if verbose:
                        print(f"      - {spec_item}: {spec_value[2]:.1f}")
                    dset.attrs["specres"] = float(spec_value[2])

        hdf5_file.close()

    @typechecked
    def add_simple_object(
        self,
        object_name: str,
        simple_database: Optional[str] = None,
    ) -> None:
        """
        Function for retrieving data of an individual object from
        the `SIMPLE database <https://simple-bd-archive.org>`_ and
        adding the data to the ``species`` database. This function
        includes code that has been adapted from `SEDkitSIMPLE
        <https://github.com/dr-rodriguez/SEDkitSIMPLE>`_.

        Parameters
        ----------
        object_name : str
            Object name that will be used for the query and also used
            as label by which the data will be stored in the database.
        simple_database : str, None
            Path to the ``SIMPLE`` database file `SIMPLE.db`. The
            binary file is automatically  downloaded from the `Github
            page <https://github.com/SIMPLE-AstroDB/SIMPLE-binary>`_
            of ``SIMPLE`` when the argument is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print_section("Add SIMPLE object")

        from astrodbkit2.astrodb import Database as astro_db
        from sqlalchemy import and_

        if simple_database is None:
            simple_database = os.path.join(self.data_folder, "SIMPLE.db")
            url = (
                "https://github.com/SIMPLE-AstroDB/" "SIMPLE-binary/raw/main/SIMPLE.db"
            )

            if not os.path.isfile(simple_database):
                pooch.retrieve(
                    url=url,
                    known_hash=None,
                    fname="SIMPLE.db",
                    path=self.data_folder,
                    progressbar=True,
                )

        print(f"SIMPLE database: {simple_database}")

        simple_db = astro_db(f"sqlite:///{simple_database}")

        def _fetch_bibcode(ref: str) -> Optional[str]:
            """
            Internal function for fetching a reference
            from the SIMPLE database

            Parameters
            ----------
            ref : str
                Reference for which the `bibcode` should be retrieved.

            Returns
            -------
            NoneType
                Bibcode of the reference. Returns a ``None`` in
                case the bibcode is not available.
            """

            bibcode = None

            if hasattr(simple_db.Publications.c, "bibcode"):
                if hasattr(simple_db.Publications.c, "reference"):
                    simple_query = simple_db.query(simple_db.Publications.c.bibcode)

                    simple_result = simple_query.filter(
                        simple_db.Publications.c.reference == ref
                    )

                    bibcode = simple_result.table()["bibcode"][0]

            return bibcode

        # Get SIMPLE name of object

        print(f"\nObject name: {object_name}")

        sources = simple_db.search_object(object_name)

        if len(sources) > 0:
            if len(sources) > 1:
                warnings.warn(
                    "Multiple sources found in the SIMPLE "
                    f"database for {object_name}. The first "
                    "retrieved source will be selected."
                )

            simple_id = sources["source"][0]

        else:
            raise ValueError(
                f"The requested source, {object_name}, is "
                "not found in the SIMPLE database."
            )

        print(f"Retrieved name: {simple_id}")

        # Get inventory of the queried object

        inventory = simple_db.inventory(simple_id, pretty_print=False)

        # Get coordinates

        coordinates = None

        if "Sources" in inventory:
            if len(inventory["Sources"]) == 1:
                data = inventory["Sources"][0]
                coordinates = SkyCoord(ra=data["ra"] * u.deg, dec=data["dec"] * u.deg)

        if coordinates is None:
            print("Coordinates: None")

        else:
            coord_str = coordinates.to_string(
                "hmsdms", alwayssign=True, precision=2, pad=True
            )
            print(f"Coordinates: {coord_str}")

        # Get parallax

        parallax = None

        if "Parallaxes" in inventory:
            for row in inventory["Parallaxes"]:
                value = Quantity(row["parallax"], unit=u.mas)
                error = Quantity(row["parallax_error"], unit=u.mas)
                bibcode = _fetch_bibcode(row["reference"])
                parallax = (value.value, error.value)

                if row["adopted"]:
                    break

        if parallax is None:
            print("Parallax: None")

        else:
            print(
                f"Parallax: {(parallax)[0]:.2f} +/- "
                f"{(parallax)[1]:.2f} mas (bibcode: {bibcode})"
            )

        # Get spectral type

        spectral_type = None

        if "SpectralTypes" in inventory:
            for row in inventory["SpectralTypes"]:
                bibcode = _fetch_bibcode(row["reference"])
                spectral_type = row["spectral_type_string"]

                if row["adopted"]:
                    break

        if spectral_type is None:
            print("Spectral type: None")

        else:
            print(f"Spectral type: {spectral_type} (bibcode: {bibcode})")

        # Get photometry

        app_mag = None

        if "Photometry" in inventory:
            app_mag = {}
            bibcode = {}

            for row in inventory["Photometry"]:
                filter_name = None

                if row["band"].split(".")[0] == "GAIA3":
                    filter_name = "GAIA/" + row["band"]
                elif row["band"].split(".")[0] == "2MASS":
                    filter_name = "2MASS/" + row["band"]
                elif row["band"].split(".")[0] == "WISE":
                    filter_name = "WISE/" + row["band"]
                elif row["band"].split(".")[0] == "WISE":
                    filter_name = "WISE/" + row["band"]

                if filter_name is not None and row["magnitude_error"] is not None:
                    bibcode[filter_name] = _fetch_bibcode(row["reference"])
                    app_mag[filter_name] = (row["magnitude"], row["magnitude_error"])

        if app_mag is None:
            print("Magnitudes: None")

        else:
            print("\nMagnitudes:")
            for key, value in app_mag.items():
                if value[1] > 1e-2:
                    print(
                        f"   - {key} = {value[0]:.2f} +/- {value[1]:.2f} (bibcode: {bibcode[key]})"
                    )
                elif value[1] < 1e-2 and value[1] > 1e-3:
                    print(
                        f"   - {key} = {value[0]:.3f} +/- {value[1]:.3f} (bibcode: {bibcode[key]})"
                    )
                elif value[1] < 1e-3 and value[1] > 1e-4:
                    print(
                        f"   - {key} = {value[0]:.4f} +/- {value[1]:.4f} (bibcode: {bibcode[key]})"
                    )
                else:
                    print(
                        f"   - {key} = {value[0]:.5f} +/- {value[1]:.5f} (bibcode: {bibcode[key]})"
                    )

        # Get spectra

        spectrum = None

        if "Spectra" in inventory:
            spectrum = {}
            bibcode = {}

            for row in inventory["Spectra"]:
                spec_tag = f"{row['telescope']} {row['instrument']}"
                if row["mode"] != "Missing":
                    spec_tag += f" {row['mode']}"

                simple_query = simple_db.query(simple_db.Spectra)

                spec_table = simple_query.filter(
                    and_(
                        simple_db.Spectra.c.source == simple_id,
                        simple_db.Spectra.c.regime == row["regime"],
                        simple_db.Spectra.c.reference == row["reference"],
                        simple_db.Spectra.c.observation_date == row["observation_date"],
                    )
                ).astropy(spectra="spectrum", spectra_format=None)

                wavel = spec_table["spectrum"][0].wavelength
                flux = spec_table["spectrum"][0].flux

                wavel = wavel.to(u.micron).value
                flux = flux.to(u.W / u.m**2 / u.um).value

                if spec_table["spectrum"][0].uncertainty is not None:
                    error = spec_table["spectrum"][0].uncertainty.quantity
                    error = error.to(u.W / u.m**2 / u.um).value
                else:
                    error = np.full(wavel.size, np.nan)

                spec_data = np.column_stack([wavel, flux, error])

                spec_res = input(
                    "Please provide the resolving power, "
                    f"R = lambda/Delta_lambda, for '{spec_tag}': "
                )

                bibcode[spec_tag] = _fetch_bibcode(row["reference"])
                spectrum[spec_tag] = (
                    spec_data,
                    np.diag(spec_data[:, 2] ** 2),
                    float(spec_res),
                )

        if spectrum is None:
            if app_mag is not None:
                print()

            print("Spectra: None")

        else:
            print("\nSpectra:")
            for key, value in spectrum.items():
                print(f"   - {key}")
                print(f"      - Array shape: {value[0].shape}")
                print(f"      - lambda/d_lambda: {value[2]}")
                print(f"      - bibcode: {bibcode[key]}")

        self.add_object(
            object_name=object_name,
            parallax=parallax,
            app_mag=app_mag,
            spectrum=spectrum,
            verbose=False,
        )

    @typechecked
    def add_photometry(self, phot_library: str) -> None:
        """
        Function for adding a photometry library to the database.
        Parameters
        ----------
        phot_library : str
            Photometric library ('vlm-plx', 'leggett', or 'beiler2024').

        Returns
        -------
        NoneType
            None
        """

        print_section("Add photometric library")

        print(f"Database tag: {phot_library}")

        with h5py.File(self.database, "a") as hdf5_file:
            if f"photometry/{phot_library}" in hdf5_file:
                del hdf5_file[f"photometry/{phot_library}"]

            if phot_library[0:7] == "vlm-plx":
                from species.data.phot_data.phot_vlm_plx import add_vlm_plx

                print("Library: Database of Ultracool Parallaxes")
                add_vlm_plx(self.data_folder, hdf5_file)

            elif phot_library[0:7] == "leggett":
                from species.data.phot_data.phot_leggett import add_leggett

                add_leggett(self.data_folder, hdf5_file)

            elif phot_library[0:10] == "beiler2024":
                from species.data.phot_data.phot_jwst_ydwarfs import add_jwst_ydwarfs

                add_jwst_ydwarfs(self.data_folder, hdf5_file)

    @typechecked
    def add_calibration(
        self,
        tag: str,
        filename: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        units: Optional[Dict[str, str]] = None,
        scaling: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Function for adding a calibration spectrum to the database.

        Parameters
        ----------
        tag : str
            Tag name in the database.
        filename : str, None
            Name of the file that contains the calibration spectrum.
            The file could be either a plain text file, in which the
            first column contains the wavelength (um), the second
            column the flux density (W m-2 um-1), and the third
            column the uncertainty (W m-2 um-1). Or, a FITS file
            can be provided in which the data is stored as a 2D
            array in the primary HDU. The ``data`` argument is
            not used if set to ``None``.
        data : np.ndarray, None
            Spectrum stored as 3D array with shape
            ``(n_wavelength, 3)``. The first column should contain the
            wavelength (um), the second column the flux density
            (W m-2 um-1), and the third column the error (W m-2 um-1).
            The ``filename`` argument is used if set to ``None``.
        units : dict, None
            Dictionary with the wavelength and flux units, e.g.
            ``{'wavelength': 'angstrom', 'flux': 'w m-2'}``. The
            default units (um and W m-2 um-1) are used if set to
            ``None``.
        scaling : tuple(float, float), None
            Scaling for the wavelength and flux as
            ``(scaling_wavelength, scaling_flux)``. Not
            used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        if filename is None and data is None:
            raise ValueError(
                "Either the 'filename' or 'data' argument should be provided."
            )

        if scaling is None:
            scaling = (1.0, 1.0)

        hdf5_file = h5py.File(self.database, "a")

        if "spectra/calibration" not in hdf5_file:
            hdf5_file.create_group("spectra/calibration")

        if "spectra/calibration/" + tag in hdf5_file:
            del hdf5_file["spectra/calibration/" + tag]

        if filename is not None:
            if filename[-5:] == ".fits":
                data = fits.getdata(filename).byteswap()
                data = data.view(data.dtype.newbyteorder("="))

                if data.ndim != 2:
                    raise RuntimeError(
                        "The FITS file that is provided "
                        "as argument of 'filename' does "
                        "not contain a 2D dataset."
                    )

                if data.shape[1] != 3 and data.shape[0]:
                    warnings.warn(
                        f"Transposing the data that is read "
                        f"from {filename} because the shape "
                        f"is {data.shape} instead of "
                        f"{data.shape[1], data.shape[0]}."
                    )

                    data = np.transpose(data)

            else:
                data = np.loadtxt(filename)

        nan_idx = np.isnan(data[:, 1])

        if np.sum(nan_idx) != 0:
            data = data[~nan_idx, :]

            warnings.warn(
                f"Found {np.sum(nan_idx)} fluxes with NaN in "
                f"the data of {filename}. Removing the "
                f"spectral fluxes that contain a NaN."
            )

        if units is None:
            wavelength = scaling[0] * data[:, 0]  # (um)
            flux = scaling[1] * data[:, 1]  # (W m-2 um-1)

        else:
            if units["wavelength"] == "um":
                wavelength = scaling[0] * data[:, 0]  # (um)

            elif units["wavelength"] == "angstrom":
                wavelength = scaling[0] * data[:, 0] * 1e-4  # (um)

            if units["flux"] == "w m-2 um-1":
                flux = scaling[1] * data[:, 1]  # (W m-2 um-1)

            elif units["flux"] == "w m-2":
                if units["wavelength"] == "um":
                    flux = scaling[1] * data[:, 1] / wavelength  # (W m-2 um-1)

        if data.shape[1] == 3:
            if units is None:
                error = scaling[1] * data[:, 2]  # (W m-2 um-1)

            else:
                if units["flux"] == "w m-2 um-1":
                    error = scaling[1] * data[:, 2]  # (W m-2 um-1)

                elif units["flux"] == "w m-2":
                    if units["wavelength"] == "um":
                        error = scaling[1] * data[:, 2] / wavelength  # (W m-2 um-1)

        else:
            error = np.repeat(0.0, wavelength.size)

        # nan_idx = np.isnan(flux)
        #
        # if np.sum(nan_idx) != 0:
        #     wavelength = wavelength[~nan_idx]
        #     flux = flux[~nan_idx]
        #     error = error[~nan_idx]
        #
        #     warnings.warn(
        #         f"Found {np.sum(nan_idx)} fluxes with NaN in "
        #         f"the calibration spectrum. Removing the "
        #         f"spectral fluxes that contain a NaN."
        #     )

        print(f"Adding calibration spectrum: {tag}...", end="", flush=True)

        hdf5_file.create_dataset(
            f"spectra/calibration/{tag}",
            data=np.vstack((wavelength, flux, error)),
            dtype="f8",
        )

        hdf5_file.close()

        print(" [DONE]")

    @typechecked
    def add_spectra(
        self, spec_library: str, sptypes: Optional[List[str]] = None
    ) -> None:
        """
        Function for adding empirical spectral libraries to the
        database. The spectra are stored together with several
        attributes such as spectral type, parallax, and Simbad ID.
        The spectra can be read with the functionalities of
        :class:`~species.read.read_spectrum.ReadSpectrum`.

        Parameters
        ----------
        spec_library : str
            Spectral library ('irtf', 'spex', 'kesseli+2017',
            'bonnefoy+2014', 'allers+2013').
        sptypes : list(str)
            Spectral types ('F', 'G', 'K', 'M', 'L', 'T'). Currently
            only implemented for ``spec_library='irtf'``.

        Returns
        -------
        NoneType
            None
        """

        from species.data.spec_data.add_spec_data import add_spec_library

        print_section("Add spectral library")

        print(f"Database tag: {spec_library}")
        print(f"Spectral types: {sptypes}")

        with h5py.File(self.database, "a") as hdf5_file:
            if f"spectra/{spec_library}" in hdf5_file:
                del hdf5_file[f"spectra/{spec_library}"]

            add_spec_library(self.data_folder, hdf5_file, spec_library, sptypes)

    @typechecked
    def add_samples(
        self,
        tag: str,
        sampler: str,
        samples: np.ndarray,
        ln_prob: np.ndarray,
        modelpar: List[str],
        bounds: Dict[str, Tuple[float, float]],
        normal_prior: Dict[str, Tuple[float, float]],
        fixed_param: Dict[str, float],
        spec_labels: Optional[List[str]] = None,
        attr_dict: Optional[Dict] = None,
    ):
        """
        This function stores the posterior samples produced
        by :class:`~species.fit.fit_model.FitModel`
        in the database, including some additional attributes.

        Parameters
        ----------
        tag : str
            Database tag.
        sampler : str
            Sampler ('multinest', 'ultranest', 'dynesty').
        samples : np.ndarray
            Samples of the posterior.
        ln_prob : np.ndarray
            Log-likelihood for each sample.
        modelpar : list(str)
            List with the model parameter names.
        bounds : dict
            Dictionary with the (log-)uniform priors.
        normal_prior : dict
            Dictionary with the normal priors.
        fixed_param : dict
            Dictionary with the fixed parameters
        spec_labels : list(str), None
            List with the spectrum labels that are used for fitting an
            additional scaling parameter. Not used if set to ``None``.
        attr_dict : dict, None
            Dictionary with data that will be stored as attributes
            of the dataset with samples. Not used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print_section("Add posterior samples")

        print(f"Database tag: {tag}")
        print(f"Sampler: {sampler}")
        print(f"Samples shape: {samples.shape}")

        if spec_labels is None:
            spec_labels = []

        with h5py.File(self.database, "a") as hdf5_file:
            if f"results/fit/{tag}" in hdf5_file:
                del hdf5_file[f"results/fit/{tag}"]

            dset = hdf5_file.create_dataset(
                f"results/fit/{tag}/samples", data=samples, dtype="f8"
            )
            hdf5_file.create_dataset(
                f"results/fit/{tag}/ln_prob", data=ln_prob, dtype="f8"
            )

            for key, value in bounds.items():
                group_path = f"results/fit/{tag}/bounds/{key}"
                hdf5_file.create_dataset(group_path, data=value, dtype="f8")

            for key, value in normal_prior.items():
                group_path = f"results/fit/{tag}/normal_prior/{key}"
                hdf5_file.create_dataset(group_path, data=value, dtype="f8")

            for key, value in fixed_param.items():
                group_path = f"results/fit/{tag}/fixed_param/{key}"
                hdf5_file.create_dataset(group_path, data=value, dtype="f8")

            if "model_type" in attr_dict:
                dset.attrs["model_type"] = attr_dict["model_type"]
            elif "spec_type" in attr_dict:
                dset.attrs["model_type"] = attr_dict["spec_type"]

            if "model_name" in attr_dict:
                dset.attrs["model_name"] = attr_dict["model_name"]
            elif "spec_name" in attr_dict:
                dset.attrs["model_name"] = attr_dict["spec_name"]

            dset.attrs["n_param"] = int(len(modelpar))
            dset.attrs["sampler"] = str(sampler)
            dset.attrs["n_bounds"] = int(len(bounds))
            dset.attrs["n_normal_prior"] = int(len(normal_prior))
            dset.attrs["n_fixed"] = int(len(fixed_param))

            if "mean_accept" in attr_dict:
                mean_accept = float(attr_dict["mean_accept"])
                dset.attrs["mean_accept"] = mean_accept
                print(f"Mean acceptance fraction: {mean_accept:.3f}")

            if "ln_evidence" in attr_dict:
                ln_evidence = attr_dict["ln_evidence"]
                dset.attrs["ln_evidence"] = ln_evidence[0]
                dset.attrs["ln_evidence_error"] = ln_evidence[1]

                print(
                    f"ln(Z) = {float(ln_evidence[0]):.2f} "
                    f"+/- {float(ln_evidence[1]):.2f}"
                )

            count_scaling = 0

            for i, item in enumerate(modelpar):
                dset.attrs[f"parameter{i}"] = str(item)

                if item in spec_labels:
                    dset.attrs[f"scaling{count_scaling}"] = str(item)
                    count_scaling += 1

            dset.attrs["n_scaling"] = int(count_scaling)

            for key, value in attr_dict.items():
                dset.attrs[key] = value

    @typechecked
    def get_probable_sample(
        self,
        tag: str,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Function for extracting the sample parameters
        with the maximum likelihood.

        Parameters
        ----------
        tag : str
            Database tag with the posterior results.
        verbose : bool
            Print output, including the parameter values.

        Returns
        -------
        dict
            Parameters of the sample with the maximum likelihood.
        """

        from species.util.model_util import binary_to_single, check_nearest_spec

        if verbose:
            print_section("Get sample with the maximum likelihood")
            print(f"Database tag: {tag}")

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/fit/{tag}/samples"]
            dset_attrs = dict(dset.attrs)

            samples = np.asarray(dset)
            ln_prob = np.asarray(hdf5_file[f"results/fit/{tag}/ln_prob"])

        if "n_param" in dset_attrs:
            n_param = dset_attrs["n_param"]
        else:
            n_param = dset_attrs["nparam"]

        index_max = np.argmax(ln_prob)
        max_sample = samples[index_max,]

        prob_sample = {}

        for i in range(n_param):
            par_key = dset_attrs[f"parameter{i}"]
            par_value = max_sample[i]

            prob_sample[par_key] = par_value

        if (
            "parallax" not in prob_sample
            and "parallax_0" not in prob_sample
            and "parallax" in dset_attrs
        ):
            prob_sample["parallax"] = dset_attrs["parallax"]

        elif "distance" not in prob_sample and "distance" in dset_attrs:
            prob_sample["distance"] = dset_attrs["distance"]

        if "pt_smooth" in dset_attrs:
            prob_sample["pt_smooth"] = dset_attrs["pt_smooth"]

        if "ext_model" in dset_attrs:
            prob_sample["ext_model"] = dset_attrs["ext_model"]

        if verbose:
            print("\nParameters:")
            for param_key, param_value in prob_sample.items():
                if isinstance(param_value, float):
                    if -0.01 < param_value < 0.01:
                        print(f"   - {param_key} = {param_value:.2e}")
                    else:
                        print(f"   - {param_key} = {param_value:.2f}")

        if "model_type" in dset_attrs and dset_attrs["model_type"] == "atmosphere":

            if "binary" in dset_attrs and dset_attrs["binary"]:
                param_0 = binary_to_single(prob_sample, 0)
                check_nearest_spec(dset_attrs["model_name"], param_0)

                param_1 = binary_to_single(prob_sample, 1)
                check_nearest_spec(dset_attrs["model_name"], param_1)

            else:
                check_nearest_spec(dset_attrs["model_name"], prob_sample)

        return prob_sample

    @typechecked
    def get_median_sample(
        self,
        tag: str,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Function for extracting the median parameter values
        from the posterior samples.

        Parameters
        ----------
        tag : str
            Database tag with the posterior results.
        verbose : bool
            Print output, including the parameter values.

        Returns
        -------
        dict
            Median parameter values of the posterior distribution.
        """

        from species.util.model_util import binary_to_single, check_nearest_spec

        if verbose:
            print_section("Get median parameters")
            print(f"Database tag: {tag}")

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/fit/{tag}/samples"]
            samples = np.asarray(dset)
            dset_attrs = dict(dset.attrs)

        if "n_param" in dset_attrs:
            n_param = dset_attrs["n_param"]
        else:
            n_param = dset_attrs["nparam"]

        median_sample = {}

        for i in range(n_param):
            par_key = dset_attrs[f"parameter{i}"]
            par_value = np.median(samples[:, i])
            median_sample[par_key] = par_value

        if (
            "parallax" not in median_sample
            and "parallax_0" not in median_sample
            and "parallax" in dset_attrs
        ):
            median_sample["parallax"] = dset_attrs["parallax"]

        elif "distance" not in median_sample and "distance" in dset_attrs:
            median_sample["distance"] = dset_attrs["distance"]

        if "pt_smooth" in dset_attrs:
            median_sample["pt_smooth"] = dset_attrs["pt_smooth"]

        if "ext_model" in dset_attrs:
            median_sample["ext_model"] = dset_attrs["ext_model"]

        if verbose:
            print("\nParameters:")
            for param_key, param_value in median_sample.items():
                if isinstance(param_value, float):
                    if param_value < 0.01:
                        print(f"   - {param_key} = {param_value:.2e}")
                    else:
                        print(f"   - {param_key} = {param_value:.2f}")

        if "model_type" in dset_attrs and dset_attrs["model_type"] == "atmosphere":

            if "binary" in dset_attrs and dset_attrs["binary"]:
                param_0 = binary_to_single(median_sample, 0)
                check_nearest_spec(dset_attrs["model_name"], param_0)

                param_1 = binary_to_single(median_sample, 1)
                check_nearest_spec(dset_attrs["model_name"], param_1)

            else:
                check_nearest_spec(dset_attrs["model_name"], median_sample)

        return median_sample

    @typechecked
    def get_compare_sample(self, tag: str, verbose: bool = True) -> Dict[str, float]:
        """
        Function for extracting the sample parameters for which
        the goodness-of-fit statistic has been minimized when using
        :func:`~species.fit.compare_spectra.CompareSpectra.compare_model`
        for comparing data with a grid of model spectra.

        Parameters
        ----------
        tag : str
            Database tag where the results from
            :func:`~species.fit.compare_spectra.CompareSpectra.compare_model`
            are stored.
        verbose : bool
            Print output, including the parameter values.

        Returns
        -------
        dict
            Dictionary with the best-fit parameters, including optional
            scaling parameters for spectra that can be applied by
            running :func:`~species.util.read_util.update_objectbox`
            on an :func:`~species.core.box.ObjectBox` by providing
            the returned dictionary from
            :func:`~species.data.database.Database.get_compare_sample`
            as argument.
        """

        if verbose:
            print_section("Get best comparison parameters")
            print(f"Database tag: {tag}")

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/comparison/{tag}/goodness_of_fit"]
            dset_attrs = dict(dset.attrs)

        n_param = dset_attrs["n_param"]
        n_scale_spec = dset_attrs["n_scale_spec"]

        model_param = {}

        for i in range(n_param):
            model_param[dset_attrs[f"parameter{i}"]] = dset_attrs[f"best_param{i}"]

        if "parallax" in dset_attrs:
            model_param["parallax"] = dset_attrs["parallax"]

        elif "distance" in dset_attrs:
            model_param["distance"] = dset_attrs["distance"]

        model_param["radius"] = dset_attrs["radius"]

        for i in range(n_scale_spec):
            scale_spec = dset_attrs[f"scale_spec{i}"]
            model_param[f"scaling_{scale_spec}"] = dset_attrs[f"scaling_{scale_spec}"]

        if verbose:
            print("\nParameters:")
            for param_key, param_value in model_param.items():
                if param_value < 0.01:
                    print(f"   - {param_key} = {param_value:.2e}")
                else:
                    print(f"   - {param_key} = {param_value:.2f}")

        return model_param

    @typechecked
    def get_mcmc_spectra(
        self,
        tag: str,
        random: Optional[int] = None,
        wavel_range: Optional[Union[Tuple[float, float], str]] = None,
        spec_res: Optional[float] = None,
        wavel_resample: Optional[np.ndarray] = None,
    ) -> Union[
        Union[List[ModelBox], List[SpectrumBox]],
        Tuple[List[ModelBox], List[ModelBox], List[ModelBox]],
    ]:
        """
        Function for drawing random model spectra from the
        sampled posterior distribution.

        Parameters
        ----------
        tag : str
            Database tag with the posterior samples.
        random : int, None
            Number of random samples. All samples are selected
            by setting the argument to ``None``.
        wavel_range : tuple(float, float), str, None
            Wavelength range (um) or filter name. Full spectrum is
            used if set to ``None``.
        spec_res : float, None
            Spectral resolution that is used for the smoothing with
            a Gaussian kernel. No smoothing is applied if the
            argument set to ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the model spectrum will
            be resampled. The resampling is applied after the optional
            smoothing to the resolution of ``spec_res``.

        Returns
        -------
        list(species.core.box.ModelBox)
            List with ``ModelBox`` objects. When fitting an unresolved
            binary system, this list contains the model spectra of
            the combined flux.
        list(species.core.box.ModelBox)
            The list of ``ModelBox`` objects is only returned when
            fitting and unresolved binary system. It contains the
            model spectra of the first component.
        list(species.core.box.ModelBox)
            The list of ``ModelBox`` objects is only returned when
            fitting and unresolved binary system. It contains the
            model spectra of the second component.
        """

        print_section(f"Get posterior spectra")

        print(f"Database tag: {tag}")
        print(f"Wavelength range (um): {wavel_range}")
        print(f"Resolution: {spec_res}")

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/fit/{tag}/samples"]
            samples = np.array(dset)
            dset_attrs = dict(dset.attrs)

        if "model_type" in dset_attrs:
            model_type = dset_attrs["model_type"]
        else:
            model_type = dset_attrs["type"]

        if "model_name" in dset_attrs:
            model_name = dset_attrs["model_name"]
        else:
            model_name = dset_attrs["spectrum"]

        if "n_param" in dset_attrs:
            n_param = dset_attrs["n_param"]
        elif "nparam" in dset_attrs:
            n_param = dset_attrs["nparam"]

        if "n_scaling" in dset_attrs:
            n_scaling = dset_attrs["n_scaling"]
        elif "nscaling" in dset_attrs:
            n_scaling = dset_attrs["nscaling"]
        else:
            n_scaling = 0

        if "n_error" in dset_attrs:
            n_error = dset_attrs["n_error"]
        else:
            n_error = 0

        if "binary" in dset_attrs:
            binary = dset_attrs["binary"]
        else:
            binary = False

        if "ext_filter" in dset_attrs:
            ext_filter = dset_attrs["ext_filter"]
        else:
            ext_filter = None

        ignore_param = []

        for i in range(n_scaling):
            ignore_param.append(dset_attrs[f"scaling{i}"])

        for i in range(n_error):
            ignore_param.append(dset_attrs[f"error{i}"])

        for i in range(n_param):
            if dset_attrs[f"parameter{i}"][:9] == "corr_len_":
                ignore_param.append(dset_attrs[f"parameter{i}"])

            elif dset_attrs[f"parameter{i}"][:9] == "corr_amp_":
                ignore_param.append(dset_attrs[f"parameter{i}"])

        if spec_res is not None and model_type == "calibration":
            warnings.warn(
                "Smoothing of the spectral resolution is not "
                "implemented for calibration spectra."
            )

        if "parallax" in dset_attrs:
            parallax = dset_attrs["parallax"]
        else:
            parallax = None

        if "distance" in dset_attrs:
            distance = dset_attrs["distance"]
        else:
            distance = None

        if random is None:
            print(f"Number of samples: {samples.shape[0]}")
        else:
            print(f"Number of samples: {random}")
            rng = np.random.default_rng()
            samples = rng.choice(samples, random, replace=False, axis=0, shuffle=False)

        param = []
        for i in range(n_param):
            param.append(dset_attrs[f"parameter{i}"])

        if model_type in ["model", "atmosphere"]:
            if model_name == "planck":
                from species.read.read_planck import ReadPlanck

                readmodel = ReadPlanck(wavel_range)

            elif model_name == "powerlaw":
                pass

            else:
                from species.read.read_model import ReadModel

                readmodel = ReadModel(model_name, wavel_range=wavel_range)

        elif model_type == "calibration":
            from species.read.read_calibration import ReadCalibration

            readcalib = ReadCalibration(model_name, filter_name=None)

        boxes = []

        if binary:
            boxes_0 = []
            boxes_1 = []
        else:
            boxes_0 = None
            boxes_1 = None

        print()

        for i in tqdm(range(samples.shape[0])):
            model_param = {}
            for j in range(samples.shape[1]):
                if param[j] not in ignore_param:
                    model_param[param[j]] = samples[i, j]

            if (
                "parallax" not in model_param
                and "parallax_0" not in model_param
                and parallax is not None
            ):
                model_param["parallax"] = parallax

            elif "distance" not in model_param and distance is not None:
                model_param["distance"] = distance

            if "ext_model" in dset_attrs:
                model_param["ext_model"] = dset_attrs["ext_model"]

            if model_type in ["model", "atmosphere"]:
                if model_name == "planck":
                    spec_box = readmodel.get_spectrum(
                        model_param,
                        spec_res,
                        wavel_resample=wavel_resample,
                    )

                elif model_name == "powerlaw":
                    if wavel_resample is not None:
                        warnings.warn(
                            "The 'wavel_resample' parameter is not "
                            "supported by the 'powerlaw' model so "
                            "the argument will be ignored."
                        )

                    from species.util.model_util import powerlaw_spectrum

                    spec_box = powerlaw_spectrum(wavel_range, model_param)

                else:
                    from species.util.model_util import binary_to_single

                    if binary:
                        param_0 = binary_to_single(model_param, 0)

                        spec_box_0 = readmodel.get_model(
                            param_0,
                            spec_res=spec_res,
                            wavel_resample=wavel_resample,
                            ext_filter=ext_filter,
                        )

                        param_1 = binary_to_single(model_param, 1)

                        spec_box_1 = readmodel.get_model(
                            param_1,
                            spec_res=spec_res,
                            wavel_resample=wavel_resample,
                            ext_filter=ext_filter,
                        )

                        # Weighted flux of two spectra for atmospheric asymmetries
                        # Or simply the same in case of an actual binary system

                        if "spec_weight" in model_param:
                            flux_comb = (
                                model_param["spec_weight"] * spec_box_0.flux
                                + (1.0 - model_param["spec_weight"]) * spec_box_1.flux
                            )

                        else:
                            flux_comb = spec_box_0.flux + spec_box_1.flux

                        spec_box = create_box(
                            boxtype="model",
                            model=model_name,
                            wavelength=spec_box_0.wavelength,
                            flux=flux_comb,
                            parameters=model_param,
                            quantity="flux",
                        )

                    else:
                        spec_box = readmodel.get_model(
                            model_param,
                            spec_res=spec_res,
                            wavel_resample=wavel_resample,
                            ext_filter=ext_filter,
                        )

            elif model_type == "calibration":
                spec_box = readcalib.get_spectrum(model_param)

            else:
                spec_box = None

            boxes.append(spec_box)

            if binary:
                boxes_0.append(spec_box_0)
                boxes_1.append(spec_box_1)

        if binary:
            return boxes, boxes_0, boxes_1
        else:
            return boxes

    @typechecked
    def get_mcmc_photometry(
        self,
        tag: str,
        filter_name: str,
        random: Optional[int] = None,
        phot_type: str = "magnitude",
        flux_units: str = "W m-2 um-1",
    ) -> np.ndarray:
        """
        Function for calculating synthetic magnitudes or fluxes
        from the posterior samples.

        Parameters
        ----------
        tag : str
            Database tag with the posterior samples.
        filter_name : str
            Filter name for which the synthetic photometry
            will be computed.
        random : int, None
            Number of random samples. All samples are selected
            by setting the argument to ``None``.
        phot_type : str
            Photometry type ('magnitude' or 'flux').
        flux_units : tuple(str, str), None
            Flux units that will be used when the ``phot_type``
            argument is set to ``'flux``. Supported units can
            be found in the docstring of
            :func:`~species.util.data_util.convert_units`.

        Returns
        -------
        np.ndarray
            Synthetic magnitudes or fluxes.
        """

        from species.read.read_filter import ReadFilter

        if phot_type not in ["magnitude", "flux"]:
            raise ValueError(
                "The argument of 'phot_type' is not recognized "
                "and should be set to 'magnitude' or 'flux'."
            )

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/fit/{tag}/samples"]
            dset_attrs = dict(dset.attrs)
            samples = np.array(dset)

        if random is not None:
            rng = np.random.default_rng()
            samples = rng.choice(samples, random, replace=False, axis=0, shuffle=False)

        if "n_param" in dset_attrs:
            n_param = dset_attrs["n_param"]
        elif "nparam" in dset_attrs:
            n_param = dset_attrs["nparam"]

        if "model_type" in dset_attrs:
            model_type = dset_attrs["model_type"]
        else:
            model_type = dset_attrs["type"]

        if "model_name" in dset_attrs:
            model_name = dset_attrs["model_name"]
        else:
            model_name = dset_attrs["spectrum"]

        if "binary" in dset_attrs:
            binary = dset_attrs["binary"]
        else:
            binary = False

        if "parallax" in dset_attrs:
            parallax = dset_attrs["parallax"]
        else:
            parallax = None

        if "distance" in dset_attrs:
            distance = dset_attrs["distance"]
        else:
            distance = None

        param = []
        for i in range(n_param):
            param.append(dset_attrs[f"parameter{i}"])

        if model_type in ["model", "atmosphere"]:
            if model_name == "powerlaw":
                from species.phot.syn_phot import SyntheticPhotometry

                synphot = SyntheticPhotometry(filter_name)
                synphot.zero_point()  # Set the wavel_range attribute

            else:
                from species.read.read_model import ReadModel

                readmodel = ReadModel(model_name, filter_name=filter_name)

        elif model_type == "calibration":
            from species.read.read_calibration import ReadCalibration

            readcalib = ReadCalibration(model_name, filter_name=filter_name)

        mcmc_phot = np.zeros((samples.shape[0]))

        for i in tqdm(range(samples.shape[0]), desc="Getting MCMC photometry"):
            model_param = {}

            for j in range(n_param):
                model_param[param[j]] = samples[i, j]

            if (
                "parallax" not in model_param
                and "parallax_0" not in model_param
                and parallax is not None
            ):
                model_param["parallax"] = parallax

            elif "distance" not in model_param and distance is not None:
                model_param["distance"] = distance

            if "ext_av" in model_param and "ext_model" in dset_attrs:
                model_param["ext_model"] = dset_attrs["ext_model"]

            if model_type in ["model", "atmosphere"]:
                if model_name == "powerlaw":
                    from species.util.model_util import powerlaw_spectrum

                    pl_box = powerlaw_spectrum(synphot.wavel_range, model_param)

                    if phot_type == "magnitude":
                        app_mag, _ = synphot.spectrum_to_magnitude(
                            pl_box.wavelength, pl_box.flux
                        )
                        mcmc_phot[i] = app_mag[0]

                    elif phot_type == "flux":
                        mcmc_phot[i], _ = synphot.spectrum_to_flux(
                            pl_box.wavelength, pl_box.flux
                        )

                else:
                    if phot_type == "magnitude":
                        if binary:
                            from species.util.model_util import binary_to_single

                            param_0 = binary_to_single(model_param, 0)
                            mcmc_phot_0, _ = readmodel.get_magnitude(param_0)

                            param_1 = binary_to_single(model_param, 1)
                            mcmc_phot_1, _ = readmodel.get_magnitude(param_1)

                            # Weighted flux of two spectra for atmospheric asymmetries
                            # Or simply the same in case of an actual binary system

                            if "spec_weight" in model_param:
                                mcmc_phot[i] = (
                                    model_param["spec_weight"] * mcmc_phot_0
                                    + (1.0 - model_param["spec_weight"]) * mcmc_phot_1
                                )

                            else:
                                mcmc_phot[i] = mcmc_phot_0 + mcmc_phot_1

                        else:
                            mcmc_phot[i], _ = readmodel.get_magnitude(model_param)

                    elif phot_type == "flux":
                        if binary:
                            from species.util.model_util import binary_to_single

                            param_0 = binary_to_single(model_param, 0)
                            mcmc_phot_0, _ = readmodel.get_flux(param_0)

                            param_1 = binary_to_single(model_param, 1)
                            mcmc_phot_1, _ = readmodel.get_flux(param_1)

                            # Weighted flux of two spectra for atmospheric asymmetries
                            # Or simply the same in case of an actual binary system

                            if "spec_weight" in model_param:
                                mcmc_phot[i] = (
                                    model_param["spec_weight"] * mcmc_phot_0
                                    + (1.0 - model_param["spec_weight"]) * mcmc_phot_1
                                )

                            else:
                                mcmc_phot[i] = mcmc_phot_0 + mcmc_phot_1

                        else:
                            mcmc_phot[i], _ = readmodel.get_flux(model_param)

            elif model_type == "calibration":
                if phot_type == "magnitude":
                    app_mag, _ = readcalib.get_magnitude(model_param=model_param)
                    mcmc_phot[i] = app_mag[0]

                elif phot_type == "flux":
                    mcmc_phot[i], _ = readcalib.get_flux(model_param=model_param)

        if phot_type == "flux":
            from species.util.data_util import convert_units

            read_filt = ReadFilter(filter_name)
            filt_wavel = read_filt.mean_wavelength()

            wavel_ones = np.full(mcmc_phot.size, filt_wavel)
            data_in = np.column_stack([wavel_ones, mcmc_phot])

            data_out = convert_units(data_in, ("um", flux_units), convert_from=False)

            mcmc_phot = data_out[:, 1]

        return mcmc_phot

    @typechecked
    def get_object(
        self,
        object_name: str,
        inc_phot: Union[bool, List[str]] = True,
        inc_spec: Union[bool, List[str]] = True,
        verbose: bool = True,
    ) -> ObjectBox:
        """
        Function for extracting the photometric and/or spectroscopic
        data of an object from the database. The spectroscopic data
        contains optionally the covariance matrix and its inverse.

        Parameters
        ----------
        object_name : str
            Object name in the database.
        inc_phot : bool, list(str)
            Include photometric data. If a boolean, either all
            (``True``) or none (``False``) of the data are selected.
            If a list, a subset of filter names (as stored in the
            database) can be provided.
        inc_spec : bool, list(str)
            Include spectroscopic data. If a boolean, either all
            (``True``) or none (``False``) of the data are selected.
            If a list, a subset of spectrum names (as stored in the
            database with
            :func:`~species.data.database.Database.add_object`) can
            be provided.
        verbose : bool
            Print output.

        Returns
        -------
        species.core.box.ObjectBox
            Box with the object's data.
        """

        from species.read.read_filter import ReadFilter

        if verbose:
            print_section(f"Get object")

            print(f"Object name: {object_name}")
            print(f"Include photometry: {inc_phot}")
            print(f"Include spectra: {inc_spec}")

        with h5py.File(self.database, "r") as hdf5_file:
            if f"objects/{object_name}" not in hdf5_file:
                raise ValueError(
                    "The argument of  'object_name' is "
                    f"set to '{object_name}' but the "
                    "data is not found in the database."
                )

            dset = hdf5_file[f"objects/{object_name}"]

            if "parallax" in dset:
                parallax = np.asarray(dset["parallax"])
            else:
                parallax = None

            if "distance" in dset:
                distance = np.asarray(dset["distance"])
            else:
                distance = None

            if inc_phot:
                magnitude = {}
                flux = {}
                mean_wavel = {}
                filter_width = {}

                for observatory in dset.keys():
                    if observatory not in ["parallax", "distance", "spectrum"]:
                        for filter_name in dset[observatory]:
                            name = f"{observatory}/{filter_name}"

                            if isinstance(inc_phot, bool) or name in inc_phot:
                                magnitude[name] = dset[name][0:2]
                                flux[name] = dset[name][2:4]

                phot_filters = list(magnitude.keys())

            else:
                magnitude = None
                flux = None
                phot_filters = None
                mean_wavel = None
                filter_width = None

            if inc_spec and f"objects/{object_name}/spectrum" in hdf5_file:
                spectrum = {}

                for item in hdf5_file[f"objects/{object_name}/spectrum"]:
                    data_group = f"objects/{object_name}/spectrum/{item}"

                    if isinstance(inc_spec, bool) or item in inc_spec:
                        if f"{data_group}/covariance" not in hdf5_file:
                            spectrum[item] = (
                                np.asarray(hdf5_file[f"{data_group}/spectrum"]),
                                None,
                                None,
                                hdf5_file[f"{data_group}"].attrs["specres"],
                            )

                        else:
                            spectrum[item] = (
                                np.asarray(hdf5_file[f"{data_group}/spectrum"]),
                                np.asarray(hdf5_file[f"{data_group}/covariance"]),
                                np.asarray(hdf5_file[f"{data_group}/inv_covariance"]),
                                hdf5_file[f"{data_group}"].attrs["specres"],
                            )

            else:
                spectrum = None

        if magnitude is not None:
            for filter_name in magnitude.keys():
                read_filt = ReadFilter(filter_name)
                mean_wavel[filter_name] = read_filt.mean_wavelength()
                filter_width[filter_name] = read_filt.filter_fwhm()

        return create_box(
            "object",
            name=object_name,
            filters=phot_filters,
            mean_wavel=mean_wavel,
            filter_width=filter_width,
            magnitude=magnitude,
            flux=flux,
            spectrum=spectrum,
            parallax=parallax,
            distance=distance,
        )

    @typechecked
    def get_samples(
        self,
        tag: str,
        random: Optional[int] = None,
        json_file: Optional[str] = None,
    ) -> SamplesBox:
        """
        Parameters
        ----------
        tag: str
            Database tag with the samples.
        random : int, None
            Number of random samples to select. All samples are
            selected if the argument is set to ``None``.
        json_file : str, None
            JSON file to store the posterior samples. The data will
            not be written if the argument is set to ``None``.

        Returns
        -------
        species.core.box.SamplesBox
            Box with the posterior samples.
        """

        print_section("Get posterior samples")

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/fit/{tag}/samples"]
            ln_prob = np.asarray(hdf5_file[f"results/fit/{tag}/ln_prob"])

            samples = np.asarray(dset)

            if random is not None:
                indices = np.random.randint(samples.shape[0], size=random)
                samples = samples[indices, :]

            print(f"Database tag: {tag}")
            print(f"Random samples: {random}")
            print(f"Samples shape: {samples.shape}")

            attributes = {}
            for item in dset.attrs:
                attributes[item] = dset.attrs[item]

            if "model_name" in dset.attrs:
                model_name = dset.attrs["model_name"]
            else:
                model_name = dset.attrs["spectrum"]

            if "n_param" in dset.attrs:
                n_param = dset.attrs["n_param"]
            elif "nparam" in dset.attrs:
                n_param = dset.attrs["nparam"]

            if "ln_evidence" in dset.attrs:
                ln_evidence = dset.attrs["ln_evidence"]
            else:
                ln_evidence = None

            param = []
            print("\nParameters:")
            for i in range(n_param):
                param.append(dset.attrs[f"parameter{i}"])
                print(f"   - {param[-1]}")

            # Printing uniform and normal priors

            uniform_priors = {}
            normal_priors = {}

            if "n_bounds" in attributes and attributes["n_bounds"] > 0:
                dset_bounds = hdf5_file[f"results/fit/{tag}/bounds"]
                print("\nUniform priors (min, max):")

                for bound_item in dset_bounds:
                    group_path = f"results/fit/{tag}/bounds/{bound_item}"

                    if isinstance(hdf5_file[group_path], h5py._hl.group.Group):
                        for filter_item in hdf5_file[group_path]:
                            # This is required for the inflation parameter
                            # of the photometric fluxes. Since the SVO
                            # filter names contain a slash which introduces
                            # a subgroup in the HDF5 database
                            prior_bound = np.array(
                                hdf5_file[f"{group_path}/{filter_item}"]
                            )
                            print(
                                f"   - {bound_item}/{filter_item} = "
                                f"({prior_bound[0]}, {prior_bound[1]})"
                            )
                            uniform_priors[f"{bound_item}/{filter_item}"] = (
                                prior_bound[0],
                                prior_bound[1],
                            )

                    else:
                        prior_bound = np.array(hdf5_file[group_path])
                        print(
                            f"   - {bound_item} = ({prior_bound[0]}, {prior_bound[1]})"
                        )
                        uniform_priors[bound_item] = (prior_bound[0], prior_bound[1])

            if "n_normal_prior" in attributes and attributes["n_normal_prior"] > 0:
                dset_prior = hdf5_file[f"results/fit/{tag}/normal_prior"]
                print("\nNormal priors (mean, sigma):")

                for prior_item in dset_prior:
                    group_path = f"results/fit/{tag}/normal_prior/{prior_item}"

                    if isinstance(hdf5_file[group_path], h5py._hl.group.Group):
                        for filter_item in hdf5_file[group_path]:
                            # This is required for the inflation parameter
                            # of the photometric fluxes. Since the SVO
                            # filter names contain a slash which introduces
                            # a subgroup in the HDF5 database
                            norm_prior = np.array(
                                hdf5_file[f"{group_path}/{filter_item}"]
                            )

                            if -0.1 < norm_prior[0] < 0.1:
                                print(
                                    f"   - {prior_item}/{filter_item} = "
                                    f"({norm_prior[0]:.2e}, {norm_prior[1]:.2e})"
                                )

                            else:
                                print(
                                    f"   - {prior_item}/{filter_item} = "
                                    f"({norm_prior[0]:.2f}, {norm_prior[1]:.2f})"
                                )

                            normal_priors[f"{prior_item}/{filter_item}"] = (
                                norm_prior[0],
                                norm_prior[1],
                            )

                    else:
                        norm_prior = np.array(hdf5_file[group_path])

                        if -0.1 < norm_prior[0] < 0.1:
                            print(
                                f"   - {prior_item} = ({norm_prior[0]:.2e}, {norm_prior[1]:.2e})"
                            )
                        else:
                            print(
                                f"   - {prior_item} = ({norm_prior[0]:.2f}, {norm_prior[1]:.2f})"
                            )

                        normal_priors[prior_item] = (norm_prior[0], norm_prior[1])

        median_sample = self.get_median_sample(tag, verbose=False)
        prob_sample = self.get_probable_sample(tag, verbose=False)

        if json_file is not None:
            samples_dict = {}

            for i, item in enumerate(param):
                samples_dict[item] = list(samples[:, i])

            with open(json_file, "w", encoding="utf-8") as out_file:
                json.dump(samples_dict, out_file, indent=4)

            print(f"\nOutput: {json_file}")

        return create_box(
            "samples",
            model_name=model_name,
            parameters=param,
            samples=samples,
            ln_prob=ln_prob,
            ln_evidence=ln_evidence,
            prob_sample=prob_sample,
            median_sample=median_sample,
            attributes=attributes,
            uniform_priors=uniform_priors,
            normal_priors=normal_priors,
        )

    @typechecked
    def get_evidence(self, tag: str) -> Tuple[float, float]:
        """
        Function for returning the log-evidence (i.e.
        marginalized likelihood) that was computed by
        the nested sampling algorithm when using
        :class:`~species.fit.fit_model.FitModel` or
        :class:`~species.fit.retrieval.AtmosphericRetrieval`.

        Parameters
        ----------
        tag: str
            Database tag with the posterior samples.

        Returns
        -------
        float
            Log-evidence.
        float
            Uncertainty on the log-evidence.
        """

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/fit/{tag}/samples"]

            if "ln_evidence" in dset.attrs:
                ln_evidence = dset.attrs["ln_evidence"]
            else:
                ln_evidence = (None, None)

        return ln_evidence[0], ln_evidence[1]

    @typechecked
    def get_pt_profiles(
        self, tag: str, random: Optional[int] = None, out_file: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function for returning the pressure-temperature profiles
        from the posterior of the atmospheric retrieval with
        ``petitRADTRANS``. The data can also optionally be
        written to an output file.

        Parameters
        ----------
        tag: str
            Database tag with the posterior samples from the
            atmospheric retrieval with
            :class:`~species.fit.retrieval.AtmosphericRetrieval`.
        random : int, None
            Number of random samples that will be used for the P-T
            profiles. All samples will be selected if set to ``None``.
        out_file : str, None
            Output file to store the P-T profiles. The data will be
            stored in a FITS file if the argument of ``out_file`` ends
            with `.fits`. Otherwise, the data will be written to a
            text file. The data has two dimensions with the first
            column containing the pressures (bar) and the remaining
            columns the temperature profiles (K). The data will not
            be written to a file if the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array (1D) with the pressures (bar).
        np.ndarray
            Array (2D) with the temperature profiles (K). The shape
            of the array is (n_pressures, n_samples).
        """

        from species.util.retrieval_util import (
            pt_ret_model,
            pt_spline_interp,
        )

        hdf5_file = h5py.File(self.database, "r")
        dset = hdf5_file[f"results/fit/{tag}/samples"]

        if "model_name" in dset.attrs:
            model_name = dset.attrs["model_name"]
        else:
            model_name = dset.attrs["spectrum"]

        pt_profile = dset.attrs["pt_profile"]

        if model_name != "petitradtrans":
            raise ValueError(
                f"The model spectrum of the posterior samples is '{model_name}' "
                f"instead of 'petitradtrans'. Extracting P-T profiles is "
                f"therefore not possible."
            )

        if "n_param" in dset.attrs:
            n_param = dset.attrs["n_param"]
        elif "nparam" in dset.attrs:
            n_param = dset.attrs["nparam"]

        if "temp_nodes" in dset.attrs:
            temp_nodes = dset.attrs["temp_nodes"]
        else:
            temp_nodes = 15

        samples = np.asarray(dset)

        if random is None:
            n_profiles = samples.shape[0]

        else:
            n_profiles = random

            indices = np.random.randint(samples.shape[0], size=random)
            samples = samples[indices, :]

        param_index = {}
        for i in range(n_param):
            param_index[dset.attrs[f"parameter{i}"]] = i

        hdf5_file.close()

        press = np.logspace(-6, 3, 180)  # (bar)

        temp = np.zeros((press.shape[0], n_profiles))

        desc = f"Extracting the P-T profiles of {tag}"

        for i in tqdm(range(samples.shape[0]), desc=desc):
            item = samples[i, :]

            if pt_profile == "molliere":
                three_temp = np.array(
                    [
                        item[param_index["t1"]],
                        item[param_index["t2"]],
                        item[param_index["t3"]],
                    ]
                )

                temp[:, i], _, _ = pt_ret_model(
                    three_temp,
                    10.0 ** item[param_index["log_delta"]],
                    item[param_index["alpha"]],
                    item[param_index["tint"]],
                    press,
                    item[param_index["metallicity"]],
                    item[param_index["c_o_ratio"]],
                )

            elif pt_profile == "mod-molliere":
                temp[:, i], _, _ = pt_ret_model(
                    None,
                    10.0 ** item[param_index["log_delta"]],
                    item[param_index["alpha"]],
                    item[param_index["tint"]],
                    press,
                    item[param_index["metallicity"]],
                    item[param_index["c_o_ratio"]],
                )

            elif pt_profile == "eddington":
                # Eddington approximation
                # delta = kappa_ir/gravity
                tau = press * 1e6 * 10.0 ** item[param_index["log_delta"]]
                temp[:, i] = (
                    0.75 * item[param_index["tint"]] ** 4.0 * (2.0 / 3.0 + tau)
                ) ** 0.25

            elif pt_profile in ["free", "monotonic"]:
                if "pt_smooth" in param_index:
                    pt_smooth = item[param_index["pt_smooth"]]
                else:
                    pt_smooth = 0.0

                knot_press = np.logspace(
                    np.log10(press[0]), np.log10(press[-1]), temp_nodes
                )

                knot_temp = []
                for k in range(temp_nodes):
                    knot_temp.append(item[param_index[f"t{k}"]])

                knot_temp = np.asarray(knot_temp)

                temp[:, i] = pt_spline_interp(knot_press, knot_temp, press, pt_smooth)

        if out_file is not None:
            data = np.hstack([press[..., np.newaxis], temp])

            if out_file.endswith(".fits"):
                fits.writeto(out_file, data, overwrite=True)

            else:
                np.savetxt(out_file, data, header="Pressure (bar) - Temperature (K)")

        return press, temp

    @typechecked
    def add_empirical(
        self,
        tag: str,
        names: List[str],
        sptypes: List[str],
        goodness_of_fit: List[float],
        flux_scaling: List[np.ndarray],
        av_ext: List[float],
        rad_vel: List[float],
        object_name: str,
        spec_name: List[str],
        spec_library: str,
    ) -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag where the results will be stored.
        names : list(str)
            Array with the names of the empirical spectra.
        sptypes : list(str)
            Array with the spectral types of ``names``.
        goodness_of_fit : list(float)
            Array with the goodness-of-fit values.
        flux_scaling : list(np.ndarray)
            List with arrays with the best-fit scaling values to match the library spectra with
            the data. The size of each array is equal to the number of spectra that are provided
            as argument of ``spec_name``.
        av_ext : list(float)
            Array with the visual extinction :math:`A_V`.
        rad_vel : list(float)
            Array with the radial velocities (km s-1).
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : list(str)
            List with spectrum names that are stored at the object data of ``object_name``.
        spec_library : str
            Name of the spectral library that was used for the empirical comparison.
        Returns
        -------
        NoneType
            None
        """

        with h5py.File(self.database, "a") as hdf5_file:
            if "results" not in hdf5_file:
                hdf5_file.create_group("results")

            if "results/empirical" not in hdf5_file:
                hdf5_file.create_group("results/empirical")

            if f"results/empirical/{tag}" in hdf5_file:
                del hdf5_file[f"results/empirical/{tag}"]

            dtype = h5py.special_dtype(vlen=str)

            dset = hdf5_file.create_dataset(
                f"results/empirical/{tag}/names", (np.size(names),), dtype=dtype
            )

            dset[...] = names

            dset.attrs["object_name"] = str(object_name)
            dset.attrs["spec_library"] = str(spec_library)
            dset.attrs["n_spec_name"] = len(spec_name)

            for i, item in enumerate(spec_name):
                dset.attrs[f"spec_name{i}"] = item

            dset = hdf5_file.create_dataset(
                f"results/empirical/{tag}/sptypes", (np.size(sptypes),), dtype=dtype
            )

            dset[...] = sptypes

            hdf5_file.create_dataset(
                f"results/empirical/{tag}/goodness_of_fit",
                data=goodness_of_fit,
                dtype="f8",
            )
            hdf5_file.create_dataset(
                f"results/empirical/{tag}/flux_scaling", data=flux_scaling, dtype="f8"
            )
            hdf5_file.create_dataset(
                f"results/empirical/{tag}/av_ext", data=av_ext, dtype="f8"
            )
            hdf5_file.create_dataset(
                f"results/empirical/{tag}/rad_vel", data=rad_vel, dtype="f8"
            )

    @typechecked
    def add_comparison(
        self,
        tag: str,
        goodness_of_fit: np.ndarray,
        flux_scaling: np.ndarray,
        model_param: List[str],
        coord_points: List[np.ndarray],
        object_name: str,
        spec_name: List[str],
        model_name: str,
        scale_spec: List[str],
        extra_scaling: Optional[np.ndarray],
        inc_phot: List[str],
    ) -> None:
        """
        Function for adding results obtained with
        :class:`~species.fit.compare_spectra.CompareSpectra`
        to the HDF5 database.

        Parameters
        ----------
        tag : str
            Database tag where the results will be stored.
        goodness_of_fit : np.ndarray
            Array with the goodness-of-fit values.
        flux_scaling : np.ndarray
            Array with the best-fit scaling values to match the model
            spectra with the data.
        model_param : list(str)
            List with the names of the model parameters.
        coord_points : list(np.ndarray)
            List with 1D arrays of the model grid points, in the same
            order as ``model_param``.
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : list(str)
            List with spectrum names that are stored at the object
            data of ``object_name``.
        model_name : str
            Atmospheric model grid that is used for the comparison.
        scale_spec : list(str)
            List with spectrum names to which an additional scaling
            has been applied.
        extra_scaling : np.ndarray. None
            Array with extra scalings that have been applied to the
            spectra of ``scale_spec``. The argument can be set to
            ``None`` if no extra scalings have been applied.
        inc_phot : list(str)
            List with filter names of which photometric data
            was included with the comparison.

        Returns
        -------
        NoneType
            None
        """

        from species.read.read_object import ReadObject

        read_obj = ReadObject(object_name)
        parallax = read_obj.get_parallax()[0]  # (mas)

        with h5py.File(self.database, "a") as hdf5_file:
            if "results" not in hdf5_file:
                hdf5_file.create_group("results")

            if "results/comparison" not in hdf5_file:
                hdf5_file.create_group("results/comparison")

            if f"results/comparison/{tag}" in hdf5_file:
                del hdf5_file[f"results/comparison/{tag}"]

            dset = hdf5_file.create_dataset(
                f"results/comparison/{tag}/goodness_of_fit",
                data=goodness_of_fit,
                dtype="f8",
            )

            dset.attrs["object_name"] = str(object_name)
            dset.attrs["model_name"] = str(model_name)
            dset.attrs["n_param"] = len(model_param)
            dset.attrs["n_spec_name"] = len(spec_name)
            dset.attrs["n_scale_spec"] = len(scale_spec)
            dset.attrs["parallax"] = parallax
            dset.attrs["n_inc_phot"] = len(inc_phot)

            for i, item in enumerate(model_param):
                dset.attrs[f"parameter{i}"] = item

            for i, item in enumerate(spec_name):
                dset.attrs[f"spec_name{i}"] = item

            for i, item in enumerate(scale_spec):
                dset.attrs[f"scale_spec{i}"] = item

            for i, item in enumerate(inc_phot):
                dset.attrs[f"inc_phot{i}"] = item

            hdf5_file.create_dataset(
                f"results/comparison/{tag}/flux_scaling", data=flux_scaling, dtype="f8"
            )

            if len(scale_spec) > 0:
                hdf5_file.create_dataset(
                    f"results/comparison/{tag}/extra_scaling",
                    data=extra_scaling,
                    dtype="f8",
                )

            for i, item in enumerate(coord_points):
                hdf5_file.create_dataset(
                    f"results/comparison/{tag}/coord_points{i}", data=item, dtype="f8"
                )

            # Indices of the best-fit model
            best_index = np.unravel_index(
                np.nanargmin(goodness_of_fit), goodness_of_fit.shape
            )
            dset.attrs["best_fit"] = goodness_of_fit[best_index]

            print("Best-fit parameters:")
            print(f"   - Goodness-of-fit = {goodness_of_fit[best_index]:.2e}")

            for param_idx, param_item in enumerate(model_param):
                best_param = coord_points[param_idx][best_index[param_idx]]
                dset.attrs[f"best_param{param_idx}"] = best_param
                print(f"   - {param_item} = {best_param}")

            scaling = flux_scaling[best_index]

            radius = np.sqrt(scaling * (1e3 * constants.PARSEC / parallax) ** 2)  # (m)
            radius /= constants.R_JUP  # (Rjup)

            dset.attrs["radius"] = radius
            print(f"   - Radius (Rjup) = {radius:.2f}")

            dset.attrs["scaling"] = scaling
            print(f"   - Scaling = {scaling:.2e}")

            for spec_idx, spec_item in enumerate(scale_spec):
                scaling_idx = np.append(best_index, spec_idx)
                scale_tmp = extra_scaling[tuple(scaling_idx)]
                dset.attrs[f"scaling_{spec_item}"] = scale_tmp
                print(f"   - {spec_item} scaling = {scale_tmp:.2f}")

    def add_retrieval(
        self, tag: str, output_folder: str, inc_teff: bool = False
    ) -> None:
        """
        Function for adding the output data from
        the atmospheric retrieval with
        :class:`~species.fit.retrieval.AtmosphericRetrieval`
        to the database.

        Parameters
        ----------
        tag : str
            Database tag to store the posterior samples.
        output_folder : str
            Output folder that was used for the output files by
            ``MultiNest``.
        inc_teff : bool
            Calculate :math:`T_\\mathrm{eff}` for each sample by
            integrating the model spectrum from 0.5 to 50 um. The
            :math:`T_\\mathrm{eff}` samples are added to the array
            with samples that are stored in the database. The
            computation time for adding :math:`T_\\mathrm{eff}` will
            be long because the spectra need to be calculated and
            integrated for all samples.

        Returns
        -------
        NoneType
            None
        """

        from species.util.retrieval_util import (
            list_to_dict,
            pt_ret_model,
            pt_spline_interp,
            quench_pressure,
            scale_cloud_abund,
        )

        print("Storing samples in the database...", end="", flush=True)

        json_filename = os.path.join(output_folder, "params.json")

        with open(json_filename, encoding="utf-8") as json_file:
            parameters = json.load(json_file)

        radtrans_filename = os.path.join(output_folder, "radtrans.json")

        with open(radtrans_filename, encoding="utf-8") as json_file:
            radtrans = json.load(json_file)

        post_new = os.path.join(output_folder, "retrieval_post_equal_weights.dat")
        post_old = os.path.join(output_folder, "post_equal_weights.dat")

        if os.path.exists(post_new):
            samples = np.loadtxt(post_new)

        elif os.path.exists(post_old):
            samples = np.loadtxt(post_old)

        else:
            raise RuntimeError("Can not find the post_equal_weights.dat file.")

        if samples.ndim == 1:
            warnings.warn(
                f"Only 1 sample found in post_equal_weights.dat "
                f"of the '{output_folder}' folder."
            )

            samples = samples[np.newaxis,]

        # Number of fixed parameters

        fixed_param = {}
        for i in range(samples.shape[1]):
            if np.amin(samples[:, i]) == np.amax(samples[:, i]):
                fixed_param[parameters[i]] = np.mean(samples[:, i])

        with h5py.File(self.database, "a") as hdf5_file:
            if "results" not in hdf5_file:
                hdf5_file.create_group("results")

            if "results/fit" not in hdf5_file:
                hdf5_file.create_group("results/fit")

            if f"results/fit/{tag}" in hdf5_file:
                del hdf5_file[f"results/fit/{tag}"]

            # Store the log-likelihood
            hdf5_file.create_dataset(
                f"results/fit/{tag}/ln_prob", data=samples[:, -1], dtype="f8"
            )

            # Remove the column with the log-likelihood value
            samples = samples[:, :-1]

            if samples.shape[1] != len(parameters):
                raise ValueError(
                    "The number of parameters is not equal to the parameter size "
                    "of the samples array."
                )

            dset = hdf5_file.create_dataset(
                f"results/fit/{tag}/samples", data=samples, dtype="f8"
            )

            dset.attrs["model_type"] = "retrieval"
            dset.attrs["model_name"] = "petitradtrans"
            dset.attrs["n_param"] = len(parameters)

            if "parallax" in radtrans:
                dset.attrs["parallax"] = radtrans["parallax"]
            else:
                dset.attrs["distance"] = radtrans["distance"]

            count_scale = 0
            count_error = 0

            for i, item in enumerate(parameters):
                dset.attrs[f"parameter{i}"] = item

            for i, item in enumerate(parameters):
                if item[0:6] == "scaling_":
                    dset.attrs[f"scaling{count_scale}"] = item
                    count_scale += 1

            for i, item in enumerate(parameters):
                if item[0:6] == "error_":
                    dset.attrs[f"error{count_error}"] = item
                    count_error += 1

            dset.attrs["n_scaling"] = count_scale
            dset.attrs["n_error"] = count_error

            for i, item in enumerate(radtrans["line_species"]):
                dset.attrs[f"line_species{i}"] = item

            for i, item in enumerate(radtrans["cloud_species"]):
                dset.attrs[f"cloud_species{i}"] = item

            dset.attrs["n_line_species"] = len(radtrans["line_species"])
            dset.attrs["n_cloud_species"] = len(radtrans["cloud_species"])

            dset.attrs["scattering"] = radtrans["scattering"]
            dset.attrs["pressure_grid"] = radtrans["pressure_grid"]
            dset.attrs["pt_profile"] = radtrans["pt_profile"]
            dset.attrs["chemistry"] = radtrans["chemistry"]
            dset.attrs["wavel_min"] = radtrans["wavel_range"][0]
            dset.attrs["wavel_max"] = radtrans["wavel_range"][1]

            if "quenching" not in radtrans or radtrans["quenching"] is None:
                dset.attrs["quenching"] = "None"
            else:
                dset.attrs["quenching"] = radtrans["quenching"]

            if "temp_nodes" not in radtrans or radtrans["temp_nodes"] is None:
                dset.attrs["temp_nodes"] = "None"
                temp_nodes = 15
            else:
                dset.attrs["temp_nodes"] = radtrans["temp_nodes"]
                temp_nodes = radtrans["temp_nodes"]

            if "pt_smooth" in radtrans:
                dset.attrs["pt_smooth"] = radtrans["pt_smooth"]

            if "max_press" in radtrans:
                dset.attrs["max_press"] = radtrans["max_press"]

            if "abund_nodes" not in radtrans or radtrans["abund_nodes"] is None:
                dset.attrs["abund_nodes"] = "None"
            else:
                dset.attrs["abund_nodes"] = radtrans["abund_nodes"]

            if "res_mode" in radtrans:
                dset.attrs["res_mode"] = radtrans["res_mode"]
            else:
                dset.attrs["res_mode"] = "c-k"

            if (
                "lbl_opacity_sampling" not in radtrans
                or radtrans["lbl_opacity_sampling"] is None
            ):
                dset.attrs["abund_nodes"] = "None"
            else:
                dset.attrs["lbl_opacity_sampling"] = radtrans["lbl_opacity_sampling"]

            dset.attrs["n_fixed"] = int(len(fixed_param))

            for key, value in fixed_param.items():
                group_path = f"results/fit/{tag}/fixed_param/{key}"
                hdf5_file.create_dataset(group_path, data=value, dtype="f8")

        print(" [DONE]")

        # Set number of pressures

        if radtrans["pressure_grid"] in ["standard", "smaller"]:
            n_pressures = 180

        elif radtrans["pressure_grid"] == "clouds":
            n_pressures = 1440

        rt_object = None

        from petitRADTRANS.chemistry.pre_calculated_chemistry import (
            PreCalculatedEquilibriumChemistryTable,
        )

        eq_chem = PreCalculatedEquilibriumChemistryTable()
        eq_chem.load()

        for i, cloud_item in enumerate(radtrans["cloud_species"]):
            if f"{cloud_item}_tau" in parameters:
                pressure = np.logspace(-6, 3, n_pressures)
                cloud_mass = np.zeros(samples.shape[0])

                if rt_object is None:
                    from petitRADTRANS.radtrans import Radtrans

                    # Pressure array for Radtrans

                    if self.pressure_grid in ["standard", "manual"]:
                        radtrans_press = np.copy(pressure)

                    elif self.pressure_grid == "smaller":
                        radtrans_press = pressure[::3]

                    elif self.pressure_grid == "clouds":
                        radtrans_press = pressure[::24]

                    rt_object = Radtrans(
                        pressures=radtrans_press,
                        line_species=radtrans["line_species"],
                        rayleigh_species=["H2", "He"],
                        cloud_species=radtrans["cloud_species"].copy(),
                        gas_continuum_contributors=["H2-H2", "H2-He"],
                        wavelength_boundaries=radtrans["wavel_range"],
                        line_opacity_mode="c-k",
                        scattering_in_emission=radtrans["scattering"],
                    )

                desc = f"Calculating mass fractions of {cloud_item}"

                for j in tqdm(range(samples.shape[0]), desc=desc):
                    sample_dict = list_to_dict(
                        parameters,
                        samples[j,],
                    )

                    if radtrans["pt_profile"] == "molliere":
                        upper_temp = np.array(
                            [sample_dict["t1"], sample_dict["t2"], sample_dict["t3"]]
                        )

                        temp, _, _ = pt_ret_model(
                            upper_temp,
                            10.0 ** sample_dict["log_delta"],
                            sample_dict["alpha"],
                            sample_dict["tint"],
                            pressure,
                            sample_dict["metallicity"],
                            sample_dict["c_o_ratio"],
                        )

                    elif (
                        radtrans["pt_profile"] == "free"
                        or radtrans["pt_profile"] == "monotonic"
                    ):
                        knot_press = np.logspace(
                            np.log10(pressure[0]), np.log10(pressure[-1]), temp_nodes
                        )

                        knot_temp = []
                        for k in range(temp_nodes):
                            knot_temp.append(sample_dict[f"t{k}"])

                        knot_temp = np.asarray(knot_temp)

                        pt_smooth = sample_dict.get("pt_smooth", radtrans["pt_smooth"])

                        temp = pt_spline_interp(
                            knot_press, knot_temp, pressure, pt_smooth=pt_smooth
                        )

                    # Set the quenching pressure (bar)

                    if "log_p_quench" in parameters:
                        quench_press = 10.0 ** sample_dict["log_p_quench"]
                    else:
                        quench_press = None

                    abund_in = eq_chem.interpolate_mass_fractions(
                        np.full(pressure.shape[0], sample_dict["c_o_ratio"]),
                        np.full(pressure.shape[0], sample_dict["metallicity"]),
                        temp,
                        pressure,
                        carbon_pressure_quench=quench_press,
                        full=False,
                    )

                    # Calculate the scaled mass fraction of the clouds

                    cloud_mass[j] = scale_cloud_abund(
                        sample_dict,
                        rt_object,
                        pressure,
                        temp,
                        abund_in["MMW"],
                        "equilibrium",
                        abund_in,
                        cloud_item,
                        sample_dict[f"{cloud_item}_tau"],
                        pressure_grid=radtrans["pressure_grid"],
                    )

                db_tag = f"results/fit/{tag}/samples"

                with h5py.File(self.database, "a") as hdf5_file:
                    dset_attrs = hdf5_file[db_tag].attrs

                    samples = np.asarray(hdf5_file[db_tag])
                    samples = np.append(samples, cloud_mass[..., np.newaxis], axis=1)

                    del hdf5_file[db_tag]
                    dset = hdf5_file.create_dataset(db_tag, data=samples, dtype="f8")

                    for attr_item in dset_attrs:
                        dset.attrs[attr_item] = dset_attrs[attr_item]

                    n_param = dset_attrs["n_param"] + 1

                    dset.attrs["n_param"] = n_param

        if radtrans["quenching"] == "diffusion":
            p_quench = np.zeros(samples.shape[0])

            desc = "Calculating quenching pressures"

            for i in tqdm(range(samples.shape[0]), desc=desc):
                # Convert list of parameters and samples into dictionary
                sample_dict = list_to_dict(
                    parameters,
                    samples[i,],
                )

                # Recalculate the P-T profile from the sampled parameters

                pressure = np.logspace(-6, 3, n_pressures)  # (bar)

                if radtrans["pt_profile"] == "molliere":
                    upper_temp = np.array(
                        [sample_dict["t1"], sample_dict["t2"], sample_dict["t3"]]
                    )

                    temp, _, _ = pt_ret_model(
                        upper_temp,
                        10.0 ** sample_dict["log_delta"],
                        sample_dict["alpha"],
                        sample_dict["tint"],
                        pressure,
                        sample_dict["metallicity"],
                        sample_dict["c_o_ratio"],
                    )

                elif (
                    radtrans["pt_profile"] == "free"
                    or radtrans["pt_profile"] == "monotonic"
                ):
                    knot_press = np.logspace(
                        np.log10(pressure[0]), np.log10(pressure[-1]), temp_nodes
                    )

                    knot_temp = []
                    for k in range(temp_nodes):
                        knot_temp.append(sample_dict[f"t{k}"])

                    knot_temp = np.asarray(knot_temp)

                    if "pt_smooth" in sample_dict:
                        pt_smooth = sample_dict["pt_smooth"]
                    else:
                        pt_smooth = radtrans["pt_smooth"]

                    temp = pt_spline_interp(
                        knot_press, knot_temp, pressure, pt_smooth=pt_smooth
                    )

                # Calculate the quenching pressure

                p_quench[i] = quench_pressure(
                    pressure,
                    temp,
                    sample_dict["metallicity"],
                    sample_dict["c_o_ratio"],
                    sample_dict["logg"],
                    sample_dict["log_kzz"],
                )

            db_tag = f"results/fit/{tag}/samples"

            with h5py.File(self.database, "a") as hdf5_file:
                dset_attrs = hdf5_file[db_tag].attrs

                samples = np.asarray(hdf5_file[db_tag])
                samples = np.append(
                    samples, np.log10(p_quench[..., np.newaxis]), axis=1
                )

                del hdf5_file[db_tag]
                dset = hdf5_file.create_dataset(db_tag, data=samples, dtype="f8")

                for item in dset_attrs:
                    dset.attrs[item] = dset_attrs[item]

                n_param = dset_attrs["n_param"] + 1

                dset.attrs["n_param"] = n_param
                dset.attrs[f"parameter{n_param-1}"] = "log_p_quench"

        if inc_teff:
            print("Calculating Teff from the posterior samples... ")

            boxes, _ = self.get_retrieval_spectra(
                tag=tag, random=None, wavel_range=(0.5, 50.0), spec_res=100.0
            )

            teff = np.zeros(len(boxes))

            for i, box_item in enumerate(boxes):
                if "parallax" in box_item.parameters:
                    sample_distance = (
                        1e3 * constants.PARSEC / box_item.parameters["parallax"]
                    )
                else:
                    sample_distance = box_item.parameters["distance"] * constants.PARSEC

                sample_radius = box_item.parameters["radius"] * constants.R_JUP

                # Scaling for the flux back to the planet surface
                sample_scale = (sample_distance / sample_radius) ** 2

                # Blackbody flux: sigma * Teff^4
                flux_int = simpson(sample_scale * box_item.flux, x=box_item.wavelength)
                teff[i] = (flux_int / constants.SIGMA_SB) ** 0.25

            db_tag = f"results/fit/{tag}/samples"

            with h5py.File(self.database, "a") as hdf5_file:
                dset_attrs = hdf5_file[db_tag].attrs

                samples = np.asarray(hdf5_file[db_tag])
                samples = np.append(samples, teff[..., np.newaxis], axis=1)

                del hdf5_file[db_tag]
                dset = hdf5_file.create_dataset(db_tag, data=samples, dtype="f8")

                for item in dset_attrs:
                    dset.attrs[item] = dset_attrs[item]

                n_param = dset_attrs["n_param"] + 1

                dset.attrs["n_param"] = n_param
                dset.attrs[f"parameter{n_param-1}"] = "teff"

    @staticmethod
    @typechecked
    def get_retrieval_spectra(
        tag: str,
        random: Optional[int],
        wavel_range: Optional[Union[Tuple[float, float], str]] = None,
        spec_res: Optional[float] = None,
        lbl_opacity_sampling: Optional[Union[int, np.int64]] = None,
    ) -> Tuple[List[ModelBox], Union[Any]]:
        """
        Function for extracting random spectra from the
        posterior distribution that was sampled with
        :class:`~species.fit.retrieval.AtmosphericRetrieval`.

        Parameters
        ----------
        tag : str
            Database tag with the posterior samples.
        random : int, None
            Number of randomly selected samples. All samples
            are selected if set to ``None``. When setting ``random=0``,
            no random spectra are sampled (so the returned list
            with ``ModelBox`` objects is empty), but the
            :class:`~species.read.read_radtrans.ReadRadtrans`
            instance is still returned.
        wavel_range : tuple(float, float), str, None
            Wavelength range (um) or filter name. The
            wavelength range from the retrieval is adopted
            (i.e. the``wavel_range`` parameter of
            :class:`~species.fit.retrieval.AtmosphericRetrieval`)
            when set to ``None``. It is mandatory to set the argument
            to ``None`` in case the ``log_tau_cloud`` parameter has
            been used with the retrieval.
        spec_res : float, None
            Spectral resolution that is used for the smoothing with a
            Gaussian kernel. No smoothing is applied when the argument
            is set to ``None``.
        lbl_opacity_sampling : int, None
            This is the same parameter as in ``petitRADTRANS`` which is
            used with ``res_mode='lbl'`` to downsample the line-by-line
            opacities by selecting every ``lbl_opacity_sampling``-th
            wavelength from the original sampling of
            :math:`\\lambda/\\Delta \\lambda = 10^6`. Setting this
            parameter will lower the computation time. By setting the
            argument to ``None``, the value that was set with the
            retrieval will be used. Setting a lower value will be
            useful when calculating a low-resolution spectrum
            across a wide wavelength range, which may require too
            much memory when using the ``res_mode='lbl'`` mode.

        Returns
        -------
        list(ModelBox)
            Boxes with the randomly sampled spectra.
        species.read.read_radtrans.ReadRadtrans
            Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        """

        from species.read.read_radtrans import ReadRadtrans
        from species.util.radtrans_util import retrieval_spectrum

        # Open configuration file

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = ConfigParser()
        config.read(config_file)

        # Read path of the HDF5 database

        database_path = config["species"]["database"]

        # Open the HDF5 database

        with h5py.File(database_path, "r") as hdf5_file:
            # Read the posterior samples
            dset = hdf5_file[f"results/fit/{tag}/samples"]
            samples = np.asarray(dset)
            attrs_dict = dict(dset.attrs)

        # Select random samples

        if random is None:
            # Required for the printed output in the for loop
            random = samples.shape[0]

        else:
            random_indices = np.random.randint(samples.shape[0], size=random)
            samples = samples[random_indices, :]

        # Get number of model parameters

        if "n_param" in attrs_dict:
            n_param = attrs_dict["n_param"]
        elif "nparam" in attrs_dict:
            n_param = attrs_dict["nparam"]

        # Get number of line and cloud species

        n_line_species = attrs_dict["n_line_species"]
        n_cloud_species = attrs_dict["n_cloud_species"]

        # Get number of abundance nodes

        if "abund_nodes" in attrs_dict:
            if attrs_dict["abund_nodes"] == "None":
                abund_nodes = None
            else:
                abund_nodes = attrs_dict["abund_nodes"]
        else:
            abund_nodes = None

        # Convert numpy boolean to regular boolean

        scattering = bool(attrs_dict["scattering"])

        # Get chemistry attributes

        chemistry = attrs_dict["chemistry"]

        if attrs_dict["quenching"] == "None":
            quenching = None
        else:
            quenching = attrs_dict["quenching"]

        # Get P-T profile attributes

        pt_profile = attrs_dict["pt_profile"]

        if "pressure_grid" in attrs_dict:
            pressure_grid = attrs_dict["pressure_grid"]
        else:
            pressure_grid = "smaller"

        # Get free temperarture nodes

        if "temp_nodes" in attrs_dict:
            if attrs_dict["temp_nodes"] == "None":
                temp_nodes = None
            else:
                temp_nodes = attrs_dict["temp_nodes"]

        else:
            temp_nodes = None

        # Get distance

        if "parallax" in attrs_dict:
            distance = 1e3 / attrs_dict["parallax"][0]
        elif "distance" in attrs_dict:
            distance = attrs_dict["distance"]
        else:
            distance = None

        # Get maximum pressure

        if "max_press" in attrs_dict:
            max_press = attrs_dict["max_press"]
        else:
            max_press = None

        # Get model parameters

        parameters = []
        for i in range(n_param):
            parameters.append(attrs_dict[f"parameter{i}"])

        parameters = np.asarray(parameters)

        # Get wavelength range for median cloud optical depth

        if "log_tau_cloud" in parameters and wavel_range is not None:
            cloud_wavel = (attrs_dict["wavel_min"], attrs_dict["wavel_max"])
        else:
            cloud_wavel = None

        # Get wavelength range for spectrum

        if wavel_range is None:
            wavel_range = (attrs_dict["wavel_min"], attrs_dict["wavel_max"])

        # Create dictionary with array indices of the model parameters

        indices = {}
        for item in parameters:
            indices[item] = np.argwhere(parameters == item)[0][0]

        # Create list with line species

        line_species = []
        for i in range(n_line_species):
            line_species.append(attrs_dict[f"line_species{i}"])

        # Create list with cloud species

        cloud_species = []
        for i in range(n_cloud_species):
            cloud_species.append(attrs_dict[f"cloud_species{i}"])

        # Get resolution mode

        if "res_mode" in attrs_dict:
            res_mode = attrs_dict["res_mode"]
        else:
            res_mode = "c-k"

        # High-resolution downsampling factor

        if lbl_opacity_sampling is None:
            if "lbl_opacity_sampling" in attrs_dict:
                if attrs_dict["lbl_opacity_sampling"] == "None":
                    lbl_opacity_sampling = None
                else:
                    lbl_opacity_sampling = attrs_dict["lbl_opacity_sampling"]
            else:
                lbl_opacity_sampling = None

        # Create an instance of ReadRadtrans
        # Afterwards, the names of the cloud_species have been shortened
        # from e.g. 'MgSiO3(c)_cd' to 'MgSiO3(c)'

        read_rad = ReadRadtrans(
            line_species=line_species,
            cloud_species=cloud_species,
            scattering=scattering,
            wavel_range=wavel_range,
            pressure_grid=pressure_grid,
            cloud_wavel=cloud_wavel,
            max_press=max_press,
            res_mode=res_mode,
            lbl_opacity_sampling=lbl_opacity_sampling,
        )

        # Set quenching attribute such that the parameter of get_model is not required

        read_rad.quenching = quenching

        # pool = multiprocessing.Pool(os.cpu_count())
        # processes = []

        # Initiate empty list for ModelBox objects

        boxes = []

        for i, item in enumerate(samples):
            print(f"\rGetting posterior spectra {i+1}/{random}...", end="")

            # Get smoothing parameter for P-T profile

            if "pt_smooth" in attrs_dict:
                pt_smooth = attrs_dict["pt_smooth"]

            elif "pt_smooth_0" in parameters:
                pt_smooth = {}
                for j in range(temp_nodes - 1):
                    pt_smooth[f"pt_smooth_{j}"] = item[-1 * temp_nodes + j]

            else:
                pt_smooth = item[indices["pt_smooth"]]

            # Get smoothing parameter for abundance profiles

            if "abund_smooth" in attrs_dict:
                if attrs_dict["abund_smooth"] == "None":
                    abund_smooth = None
                else:
                    abund_smooth = attrs_dict["abund_smooth"]
            else:
                abund_smooth = None

            # Calculate the petitRADTRANS spectrum

            model_box = retrieval_spectrum(
                indices=indices,
                chemistry=chemistry,
                pt_profile=pt_profile,
                line_species=line_species,
                cloud_species=cloud_species,
                quenching=quenching,
                spec_res=spec_res,
                distance=distance,
                pt_smooth=pt_smooth,
                temp_nodes=temp_nodes,
                abund_nodes=abund_nodes,
                abund_smooth=abund_smooth,
                read_rad=read_rad,
                sample=item,
            )

            # Add the ModelBox to the list

            boxes.append(model_box)

            # proc = pool.apply_async(retrieval_spectrum,
            #                         args=(indices,
            #                               chemistry,
            #                               pt_profile,
            #                               line_species,
            #                               cloud_species,
            #                               quenching,
            #                               spec_res,
            #                               read_rad,
            #                               item))
            #
            # processes.append(proc)

        # pool.close()
        #
        # for i, item in enumerate(processes):
        #     boxes.append(item.get(timeout=30))
        #     print(f'\rGetting posterior spectra {i+1}/{random}...', end='', flush=True)

        print(" [DONE]")

        return boxes, read_rad

    @typechecked
    def get_retrieval_teff(
        self,
        tag: str,
        wavel_range: Tuple[float, float],
        random: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function for calculating :math:`T_\\mathrm{eff}`
        and :math:`L_\\mathrm{bol}` from randomly drawn samples of
        the posterior distribution that is estimated with
        :class:`~species.fit.retrieval.AtmosphericRetrieval`.
        This requires the recalculation of the spectra across a
        broad wavelength range (0.5-50 um).

        Parameters
        ----------
        tag : str
            Database tag with the posterior samples.
        wavel_range : tuple(float, float)
            Wavelength range (um) across which the spectra are calculated
            and interpolated. A wide range should be set for an accurate
            calculation of the bolometric luminosity, but the boundaries
            should be within the available range of the opacities.
        random : int
            Number of randomly selected samples (default: 100).

        Returns
        -------
        np.ndarray
            Array with :math:`T_\\mathrm{eff}` samples.
        np.ndarray
            Array with :math:`\\log(L/L_\\mathrm{sun})` samples.
        """

        print(f"Calculating Teff from {random} posterior samples... ")

        boxes, _ = self.get_retrieval_spectra(
            tag=tag,
            random=random,
            wavel_range=wavel_range,
            spec_res=None,
            lbl_opacity_sampling=10000,
        )

        t_eff = np.zeros(len(boxes))
        l_bol = np.zeros(len(boxes))

        for i, box_item in enumerate(boxes):
            if "parallax" in box_item.parameters:
                sample_distance = (
                    1e3 * constants.PARSEC / box_item.parameters["parallax"]
                )
            else:
                sample_distance = box_item.parameters["distance"] * constants.PARSEC

            sample_radius = box_item.parameters["radius"] * constants.R_JUP

            # Scaling for the flux back to the planet surface
            sample_scale = (sample_distance / sample_radius) ** 2

            # Blackbody flux: sigma * Teff^4
            flux_int = simpson(sample_scale * box_item.flux, x=box_item.wavelength)
            t_eff[i] = (flux_int / constants.SIGMA_SB) ** 0.25

            # Bolometric luminosity: 4 * pi * R^2 * sigma * Teff^4
            l_bol[i] = 4.0 * np.pi * sample_radius**2 * flux_int
            l_bol[i] = np.log10(l_bol[i] / constants.L_SUN)

            # np.savetxt(f'output/spectrum/spectrum{i:04d}.dat',
            #            np.column_stack([box_item.wavelength, sample_scale*box_item.flux]),
            #            header='Wavelength (um) - Flux (W m-2 um-1)')

        q_16_teff, q_50_teff, q_84_teff = np.nanpercentile(t_eff, [16.0, 50.0, 84.0])
        print(
            f"Teff (K) = {q_50_teff:.2f} "
            f"(-{q_50_teff-q_16_teff:.2f} "
            f"+{q_84_teff-q_50_teff:.2f})"
        )

        q_16_lbol, q_50_lbol, q_84_lbol = np.nanpercentile(l_bol, [16.0, 50.0, 84.0])
        print(
            f"log(L/Lsun) = {q_50_lbol:.2f} "
            f"(-{q_50_lbol-q_16_lbol:.2f} "
            f"+{q_84_lbol-q_50_lbol:.2f})"
        )

        with h5py.File(self.database, "a") as hdf5_file:
            print(
                f"Storing Teff (K) as attribute of results/fit/{tag}/samples...",
                end="",
            )

            dset = hdf5_file[f"results/fit/{tag}/samples"]

            dset.attrs["teff"] = (
                q_50_teff - q_16_teff,
                q_50_teff,
                q_84_teff - q_50_teff,
            )

            print(" [DONE]")

            print(
                f"Storing log(L/Lsun) as attribute of results/fit/{tag}/samples...",
                end="",
            )

            dset.attrs["log_l_bol"] = (
                q_50_lbol - q_16_lbol,
                q_50_lbol,
                q_84_lbol - q_50_lbol,
            )

            print(" [DONE]")

        return t_eff, l_bol

    @typechecked
    def petitcode_param(
        self, tag: str, sample_type: str = "median", json_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Function for converting the median or maximum likelihood
        posterior parameters of ``petitRADTRANS`` into a dictionary
        of input parameters for ``petitCODE``.

        Parameters
        ----------
        tag : str
            Database tag with the posterior samples.
        sample_type : str
            Sample type that will be selected from the posterior
            ('median' or 'probable'). Either the median or maximum
            likelihood parameters are used.
        json_file : str, None
            JSON file to store the posterior samples. The data will
            not be written if the argument is set to ``None``.

        Returns
        -------
        dict
            Dictionary with parameters for ``petitCODE``.
        """

        from species.read.read_radtrans import ReadRadtrans

        from species.util.retrieval_util import (
            find_cloud_deck,
            log_x_cloud_base,
            pt_ret_model,
            pt_spline_interp,
        )

        if sample_type == "median":
            model_param = self.get_median_sample(tag, verbose=False)

        elif sample_type == "probable":
            model_param = self.get_probable_sample(tag, verbose=False)

        else:
            raise ValueError(
                "The argument of 'sample_type' should be set "
                "to either 'median' or 'probable'."
            )

        sample_box = self.get_samples(tag)

        line_species = []
        for i in range(sample_box.attributes["n_line_species"]):
            line_species.append(sample_box.attributes[f"line_species{i}"])

        cloud_species = []
        cloud_species_full = []

        for i in range(sample_box.attributes["n_cloud_species"]):
            cloud_species.append(sample_box.attributes[f"cloud_species{i}"])
            cloud_species_full.append(sample_box.attributes[f"cloud_species{i}"])

        pcode_param = {}

        pcode_param["logg"] = model_param["logg"]
        pcode_param["metallicity"] = model_param["metallicity"]
        pcode_param["c_o_ratio"] = model_param["c_o_ratio"]

        if "fsed" in model_param:
            pcode_param["fsed"] = model_param["fsed"]

        if "log_kzz" in model_param:
            pcode_param["log_kzz"] = model_param["log_kzz"]

        if "sigma_lnorm" in model_param:
            pcode_param["sigma_lnorm"] = model_param["sigma_lnorm"]

        if "log_p_quench" in model_param:
            pcode_param["log_p_quench"] = model_param["log_p_quench"]
            p_quench = 10.0 ** model_param["log_p_quench"]
        else:
            p_quench = None

        if "temp_nodes" in sample_box.attributes:
            temp_nodes = sample_box.attributes["temp_nodes"]
        else:
            temp_nodes = 15

        if "pressure_grid" in sample_box.attributes:
            pressure_grid = sample_box.attributes["pressure_grid"]
        else:
            pressure_grid = "smaller"

        pressure = np.logspace(-6.0, 3.0, 180)

        if sample_box.attributes["pt_profile"] == "molliere":
            temperature, _, _ = pt_ret_model(
                np.array([model_param["t1"], model_param["t2"], model_param["t3"]]),
                10.0 ** model_param["log_delta"],
                model_param["alpha"],
                model_param["tint"],
                pressure,
                model_param["metallicity"],
                model_param["c_o_ratio"],
            )

        else:
            knot_press = np.logspace(
                np.log10(pressure[0]), np.log10(pressure[-1]), temp_nodes
            )

            knot_temp = []
            for i in range(temp_nodes):
                knot_temp.append(model_param[f"t{i}"])

            knot_temp = np.asarray(knot_temp)

            if "pt_smooth" in model_param:
                pt_smooth = model_param["pt_smooth"]
            else:
                pt_smooth = 0.0

            temperature = pt_spline_interp(
                knot_press, knot_temp, pressure, pt_smooth=pt_smooth
            )

        from petitRADTRANS.chemistry.pre_calculated_chemistry import (
            PreCalculatedEquilibriumChemistryTable,
        )

        eq_chem = PreCalculatedEquilibriumChemistryTable()
        eq_chem.load()

        # Interpolate the abundances, following chemical equilibrium

        abund_in, mmw, _ = eq_chem.interpolate_mass_fractions(
            np.full(pressure.shape[0], model_param["c_o_ratio"]),
            np.full(pressure.shape[0], model_param["metallicity"]),
            temperature,
            pressure,
            carbon_pressure_quench=p_quench,
            full=True,
        )

        cloud_fractions = {}

        if "log_tau_cloud" in model_param:
            # tau_cloud = 10.0 ** model_param["log_tau_cloud"]

            for i, item in enumerate(cloud_species):
                if i == 0:
                    cloud_fractions[item] = 0.0

                else:
                    cloud_fractions[item] = model_param[
                        f"{item}_{cloud_species[0]}_ratio"
                    ]

        else:
            # tau_cloud = None

            for i, item in enumerate(cloud_species):
                cloud_fractions[item] = model_param[f"{item}_fraction"]

        log_x_base = log_x_cloud_base(
            model_param["c_o_ratio"], model_param["metallicity"], cloud_fractions
        )

        p_base = {}

        for item in cloud_species:
            p_base_item = find_cloud_deck(
                item,
                pressure,
                temperature,
                model_param["metallicity"],
                model_param["c_o_ratio"],
                mmw=np.mean(mmw),
                plotting=False,
            )

            abund_in[item] = np.zeros_like(temperature)

            abund_in[item][pressure < p_base_item] = (
                10.0 ** log_x_base[item]
                * (pressure[pressure <= p_base_item] / p_base_item)
                ** model_param["fsed"]
            )

            p_base[item] = p_base_item

            indices = np.where(pressure <= p_base_item)[0]
            pcode_param[f"{item}_base"] = pressure[np.amax(indices)]

        # abundances = create_abund_dict(
        #     abund_in, line_species, sample_box.attributes['chemistry'],
        #     pressure_grid='smaller', indices=None)

        cloud_wavel = (
            sample_box.attributes["wavel_min"],
            sample_box.attributes["wavel_max"],
        )

        read_rad = ReadRadtrans(
            line_species=line_species,
            cloud_species=cloud_species,
            scattering=True,
            wavel_range=(0.5, 50.0),
            pressure_grid=pressure_grid,
            res_mode="c-k",
            cloud_wavel=cloud_wavel,
        )

        print(f"Converting {tag} to petitCODE parameters...", end="", flush=True)

        if sample_box.attributes["quenching"] == "None":
            quenching = None
        else:
            quenching = sample_box.attributes["quenching"]

        model_box = read_rad.get_model(
            model_param=model_param, quenching=quenching, spec_res=500.0
        )

        # Distance (pc)
        if "parallax" in model_param:
            distance = 1e3 * constants.PARSEC / model_param["parallax"]
        else:
            distance = model_param["distance"] * constants.PARSEC

        # Radius (Rjup)
        radius = model_param["radius"] * constants.R_JUP

        # Blackbody flux: sigma * Teff^4
        # Scale the flux back to the planet surface
        flux_int = simpson(
            model_box.flux * (distance / radius) ** 2, x=model_box.wavelength
        )
        pcode_param["teff"] = (flux_int / constants.SIGMA_SB) ** 0.25

        if "log_tau_cloud" in model_param:
            cloud_scaling = read_rad.rt_object.cloud_scaling_factor

            for item in cloud_species_full:
                cloud_abund = abund_in[item]
                indices = np.where(cloud_abund > 0.0)[0]
                pcode_param[f"{item}_abund"] = (
                    cloud_scaling * cloud_abund[np.amax(indices)]
                )

        else:
            for item in cloud_species_full:
                cloud_abund = abund_in[item]
                indices = np.where(cloud_abund > 0.0)[0]
                pcode_param[f"{item}_abund"] = cloud_abund[np.amax(indices)]

        if json_file is not None:
            with open(json_file, "w", encoding="utf-8") as out_file:
                json.dump(pcode_param, out_file, indent=4)

        print(" [DONE]")

        return pcode_param

    @typechecked
    def get_spectral_type(self, tag: str, verbose: bool = True) -> SpectrumBox:
        """
        Function for extracting the spectral template for which the
        goodness-of-fit statistic has been minimized when using
        :func:`~species.fit.compare_spectra.CompareSpectra.spectral_type`
        for comparing data with the empirical templates in a spectral library.

        Parameters
        ----------
        tag : str
            Database tag where the results from the empirical comparison with
            :class:`~species.fit.compare_spectra.CompareSpectra.spectral_type`
            are stored.
        verbose : bool
            Print output, including the parameter values.

        Returns
        -------
        SpectrumBox
            A ``SpectrumBox`` that contains the spectrum and
            attributes of the best-fit template.
        """

        if verbose:
            print_section("Get best spectral type")
            print(f"Database tag: {tag}")

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"results/empirical/{tag}/names"]

            object_name = dset.attrs["object_name"]
            spec_library = dset.attrs["spec_library"]
            n_spec_name = dset.attrs["n_spec_name"]

            if verbose:
                print(f"Object name: {object_name}")
                print(f"Spectral library: {spec_library}")

            if verbose:
                print("\nIncluded spectra:")

            spec_name = []
            for i in range(n_spec_name):
                spec_name.append(dset.attrs[f"spec_name{i}"])
                if verbose:
                    print(f"   - {spec_name[-1]}")

            names = np.array(dset)
            flux_scaling = np.array(hdf5_file[f"results/empirical/{tag}/flux_scaling"])
            av_ext = np.array(hdf5_file[f"results/empirical/{tag}/av_ext"])
            rad_vel = 1e3 * np.array(
                hdf5_file[f"results/empirical/{tag}/rad_vel"]
            )  # (m s-1)
            goodness_fit = np.array(
                [hdf5_file[f"results/empirical/{tag}/goodness_of_fit"]]
            )

        if isinstance(names[i], str):
            best_name = names[i]
        else:
            best_name = names[i].decode("utf-8")

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"spectra/{spec_library}/{best_name}"]
            spectrum = np.array(dset)

            sptype = dset.attrs["sptype"]
            simbad = dset.attrs["simbad"]
            parallax = dset.attrs["parallax"]
            spec_res = dset.attrs["spec_res"]

        if verbose:
            print("\nBest-fit spectral template:")
            print(f"   - G_k = {goodness_fit[0][0]:.2f}")
            print(f"   - Object name = {best_name}")
            print(f"   - SIMBAD name = {simbad}")
            print(f"   - Spectral type = {sptype}")
            print(f"   - Flux scaling = {flux_scaling[0][0]:.2e}")

            if np.amin(av_ext) > 0.0:
                print(f"   - Extinction A_V = {av_ext[0]}")
            if np.amin(rad_vel) > 0.0:
                print(f"   - Radial velocity (km/s) = {rad_vel[0]}")

        return create_box(
            "spectrum",
            spectrum=spec_library,
            wavelength=spectrum[:, 0],
            flux=spectrum[:, 1],
            error=spectrum[:, 2],
            name=best_name,
            sptype=sptype,
            simbad=best_name,
            parallax=parallax,
            spec_res=spec_res,
        )
