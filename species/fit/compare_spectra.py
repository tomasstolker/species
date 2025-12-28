"""
Module with functionalities for comparing a spectrum with a library of
empirical or model spectra. Empirical libraries of field or low-gravity
objects can be used to determine the spectral type. A comparison with
model spectra are useful for exploration of the atmospheric parameter.
"""

import configparser
import os
import warnings

from typing import List, Optional, Tuple, Union

import h5py
import numpy as np

from scipy.interpolate import interp1d
from typeguard import typechecked

from species.core import constants
from species.data.spec_data.add_spec_data import add_spec_library
from species.phot.syn_phot import SyntheticPhotometry
from species.read.read_filter import ReadFilter
from species.read.read_model import ReadModel
from species.read.read_object import ReadObject
from species.util.dust_util import ism_extinction
from species.util.spec_util import smooth_spectrum


class CompareSpectra:
    """
    Class for comparing one or multiple spectra of an object with a
    library of empirical spectra or a grid of model spectra.
    """

    @typechecked
    def __init__(
        self,
        object_name: str,
        spec_name: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : str, list(str), None
            Name of the spectrum or a list with the names of the
            spectra that will be used for the comparison. The
            spectrum names should have been stored at the object
            data of ``object_name``. No spectra are selected if
            the argument is set to ``None``, which is only
            possible when selecting photometric fluxes with the
            ``inc_phot`` parameter in ``compare_model``.

        Returns
        -------
        NoneType
            None
        """

        self.object_name = object_name
        self.spec_name = spec_name

        if self.spec_name is None:
            self.spec_name = []

        elif isinstance(self.spec_name, str):
            self.spec_name = [self.spec_name]

        self.object = ReadObject(object_name)

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.data_folder = config["species"]["data_folder"]

    @typechecked
    def spectral_type(
        self,
        tag: str,
        spec_library: str,
        wavel_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        sptypes: Optional[List[str]] = None,
        av_ext: Optional[Union[List[float], np.ndarray]] = None,
        rad_vel: Optional[Union[List[float], np.ndarray]] = None,
    ) -> List[Tuple[float, str, str]]:
        """
        Method for finding the best matching empirical spectra
        from the selected library by evaluating the goodness-of-fit
        statistic from `Cushing et al. (2008) <https://ui.adsabs.
        harvard.edu/abs/2008ApJ...678.1372C/abstract>`_.

        Parameters
        ----------
        tag : str
            Database tag where for each spectrum from the spectral
            library the best-fit parameters will be stored. So when
            testing a range of values for ``av_ext`` and ``rad_vel``,
            only the parameters that minimize the goodness-of-fit
            statistic will be stored.
        spec_library : str
            Name of the spectral library (e.g. 'irtf', 'spex',
            'kesseli+2017', 'bonnefoy+2014').
        wavel_range : tuple(float, float), None
            Wavelength range (:math:`\\mu\\mathrm{m}`) that
            is evaluated.
        sptypes : list(str), None
            List with spectral types to compare with, for example
            ``sptypes=['M', 'L']``. All available spectral types
            in the ``spec_library`` are compared with if the
            argument is set to ``None``.
        av_ext : list(float), np.ndarray, None
            List of :math:`A_V` for which the goodness-of-fit
            statistic is tested. The extinction is calculated with
            the empirical relation from `Cardelli et al. (1989)
            <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/
            abstract>`_. The extinction is not varied if the argument
            is set to ``None``.
        rad_vel : list(float), np.ndarray, None
            List of radial velocities (km s-1) for which the
            goodness-of-fit statistic is tested. The radial
            velocity is not varied if the argument is set
            to ``None``.

        Returns
        -------
        list(tuple(float, str, str))
            List with the 10 best matching spectra. Each item in the
            list includes the goodness-of-fit, the object name, and
            the spectral type. Less than 10 objects are stored if
            there were less than 10 spectra selected from the
            ``spec_library``.
        """

        w_i = 1.0

        if av_ext is None:
            av_ext = [0.0]

        if rad_vel is None:
            rad_vel = [0.0]

        with h5py.File(self.database, "a") as hdf5_file:
            if f"spectra/{spec_library}" not in hdf5_file:
                add_spec_library(self.data_folder, hdf5_file, spec_library)

            # Read object spectra and resolution

            obj_spec = []
            obj_res = []

            for item in self.spec_name:
                obj_spec.append(self.object.get_spectrum()[item][0])
                obj_res.append(self.object.get_spectrum()[item][3])

            # Read inverted covariance matrix
            # obj_inv_cov = self.object.get_spectrum()[self.spec_name][2]

            # Create empty lists for results

            name_list = []
            spt_list = []
            gk_list = []
            ck_list = []
            av_list = []
            rv_list = []

            print_message = ""

            # Start looping over library spectra

            for i, item in enumerate(hdf5_file[f"spectra/{spec_library}"]):
                # Read spectrum spectral type from library
                dset = hdf5_file[f"spectra/{spec_library}/{item}"]

                if isinstance(dset.attrs["sptype"], str):
                    item_sptype = dset.attrs["sptype"]
                else:
                    # Use decode for backward compatibility
                    item_sptype = dset.attrs["sptype"].decode("utf-8")

                if item_sptype == "None":
                    continue

                if sptypes is None or item_sptype[0] in sptypes:
                    # Convert HDF5 dataset into numpy array
                    spectrum = np.asarray(dset)

                    if wavel_range is not None:
                        # Select subset of the spectrum

                        if wavel_range[0] is None:
                            indices = np.where((spectrum[:, 0] < wavel_range[1]))[0]

                        elif wavel_range[1] is None:
                            indices = np.where((spectrum[:, 0] > wavel_range[0]))[0]

                        else:
                            indices = np.where(
                                (spectrum[:, 0] > wavel_range[0])
                                & (spectrum[:, 0] < wavel_range[1])
                            )[0]

                        if len(indices) == 0:
                            raise ValueError(
                                "The selected wavelength range does not "
                                "cover any wavelength points of the input "
                                "spectrum. Please use a broader range as "
                                "argument of 'wavel_range'."
                            )

                        spectrum = spectrum[indices,]

                    empty_message = len(print_message) * " "
                    print(f"\r{empty_message}", end="")

                    print_message = f"Processing spectra... {item}"
                    print(f"\r{print_message}", end="")

                    # Loop over all values of A_V and RV that will be tested

                    for av_item in av_ext:
                        for rv_item in rad_vel:
                            for j, spec_item in enumerate(obj_spec):
                                # Dust extinction
                                ism_ext = ism_extinction(av_item, 3.1, spectrum[:, 0])
                                flux_scaling = 10.0 ** (-0.4 * ism_ext)

                                # Shift wavelengths by RV
                                wavel_shifted = (
                                    spectrum[:, 0]
                                    + spectrum[:, 0] * 1e3 * rv_item / constants.LIGHT
                                )

                                # Smooth spectrum
                                flux_smooth = smooth_spectrum(
                                    wavel_shifted,
                                    spectrum[:, 1] * flux_scaling,
                                    spec_res=obj_res[j],
                                    force_smooth=True,
                                )

                                # Interpolate library spectrum to object wavelengths
                                interp_spec = interp1d(
                                    spectrum[:, 0],
                                    flux_smooth,
                                    kind="linear",
                                    fill_value="extrapolate",
                                )

                                indices = np.where(
                                    (spec_item[:, 0] > np.amin(spectrum[:, 0]))
                                    & (spec_item[:, 0] < np.amax(spectrum[:, 0]))
                                )[0]

                                flux_resample = interp_spec(spec_item[indices, 0])

                                c_numer = (
                                    w_i
                                    * spec_item[indices, 1]
                                    * flux_resample
                                    / spec_item[indices, 2] ** 2
                                )

                                c_denom = (
                                    w_i * flux_resample**2 / spec_item[indices, 2] ** 2
                                )

                                if j == 0:
                                    g_k = 0.0
                                    c_k_spec = []

                                idx_select = np.isfinite(c_numer) & np.isfinite(c_denom)

                                c_k = np.sum(c_numer[idx_select]) / np.sum(
                                    c_denom[idx_select]
                                )
                                c_k_spec.append(c_k)

                                chi_sq = (
                                    spec_item[indices, 1][idx_select]
                                    - c_k * flux_resample[idx_select]
                                ) / spec_item[indices, 2][idx_select]

                                g_k += np.sum(w_i * chi_sq**2)

                                # obj_inv_cov_crop = obj_inv_cov[indices, :]
                                # obj_inv_cov_crop = obj_inv_cov_crop[:, indices]

                                # g_k = np.dot(spec_item[indices, 1]-c_k*flux_resample,
                                #     np.dot(obj_inv_cov_crop,
                                #            spec_item[indices, 1]-c_k*flux_resample))

                            if np.isnan(c_k_spec):
                                # This can happen if the spectrum only contains NaNs
                                # because there is an issue with the flux calibration
                                # For example: ULAS J141623.94+134836.3 (SpeX)
                                g_k = np.inf

                            # Append to the lists of results

                            name_list.append(item)
                            spt_list.append(item_sptype)
                            gk_list.append(g_k)
                            ck_list.append(c_k_spec)
                            av_list.append(av_item)
                            rv_list.append(rv_item)

        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print("\rProcessing spectra... [DONE]")

        name_list = np.asarray(name_list)
        spt_list = np.asarray(spt_list)
        gk_list = np.asarray(gk_list)
        ck_list = np.asarray(ck_list)
        av_list = np.asarray(av_list)
        rv_list = np.asarray(rv_list)

        sort_index = np.argsort(gk_list)

        name_list = name_list[sort_index]
        spt_list = spt_list[sort_index]
        gk_list = gk_list[sort_index]
        ck_list = ck_list[sort_index]
        av_list = av_list[sort_index]
        rv_list = rv_list[sort_index]

        name_select = []
        spt_select = []
        gk_select = []
        ck_select = []
        av_select = []
        rv_select = []

        for i, item in enumerate(name_list):
            if item not in name_select:
                name_select.append(item)
                spt_select.append(spt_list[i])
                gk_select.append(gk_list[i])
                ck_select.append(ck_list[i])
                av_select.append(av_list[i])
                rv_select.append(rv_list[i])

        print("Best-fitting spectra:")

        best_list = []

        if len(gk_select) < 10:
            for gk_idx, gk_item in enumerate(gk_select):
                best_list.append((gk_item, name_select[gk_idx], spt_select[gk_idx]))

                print(
                    f"   {gk_idx+1:2d}. G = {gk_item:.2e} -> "
                    "{name_select[gk_idx]}, {spt_select[gk_idx]}, "
                    f"A_V = {av_select[gk_idx]:.2f}, "
                    f"RV = {rv_select[gk_idx]:.0f} km/s,\n"
                    f"                       scalings = {ck_select[gk_idx]}"
                )

        else:
            for gk_idx in range(10):
                best_list.append(
                    (gk_select[gk_idx], name_select[gk_idx], spt_select[gk_idx])
                )

                print(
                    f"   {gk_idx+1:2d}. G = {gk_select[gk_idx]:.2e} -> "
                    f"{name_select[gk_idx]}, {spt_select[gk_idx]}, "
                    f"A_V = {av_select[gk_idx]:.2f}, "
                    f"RV = {rv_select[gk_idx]:.0f} km/s,\n"
                    f"                       scalings = {ck_select[gk_idx]}"
                )

        from species.data.database import Database

        species_db = Database()

        species_db.add_empirical(
            tag=tag,
            names=name_select,
            sptypes=spt_select,
            goodness_of_fit=gk_select,
            flux_scaling=ck_select,
            av_ext=av_select,
            rad_vel=rv_select,
            object_name=self.object_name,
            spec_name=self.spec_name,
            spec_library=spec_library,
        )

        return best_list

    @typechecked
    def compare_model(
        self,
        tag: str,
        model: str,
        av_points: Optional[Union[List[float], np.ndarray]] = None,
        fix_logg: Optional[float] = None,
        scale_spec: Optional[List[str]] = None,
        weights: bool = True,
        inc_phot: Union[List[str], bool] = False,
    ) -> None:
        """
        Method for evaluating the goodness-of-fit
        statistic from `Cushing et al. (2008) <https://ui.adsabs.
        harvard.edu/abs/2008ApJ...678.1372C/abstract>`_ for all
        combination of parameters in a grid of model spectra.
        For each set of parameters, the model will be scaled in
        order to minimize the goodness-of-fit. The scaling gives
        the radius since the distance is adopted from the selected
        ``object_name``. The model spectra are not interpolated so
        only parameters available from the grid are tested.

        Parameters
        ----------
        tag : str
            Database tag where for each spectrum from the spectral
            library the best-fit parameters will be stored.
        model : str
            Name of the model grid with synthetic spectra.
        av_points : list(float), np.ndarray, None
            List of :math:`A_V` extinction values for which the
            goodness-of-fit statistic will be tested. The extinction is
            calculated with the relation from `Cardelli et al. (1989)
            <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/
            abstract>`_.
        fix_logg : float, None
            Fix the value of :math:`\\log(g)`, for example if estimated
            from gravity-sensitive spectral features. Typically,
            :math:`\\log(g)` cannot be accurately determined when
            comparing the spectra over a broad wavelength range.
        scale_spec : list(str), None
            List with names of observed spectra to which an additional
            flux scaling is applied to best match each model spectrum.
            This parameter can be used to account for a difference in
            absolute calibration between spectra.
        weights : bool
            Apply a weighting of the spectra and photometry based on
            the widths of the wavelengths bins and the FWHM of the
            filter profiles, respectively. No weighting is applied
            if the argument is set to ``False``.
        inc_phot : list(str), bool
            Filter names for which photometric fluxes of the
            selected ``object_name`` will be included in the
            comparison. The argument can be a list with filter names
            or a boolean to select all or none of the photometry.
            By default the argument is set to ``False`` so
            photometric fluxes are not included.

        Returns
        -------
        NoneType
            None
        """

        w_i = {}

        for spec_item in self.spec_name:
            # Determine width of wavelength bins for the spectra
            # Set the optional weights for the statistic to the
            # width of the bins or otherwise to one
            obj_wavel = self.object.get_spectrum()[spec_item][0][:, 0]

            diff = (np.diff(obj_wavel)[1:] + np.diff(obj_wavel)[:-1]) / 2.0
            diff = np.insert(diff, 0, diff[0])
            diff = np.append(diff, diff[-1])

            if weights:
                w_i[spec_item] = diff
            else:
                w_i[spec_item] = np.ones(obj_wavel.shape[0])

        read_object = ReadObject(self.object_name)

        if isinstance(inc_phot, bool):
            # The argument of inc_phot is a boolean
            if inc_phot:
                # Select all filters if inc_phot=True
                inc_phot = read_object.list_filters()

            else:
                inc_phot = []

        object_flux = {}
        for filter_item in inc_phot:
            object_flux[filter_item] = read_object.get_photometry(filter_item)[2:]

        if scale_spec is None:
            scale_spec = []

        phot_wavel = {}
        for phot_item in inc_phot:
            read_filt = ReadFilter(phot_item)
            phot_wavel[phot_item] = read_filt.mean_wavelength()

            if weights:
                # Set the weight for the photometric fluxes
                # to the FWHM of the filter profile
                w_i[phot_item] = read_filt.filter_fwhm()
            else:
                w_i[phot_item] = 1.0

        model_reader = ReadModel(model)
        model_param = model_reader.get_parameters()
        grid_points = model_reader.get_points()

        coord_points = []
        for key, value in grid_points.items():
            if key == "logg" and fix_logg is not None:
                if fix_logg in value:
                    coord_points.append(np.array([fix_logg]))

                else:
                    raise ValueError(
                        f"The argument of 'fix_logg' ({fix_logg}) is "
                        f"not found in the parameter grid of the "
                        f"model spectra. The following values of "
                        f"log(g) are available: {value}"
                    )

            else:
                coord_points.append(value)

        if av_points is not None:
            model_param.append("ism_ext")
            coord_points.append(av_points)

        grid_shape = []
        for item in coord_points:
            grid_shape.append(len(item))

        for _ in range(len(coord_points), 6):
            model_param.append(None)
            coord_points.append([None])
            grid_shape.append(1)

        spec_data = self.object.get_spectrum()

        fit_stat = np.zeros(grid_shape)
        flux_scaling = np.zeros(grid_shape)

        if len(scale_spec) > 0:
            grid_shape.append(len(scale_spec))
            extra_scaling = np.zeros(grid_shape)

        else:
            extra_scaling = None

        n_iter = 1
        for item in coord_points:
            if len(item) > 0:
                n_iter *= len(item)

        count = 1

        for coord_0_idx, coord_0_item in enumerate(coord_points[0]):
            for coord_1_idx, coord_1_item in enumerate(coord_points[1]):
                for coord_2_idx, coord_2_item in enumerate(coord_points[2]):
                    for coord_3_idx, coord_3_item in enumerate(coord_points[3]):
                        for coord_4_idx, coord_4_item in enumerate(coord_points[4]):
                            for coord_5_idx, coord_5_item in enumerate(coord_points[5]):
                                print(
                                    f"\rProcessing model spectrum {count}/{n_iter}...",
                                    end="",
                                )

                                param_dict = {}
                                model_spec = {}

                                if model_param[0] is not None:
                                    param_dict[model_param[0]] = coord_0_item

                                if model_param[1] is not None:
                                    param_dict[model_param[1]] = coord_1_item

                                if model_param[2] is not None:
                                    param_dict[model_param[2]] = coord_2_item

                                if model_param[3] is not None:
                                    param_dict[model_param[3]] = coord_3_item

                                if model_param[4] is not None:
                                    param_dict[model_param[4]] = coord_4_item

                                if model_param[5] is not None:
                                    param_dict[model_param[5]] = coord_5_item

                                model_box_full = model_reader.get_data(param_dict)

                                for spec_item in self.spec_name:
                                    obj_spec = self.object.get_spectrum()[spec_item][0]
                                    obj_res = self.object.get_spectrum()[spec_item][3]

                                    # Smooth model spectrum

                                    model_flux = smooth_spectrum(
                                        model_box_full.wavelength,
                                        model_box_full.flux,
                                        obj_res,
                                    )

                                    # Resample model spectrum

                                    flux_intep = interp1d(
                                        model_box_full.wavelength,
                                        model_flux,
                                        bounds_error=False,
                                    )
                                    model_flux = flux_intep(obj_spec[:, 0])

                                    nan_wavel = np.sum(np.isnan(model_flux))

                                    if nan_wavel > 0:
                                        warnings.warn(
                                            "The extracted model spectrum contains "
                                            f"{nan_wavel} fluxes with NaN. Probably "
                                            "because some of the wavelengths of the "
                                            "data are outside the available "
                                            "wavelength range of the model grid. "
                                            "These wavelengths are ignored when "
                                            "calculating the goodness-of-fit statistic."
                                        )

                                    model_spec[spec_item] = model_flux

                                g_fit = 0.0

                                model_list = []
                                data_list = []
                                weights_list = []

                                for spec_item in self.spec_name:
                                    if spec_item not in scale_spec:
                                        model_list.append(model_spec[spec_item])
                                        data_list.append(spec_data[spec_item][0])
                                        weights_list.append(w_i[spec_item])

                                model_phot = {}

                                for phot_item in inc_phot:
                                    syn_phot = SyntheticPhotometry(phot_item)
                                    model_phot[phot_item] = syn_phot.spectrum_to_flux(
                                        model_box_full.wavelength, model_box_full.flux
                                    )[0]

                                    phot_flux = object_flux[phot_item]

                                    if phot_flux.ndim == 1:
                                        phot_data = np.array(
                                            [
                                                [
                                                    phot_wavel[phot_item],
                                                    phot_flux[0],
                                                    phot_flux[1],
                                                ]
                                            ]
                                        )
                                        data_list.append(phot_data)
                                        model_list.append(
                                            np.array([model_phot[phot_item]])
                                        )
                                        weights_list.append(np.array([w_i[phot_item]]))

                                    else:
                                        for phot_idx in range(phot_flux.shape[1]):
                                            phot_data = np.array(
                                                [
                                                    [
                                                        phot_wavel[phot_item],
                                                        phot_flux[0][phot_idx],
                                                        phot_flux[1][phot_idx],
                                                    ]
                                                ]
                                            )
                                            data_list.append(phot_data)
                                            model_list.append(
                                                np.array([model_phot[phot_item]])
                                            )
                                            weights_list.append(
                                                np.array([w_i[phot_item]])
                                            )

                                data_list = np.concatenate(data_list)
                                model_list = np.concatenate(model_list)
                                weights_list = np.concatenate(weights_list)

                                c_numer = (
                                    weights_list
                                    * data_list[:, 1]
                                    * model_list
                                    / data_list[:, 2] ** 2
                                )

                                c_denom = (
                                    weights_list * model_list**2 / data_list[:, 2] ** 2
                                )

                                if np.nansum(model_list) == 0.0:
                                    # This happens if model spectra contain
                                    # only zeros because the grid point was
                                    # missing and could not be fixed with
                                    # an interpolation when the grid was
                                    # added to the database
                                    scaling = np.nan

                                else:
                                    scaling = np.nansum(c_numer) / np.nansum(c_denom)

                                flux_scaling[
                                    coord_0_idx,
                                    coord_1_idx,
                                    coord_2_idx,
                                    coord_3_idx,
                                    coord_4_idx,
                                    coord_5_idx,
                                ] = scaling

                                for spec_item in scale_spec:
                                    spec_idx = scale_spec.index(spec_item)

                                    c_numer = (
                                        w_i[spec_item]
                                        * scaling
                                        * model_spec[spec_item]
                                        * spec_data[spec_item][0][:, 1]
                                        / spec_data[spec_item][0][:, 2] ** 2
                                    )

                                    c_denom = (
                                        w_i[spec_item]
                                        * spec_data[spec_item][0][:, 1] ** 2
                                        / spec_data[spec_item][0][:, 2] ** 2
                                    )

                                    spec_scaling = np.nansum(c_numer) / np.nansum(
                                        c_denom
                                    )

                                    extra_scaling[
                                        coord_0_idx,
                                        coord_1_idx,
                                        coord_2_idx,
                                        coord_3_idx,
                                        coord_4_idx,
                                        coord_5_idx,
                                        spec_idx,
                                    ] = spec_scaling

                                for phot_item in inc_phot:
                                    phot_flux = object_flux[phot_item]

                                    g_fit += np.nansum(
                                        w_i[phot_item]
                                        * (
                                            phot_flux[0]
                                            - scaling * model_phot[phot_item]
                                        )
                                        ** 2
                                        / phot_flux[1] ** 2
                                    )

                                for spec_item in self.spec_name:
                                    if spec_item in scale_spec:
                                        spec_idx = scale_spec.index(spec_item)

                                        data_scaling = extra_scaling[
                                            coord_0_idx,
                                            coord_1_idx,
                                            coord_2_idx,
                                            coord_3_idx,
                                            coord_4_idx,
                                            coord_5_idx,
                                            spec_idx,
                                        ]

                                    else:
                                        data_scaling = 1.0

                                    g_fit += np.nansum(
                                        w_i[spec_item]
                                        * (
                                            data_scaling * spec_data[spec_item][0][:, 1]
                                            - scaling * model_spec[spec_item]
                                        )
                                        ** 2
                                        / spec_data[spec_item][0][:, 2] ** 2
                                    )

                                if np.nansum(model_list) == 0.0:
                                    # This happens if model spectra contain
                                    # only zeros because the grid point was
                                    # missing and could not be fixed with
                                    # an interpolation when the grid was
                                    # added to the database
                                    g_fit = np.nan

                                fit_stat[
                                    coord_0_idx,
                                    coord_1_idx,
                                    coord_2_idx,
                                    coord_3_idx,
                                    coord_4_idx,
                                    coord_5_idx,
                                ] = g_fit

                                count += 1

        print(" [DONE]")

        for param_idx, param_item in enumerate(model_param):
            if param_item is None:
                model_param = model_param[:param_idx]
                coord_points = coord_points[:param_idx]
                break

        for dim_idx in range(fit_stat.ndim, 0, -1):
            if dim_idx > len(model_param):
                fit_stat = fit_stat[..., 0]
                flux_scaling = flux_scaling[..., 0]

                if extra_scaling is not None:
                    extra_scaling = extra_scaling[..., 0, :]

        from species.data.database import Database

        species_db = Database()

        species_db.add_comparison(
            tag=tag,
            goodness_of_fit=fit_stat,
            flux_scaling=flux_scaling,
            model_param=model_param,
            coord_points=coord_points,
            object_name=self.object_name,
            spec_name=self.spec_name,
            model_name=model,
            scale_spec=scale_spec,
            extra_scaling=extra_scaling,
            inc_phot=inc_phot,
        )
