"""
Module with functionalities for the analysis of emission lines.
"""

import configparser
import os
import warnings

from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    import ultranest
except:
    warnings.warn(
        "UltraNest could not be imported. Perhaps "
        "because cython was not correctly compiled?"
    )

from astropy import units as u
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import StdDevUncertainty
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from typeguard import typechecked

from species.core import constants
from species.data import database
from species.read import read_object
from species.util import read_util


class EmissionLine:
    """
    Class for the analysis of emission lines.
    """

    @typechecked
    def __init__(
        self,
        object_name: str,
        spec_name: str,
        hydrogen_line: Optional[str] = None,
        lambda_rest: Optional[float] = None,
        wavel_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : str
            Name of the spectrum that is stored at the object data
            of ``object_name``.
        hydrogen_line : str, None
            Name of the hydrogen line that will be analyzed. The
            names available lines can be checked with the
            :func:`~species.analysis.emission_line.EmissionLine.
            list_hydrogen_lines` method. If the argument is set
            to ``None`` then provide the rest wavelength as
            argument of ``lambda_rest``.
        lambda_rest : float, None
            Rest wavelength (um) of the emission line. The parameter
            if used for calculating the radial velocity and its
            uncertainty. The argument can be set to ``None`` and will
            be ignored if the argument of ``hydrogen_line`` is used.
        wavel_range : tuple(float, float), None
            Wavelength range (um) that is cropped from the
            spectrum. The full spectrum is used if the argument
            is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        self.object_name = object_name
        self.spec_name = spec_name
        self.hydrogen_line = hydrogen_line

        self.object = read_object.ReadObject(object_name)
        self.parallax = self.object.get_parallax()[0]
        self.spectrum = self.object.get_spectrum()[spec_name][0]

        if wavel_range is None:
            self.wavel_range = (self.spectrum[0, 0], self.spectrum[-1, 0])

        else:
            self.wavel_range = wavel_range

            indices = np.where(
                (self.spectrum[:, 0] >= wavel_range[0])
                & (self.spectrum[:, 0] <= wavel_range[1])
            )[0]

            self.spectrum = self.spectrum[
                indices,
            ]

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

        h5_database = h5py.File(self.database, "r")

        if "accretion" not in h5_database:
            h5_database.close()
            species_db = database.Database()
            species_db.add_accretion()

        if self.hydrogen_line is None and lambda_rest is not None:
            self.lambda_rest = lambda_rest

        else:
            if self.hydrogen_line is None:
                self.list_hydrogen_lines()

                warnings.warn(
                    "Provide an argument for either "
                    "the 'hydrogen_line' or 'lambda_rest' "
                    "parameter. Only one of the two "
                    "arguments should be set to None."
                )

                self.hydrogen_line = input("Please provide the name "
                                           "of the hydrogen line: ")

            h5_database = h5py.File(self.database, "r")

            line_names = np.array(h5_database["accretion/hydrogen_lines"], dtype=str)
            line_wavel = np.array(h5_database["accretion/wavelengths"])

            if self.hydrogen_line not in line_names:
                raise ValueError(
                    "The hydrogen line with the name "
                    f"'{self.hydrogen_line}' is not found. Please "
                    "use the 'list_hydrogen_lines' method to check "
                    "which hydrogen lines are available."
                )

            line_idx = np.argwhere(line_names == self.hydrogen_line)[0][0]

            # Vacuum rest wavelength (um)
            self.lambda_rest = line_wavel[line_idx]
            print(f"Hydrogen line = {self.hydrogen_line}")

        print(f"Rest wavelength (um) = {self.lambda_rest:.4f}")

        self.spec_vrad = (
            1e-3
            * constants.LIGHT
            * (self.spectrum[:, 0] - self.lambda_rest)
            / self.lambda_rest
        )

        self.continuum_flux = np.full(self.spectrum.shape[0], 0.0)
        self.continuum_check = False

    @typechecked
    def list_hydrogen_lines(self) -> List[str]:
        """
        Function to list the hydrogen lines for which an accretion
        luminosity relation :math:`L_\\mathrm{acc}(L_\\mathrm{line}`)
        was fit explicitly provided in `Aoyama et al. (2021)
        <https://ui.adsabs.harvard.edu/abs/2021ApJ...917L..30A/
        abstract>`_ [Ao21] or `Marleau & Aoyama (2022)
        <https://ui.adsabs.harvard.edu/abs/2022RNAAS...6..262M/
        abstract>`_ [MA22]. These names can be set as argument
        ``hydrogen_line``. In that case, the measured line luminosity
        from :func:`~species.analysis.emission_line.EmissionLine.
        fit_gaussian` will be automatically converted to an accretion
        luminosity with the relation from [Ao21] and [MA22].

        Returns
        -------
        list(str)
            List with the names of the hydrogen lines for which there
            are coefficients available for the accretion relation.
        """

        with h5py.File(self.database, "r") as h5_file:
            line_names = list(h5_file["accretion/hydrogen_lines/"])

            # Convert from bytes to strings
            for i, item in enumerate(line_names):
                if isinstance(item, bytes):
                    line_names[i] = item.decode("utf-8")

        print(f"Available hydrogen lines:\n{line_names}")

        return line_names

    @typechecked
    def subtract_continuum(
        self,
        poly_degree: int = 3,
        plot_filename: Optional[str] = "continuum.pdf",
        spec_filename: Optional[str] = None,
    ) -> None:
        """
        Method for fitting the continuum with a polynomial function of
        the following form: :math:`P = \\sum_{i=0}^{i=n}C_{i} * x^{i}`.
        The spectrum is first smoothed with a median filter and then
        fitted with a linear least squares algorithm.

        Parameters
        ----------
        poly_degree : int
            Degree of the polynomial series.
        plot_filename : str, None
            Filename for the plots with the continuum fit and the
            continuum-subtracted spectrum. The plot is shown in an
            interface window if the argument is set to ``None``.
        spec_filename : str, None
            Output text file for writing the continuum-subtracted
            spectrum. The file will not be created if the argument
            is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        # Fit continuum

        print("Fitting continuum...", end="", flush=True)

        spec_extract = Spectrum1D(
            flux=self.spectrum[:, 1] * u.W,
            spectral_axis=self.spectrum[:, 0] * u.um,
            uncertainty=StdDevUncertainty(self.spectrum[:, 2] * u.W),
        )

        g1_fit = fit_generic_continuum(
            spec_extract,
            median_window=3,
            model=Polynomial1D(poly_degree),
            fitter=LinearLSQFitter(),
        )

        continuum_fit = g1_fit(spec_extract.spectral_axis)

        print(" [DONE]")

        # Subtract continuum

        spec_cont_sub = spec_extract - continuum_fit

        self.continuum_flux = continuum_fit / u.W

        # Create plot

        if plot_filename is None:
            print("Plotting continuum fit...", end="", flush=True)
        else:
            print(f"Plotting continuum fit: {plot_filename}...", end="", flush=True)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams["axes.axisbelow"] = False

        plt.figure(figsize=(6, 6))
        gs = mpl.gridspec.GridSpec(2, 1)
        gs.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = ax1.twiny()
        ax4 = ax2.twiny()

        ax1.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
            labelbottom=False,
        )

        ax1.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
            labelbottom=False,
        )

        ax2.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
        )

        ax2.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
        )

        ax3.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
        )

        ax3.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
        )

        ax4.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
            labeltop=False,
        )

        ax4.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
            labeltop=False,
        )

        ax1.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax2.set_xlabel("Wavelength (µm)", fontsize=16)
        ax2.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax3.set_xlabel("Velocity (km s$^{-1}$)", fontsize=16)

        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        ax2.get_xaxis().set_label_coords(0.5, -0.1)
        ax2.get_yaxis().set_label_coords(-0.1, 0.5)
        ax3.get_xaxis().set_label_coords(0.5, 1.12)

        ax1.plot(
            spec_extract.spectral_axis,
            spec_extract.flux,
            color="black",
            label=self.spec_name,
        )
        ax1.plot(
            spec_extract.spectral_axis,
            continuum_fit,
            color="tab:blue",
            label="Continuum fit",
        )

        ax2.plot(
            spec_cont_sub.spectral_axis,
            spec_cont_sub.flux,
            color="black",
            label="Continuum subtracted",
        )

        ax3.plot(self.spec_vrad, spec_extract.flux, ls="-", lw=0.0)
        ax4.plot(self.spec_vrad, spec_cont_sub.flux, ls="-", lw=0.0)

        ax1.legend(loc="upper right", frameon=False, fontsize=12.0)
        ax2.legend(loc="upper right", frameon=False, fontsize=12.0)

        print(" [DONE]")

        if plot_filename is None:
            plt.show()
        else:
            plt.savefig(plot_filename, bbox_inches="tight")

        plt.clf()
        plt.close()

        # Overwrite original spectrum with continuum-subtracted spectrum
        self.spectrum[:, 1] = spec_cont_sub.flux

        self.continuum_check = True

        if spec_filename is not None:
            print(
                f"Writing continuum-subtracted spectrum: {spec_filename}...",
                end="",
                flush=True,
            )
            header = "Wavelength (um) - Flux (W m-2 um-1) - Error (W m-2 um-1)"
            np.savetxt(spec_filename, self.spectrum, header=header)
            print(" [DONE]")

    @typechecked
    def integrate_flux(
        self,
        wavel_int: Tuple[float, float],
        interp_kind: str = "linear",
        plot_filename: Optional[str] = "int_line.pdf",
    ) -> Union[np.float64, np.float64]:
        """
        Method for calculating the integrated line flux and error. The
        spectrum is first interpolated to :math:`R = 100000` and then
        integrated across the specified wavelength range with the
        composite trapezoidal rule of ``np.trapz``. The error is
        estimated with a Monte Carlo approach from 1000 samples.
        The accretion luminosity is also calculated with the relation
        from `Aoyama et al. (2021) <https://ui.adsabs.harvard.edu/
        abs/2021ApJ...917L..30A/abstract>`_ and `Marleau & Aoyama
        (2022) <https://ui.adsabs.harvard.edu/abs/
        2022RNAAS...6..262M/abstract>`_ if the argument of
        ``hydrogen_line`` was set when creating an instance of the
        class.

        Parameters
        ----------
        wavel_int : tuple(float, float)
            Wavelength range (um) across which the flux
            will be integrated.
        interp_kind : str
            Kind of interpolation kind for
            ``scipy.interpolate.interp1d`` (default: 'linear').
        plot_filename : str, None
            Filename for the plot with the interpolated line profile.
            The plot is shown in an interface window if the argument
            is set to ``None``.

        Returns
        -------
        float
            Integrated line flux (W m-2).
        float
            Flux error (W m-2).
        """

        if plot_filename is None:
            print("Plotting integrated line...", end="", flush=True)
        else:
            print(f"Plotting integrated line: {plot_filename}...", end="", flush=True)

        n_samples = 1000

        wavel_high_res = read_util.create_wavelengths(wavel_int, 1e5)

        # Creating plot

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams["axes.axisbelow"] = False

        plt.figure(figsize=(6, 3))
        gs = mpl.gridspec.GridSpec(1, 1)
        gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = ax1.twiny()

        ax1.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
        )

        ax1.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
        )

        ax2.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=False,
            left=False,
            right=True,
        )

        ax2.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=False,
            left=False,
            right=True,
        )

        ax1.set_xlabel("Wavelength (µm)", fontsize=16)
        ax1.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax2.set_xlabel("Velocity (km s$^{-1}$)", fontsize=16)

        ax1.get_xaxis().set_label_coords(0.5, -0.12)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        ax2.get_xaxis().set_label_coords(0.5, 1.12)

        ax1.plot(
            self.spectrum[:, 0],
            self.spectrum[:, 1],
            color="black",
            label=self.spec_name,
        )
        ax2.plot(self.spec_vrad, self.spectrum[:, 1], ls="-", lw=0.0)

        flux_sample = np.zeros(n_samples)
        fwhm_sample = np.zeros(n_samples)
        mean_sample = np.zeros(n_samples)
        vrad_sample = np.zeros(n_samples)
        lum_sample = np.zeros(n_samples)

        for i in range(n_samples):
            # Sample fluxes from random errors
            spec_rand = np.random.normal(self.spectrum[:, 1], self.spectrum[:, 2])

            # Interpolate sampled spectrum
            spec_interp = interp1d(
                self.spectrum[:, 0], spec_rand, kind=interp_kind, bounds_error=False
            )

            # Resample to high-resolution wavelengths
            flux_rand = spec_interp(wavel_high_res)

            # Integrate line flux (W m-2)
            flux_sample[i] = np.trapz(flux_rand, x=wavel_high_res)

            # Line luminosity (Lsun)
            lum_sample[i] = (
                4.0
                * np.pi
                * (1e3 * constants.PARSEC / self.parallax) ** 2
                * flux_sample[i]
            )
            lum_sample[i] /= constants.L_SUN  # (Lsun)

            # Weighted (with flux) mean wavelength (um)
            mean_sample[i] = np.trapz(
                wavel_high_res * flux_rand, x=wavel_high_res
            ) / np.trapz(flux_rand, x=wavel_high_res)

            # Radial velocity (km s-1)
            vrad_sample[i] = (
                1e-3
                * constants.LIGHT
                * (mean_sample[i] - self.lambda_rest)
                / self.lambda_rest
            )

            # Find full width at half maximum

            spline = InterpolatedUnivariateSpline(
                wavel_high_res, flux_rand - np.max(flux_rand) / 2.0
            )
            root = spline.roots()

            diff = root - mean_sample[i]

            root1 = np.amax(diff[diff < 0.0])
            root2 = np.amin(diff[diff > 0.0])

            fwhm_sample[i] = 1e-3 * constants.LIGHT * (root2 - root1) / mean_sample[i]

            # Add 30 samples to the plot

            if i == 0:
                ax1.plot(
                    wavel_high_res,
                    flux_rand,
                    ls="-",
                    lw=0.5,
                    color="gray",
                    alpha=0.4,
                    label="Random samples",
                )

            elif i < 30:
                ax1.plot(
                    wavel_high_res, flux_rand, ls="-", lw=0.5, color="gray", alpha=0.4
                )

        # Line flux from original, interpolated spectrum

        spec_interp = interp1d(
            self.spectrum[:, 0],
            self.spectrum[:, 1],
            kind=interp_kind,
            bounds_error=False,
        )

        flux_high_res = spec_interp(wavel_high_res)

        line_flux = np.trapz(flux_high_res, x=wavel_high_res)

        ax1.plot(
            wavel_high_res, flux_high_res, color="tab:blue", label="High resolution"
        )

        ax1.legend(loc="upper right", frameon=False, fontsize=12.0)

        print(" [DONE]")

        if plot_filename is None:
            plt.show()
        else:
            plt.savefig(plot_filename, bbox_inches="tight")

        plt.clf()
        plt.close()

        wavel_mean, wavel_std = np.mean(mean_sample), np.std(mean_sample)
        print(f"Mean wavelength (nm) = {1e3*wavel_mean:.2f} +/- {1e3*wavel_std:.2f}")

        fwhm_mean, fwhm_std = np.mean(fwhm_sample), np.std(fwhm_sample)
        print(f"FWHM (km s-1) = {fwhm_mean:.2f} +/- {fwhm_std:.2f}")

        vrad_mean, vrad_std = np.mean(vrad_sample), np.std(vrad_sample)
        print(f"Radial velocity (km s-1) = {vrad_mean:.1f} +/- {vrad_std:.1f}")

        line_error = np.std(flux_sample)
        print(f"Line flux (W m-2) = {line_flux:.2e} +/- {line_error:.2e}")

        line_lum_mean = np.mean(lum_sample)
        line_lum_std = np.std(lum_sample)
        print(f"Line luminosity (Lsun) = {line_lum_mean:.2e} +/- {line_lum_std:.2e}")

        line_lum_mean = np.mean(np.log10(lum_sample))
        line_lum_std = np.std(np.log10(lum_sample))
        print(
            f"Line luminosity log10(L/Lsun) = {line_lum_mean:.2f} +/- {line_lum_std:.2f}"
        )

        if self.hydrogen_line is not None:
            log_acc_sample = np.log10(self.accretion_luminosity(lum_sample))

            log_acc_mean = np.mean(log_acc_sample)
            log_acc_std = np.std(log_acc_sample)

            print(
                "Inflating the uncertainty on the "
                "accretion luminosity by 0.3 dex\n to "
                "account for the model uncertainty "
                "(see Aoyama et al. 2021)..."
            )

            log_acc_std = np.sqrt(log_acc_std**2 + 0.3**2)

            print(
                "Accretion luminosity log10(L/Lsun) = "
                f"{log_acc_mean:.2f} +/- {log_acc_std:.2f}"
            )

            # acc_lum_new = np.log10(acc_lum_sample) + np.random.normal(
            #     0.0, acc_lum_std, size=acc_lum_sample.size
            # )
            #
            # acc_lum_mean = np.mean(10.0**acc_lum_new)
            # acc_lum_std = np.std(10.0**acc_lum_new)
            #
            # print(
            #     f"Accretion luminosity (Lsun): {acc_lum_mean:.2e} +/- {acc_lum_std:.2e}"
            # )

        return line_flux, line_error

    @typechecked
    def fit_gaussian(
        self,
        tag: str,
        min_num_live_points: float = 400,
        bounds: Dict[str, Union[Tuple[float, float]]] = None,
        output: str = "ultranest/",
        plot_filename: Optional[str] = "line_fit.pdf",
        show_status: bool = True,
        double_gaussian: bool = False,
    ) -> None:
        """
        Method for fitting a Gaussian profile to an emission line and
        using ``UltraNest`` for sampling the posterior distributions
        and estimating the evidence.

        Parameters
        ----------
        tag : str
            Database tag where the posterior samples will be stored.
        min_num_live_points : int
            Minimum number of live points (see
            https://johannesbuchner.github.io/UltraNest/issues.html).
        bounds : dict(str, tuple(float, float)), None
            The boundaries that are used for the uniform priors of the
            3 Gaussian parameters (``gauss_amplitude``, ``gauss_mean``,
            and ``gauss_sigma``). Conservative prior boundaries will
            be estimated from the spectrum if the argument is set to
            ``None`` or if any of the required parameters is missing
            in the ``bounds`` dictionary.
        output : str
            Path that is used for the output files from ``UltraNest``.
        plot_filename : str
            Filename for the plot with the best-fit line profile.
            The plot is shown in an interface window if the
            argument is set to ``None``.
        show_status : bool
            Print information about the convergence.
        double_gaussian : bool
            Set to ``True`` for fitting a double instead of a single
            Gaussian. In that case, the ``bounds`` dictionary may also
            contain ``'gauss_amplitude_2'``, ``'gauss_mean_2'``, and
            ``'gauss_sigma_2'`` (otherwise conservative parameter
            boundaries are estimated from the data).

        Returns
        -------
        NoneType
            None
        """

        high_spec_res = 1e5

        @typechecked
        def gaussian_function(
            amplitude: float, mean: float, sigma: float, wavel: np.ndarray
        ):

            return amplitude * np.exp(-0.5 * (wavel - mean) ** 2 / sigma**2)

        # Model parameters

        modelpar = ["gauss_amplitude", "gauss_mean", "gauss_sigma"]

        if double_gaussian:
            modelpar.append("gauss_amplitude_2")
            modelpar.append("gauss_mean_2")
            modelpar.append("gauss_sigma_2")

        # Create a dictionary with the cube indices of the parameters

        cube_index = {}
        for i, item in enumerate(modelpar):
            cube_index[item] = i

        # Check if all prior boundaries are present

        if bounds is None:
            bounds = {}

        if "gauss_amplitude" not in bounds:
            bounds["gauss_amplitude"] = (0.0, 2.0 * np.amax(self.spectrum[:, 1]))

        if "gauss_mean" not in bounds:
            bounds["gauss_mean"] = (self.spectrum[0, 0], self.spectrum[-1, 0])

        if "gauss_sigma" not in bounds:
            bounds["gauss_sigma"] = (0.0, self.spectrum[-1, 0] - self.spectrum[0, 0])

        if double_gaussian:
            if "gauss_amplitude_2" not in bounds:
                bounds["gauss_amplitude_2"] = (0.0, 2.0 * np.amax(self.spectrum[:, 1]))

            if "gauss_mean_2" not in bounds:
                bounds["gauss_mean_2"] = (self.spectrum[0, 0], self.spectrum[-1, 0])

            if "gauss_sigma_2" not in bounds:
                bounds["gauss_sigma_2"] = (
                    0.0,
                    self.spectrum[-1, 0] - self.spectrum[0, 0],
                )

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        @typechecked
        def lnprior_ultranest(cube: np.ndarray) -> np.ndarray:
            """
            Function for transforming the unit cube
            into the parameter cube.

            Parameters
            ----------
            cube : np.ndarray
                Array with unit parameters.

            Returns
            -------
            np.ndarray
                Array with physical parameters.
            """

            params = cube.copy()

            for item in cube_index:
                # Uniform priors for all parameters
                params[cube_index[item]] = (
                    bounds[item][0]
                    + (bounds[item][1] - bounds[item][0]) * params[cube_index[item]]
                )

            return params

        @typechecked
        def lnlike_ultranest(params: np.ndarray) -> np.float64:
            """
            Function for calculating the log-likelihood for the
            sampled parameter cube.

            Parameters
            ----------
            params : np.ndarray
                Cube with physical parameters.

            Returns
            -------
            float
                Log-likelihood.
            """

            data_flux = self.spectrum[:, 1]
            data_var = self.spectrum[:, 2] ** 2

            model_flux = gaussian_function(
                params[cube_index["gauss_amplitude"]],
                params[cube_index["gauss_mean"]],
                params[cube_index["gauss_sigma"]],
                self.spectrum[:, 0],
            )

            if double_gaussian:
                model_flux += gaussian_function(
                    params[cube_index["gauss_amplitude_2"]],
                    params[cube_index["gauss_mean_2"]],
                    params[cube_index["gauss_sigma_2"]],
                    self.spectrum[:, 0],
                )

            chi_sq = -0.5 * (data_flux - model_flux) ** 2 / data_var

            return np.nansum(chi_sq)

        sampler = ultranest.ReactiveNestedSampler(
            modelpar,
            lnlike_ultranest,
            transform=lnprior_ultranest,
            resume="subfolder",
            log_dir=output,
        )

        result = sampler.run(
            show_status=show_status,
            viz_callback=False,
            min_num_live_points=min_num_live_points,
        )

        # Log-evidence

        ln_z = result["logz"]
        ln_z_error = result["logzerr"]
        print(f"Log-evidence = {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Best-fit parameters

        print("Best-fit parameters (mean +/- std):")

        for i, item in enumerate(modelpar):
            mean = np.mean(result["samples"][:, i])
            std = np.std(result["samples"][:, i])

            print(f"   - {item} = {mean:.2e} +/- {std:.2e}")

        # Maximum likelihood sample

        print("Maximum likelihood sample:")

        max_lnlike = result["maximum_likelihood"]["logl"]
        print(f"   - Log-likelihood = {max_lnlike:.2e}")

        for i, item in enumerate(result["maximum_likelihood"]["point"]):
            print(f"   - {modelpar[i]} = {item:.2e}")

        # Posterior samples

        samples = result["samples"]

        # Best-fit model parameters

        model_param = {
            "gauss_amplitude": np.median(samples[:, 0]),
            "gauss_mean": np.median(samples[:, 1]),
            "gauss_sigma": np.median(samples[:, 2]),
        }

        if double_gaussian:
            model_param["gauss_amplitude_2"] = np.median(samples[:, 3])
            model_param["gauss_mean_2"] = np.median(samples[:, 4])
            model_param["gauss_sigma_2"] = np.median(samples[:, 5])

        best_model = read_util.gaussian_spectrum(
            self.wavel_range,
            model_param,
            spec_res=high_spec_res,
            double_gaussian=double_gaussian,
        )

        # Interpolate high-resolution continuum

        if self.continuum_check:
            cont_interp = interp1d(
                self.spectrum[:, 0], self.continuum_flux, bounds_error=False
            )
            cont_high_res = cont_interp(best_model.wavelength)

        else:
            cont_high_res = np.full(best_model.wavelength.shape[0], 0.0)

        # Add FWHM velocity

        modelpar.append("gauss_fwhm")

        gauss_mean = samples[:, 1]  # (um)
        gauss_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * samples[:, 2]  # (um)

        vel_fwhm = 1e-3 * constants.LIGHT * gauss_fwhm / gauss_mean  # (km s-1)
        vel_fwhm = vel_fwhm[..., np.newaxis]

        samples = np.append(samples, vel_fwhm, axis=1)

        # Add line flux and luminosity

        print("Calculating line fluxes...", end="", flush=True)

        modelpar.append("line_flux")
        modelpar.append("log_line_lum")

        if self.hydrogen_line is not None:
            modelpar.append("log_acc_lum")

        line_flux = np.zeros(samples.shape[0])
        line_lum = np.zeros(samples.shape[0])

        if self.continuum_check:
            modelpar.append("line_eq_width")
            eq_width = np.zeros(samples.shape[0])

        for i in range(samples.shape[0]):
            model_param = {
                "gauss_amplitude": samples[i, 0],
                "gauss_mean": samples[i, 1],
                "gauss_sigma": samples[i, 2],
            }

            if double_gaussian:
                model_param["gauss_amplitude_2"] = samples[i, 3]
                model_param["gauss_mean_2"] = samples[i, 4]
                model_param["gauss_sigma_2"] = samples[i, 5]

            model_box = read_util.gaussian_spectrum(
                self.wavel_range,
                model_param,
                spec_res=high_spec_res,
                double_gaussian=double_gaussian,
            )

            line_flux[i] = np.trapz(model_box.flux, x=model_box.wavelength)  # (W m-2)

            line_lum[i] = (
                4.0
                * np.pi
                * (1e3 * constants.PARSEC / self.parallax) ** 2
                * line_flux[i]
            )  # (W)
            line_lum[i] /= constants.L_SUN  # (Lsun)

            if self.continuum_check:
                # Normalize the spectrum to the continuum
                spec_norm = (model_box.flux + cont_high_res) / cont_high_res

                # Check if the flux is NaN (due to interpolation errors at the spectrum edge)
                indices = ~np.isnan(spec_norm)

                eq_width[i] = np.trapz(
                    1.0 - spec_norm[indices], x=model_box.wavelength[indices]
                )  # (um)
                eq_width[i] *= 1e4  # (A)

        print(" [DONE]")

        line_flux = line_flux[..., np.newaxis]
        samples = np.append(samples, line_flux, axis=1)

        line_lum = line_lum[..., np.newaxis]
        samples = np.append(samples, np.log10(line_lum), axis=1)

        if self.hydrogen_line is not None:
            log_acc_lum = np.log10(self.accretion_luminosity(line_lum[:, 0]))

            print(
                "Inflating the uncertainty on the "
                "accretion luminosity by 0.3 dex\n to "
                "account for the model uncertainty "
                "(see Aoyama et al. 2021)..."
            )

            log_acc_lum += np.random.normal(0.0, 0.3, size=log_acc_lum.size)
            log_acc_lum = log_acc_lum[..., np.newaxis]
            samples = np.append(samples, log_acc_lum, axis=1)

        if self.continuum_check:
            eq_width = eq_width[..., np.newaxis]
            samples = np.append(samples, eq_width, axis=1)

        if self.lambda_rest is not None:
            # Radial velocity (km s-1)

            if double_gaussian:
                # Weighted (with Gaussian amplitudes) mean of the central wavelength
                v_fit = (
                    samples[:, 0] * samples[:, 1] + samples[:, 3] * samples[:, 4]
                ) / (samples[:, 0] + samples[:, 3])

            else:
                v_fit = samples[:, 1]

            v_rad = (
                1e-3 * constants.LIGHT * (v_fit - self.lambda_rest) / self.lambda_rest
            )
            v_rad = v_rad[..., np.newaxis]

            modelpar.append("line_vrad")
            samples = np.append(samples, v_rad, axis=1)

        # Log-likelihood

        ln_prob = result["weighted_samples"]["logl"]

        # Log-evidence

        ln_z = result["logz"]
        ln_z_error = result["logzerr"]
        print(f"Log-evidence = {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Dictionary with attributes that will be stored

        attr_dict = {
            "spec_type": "model",
            "spec_name": "gaussian",
            "ln_evidence": (ln_z, ln_z_error),
            "parallax": self.parallax,
        }

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process

            species_db = database.Database()

            species_db.add_samples(
                sampler="ultranest",
                samples=samples,
                ln_prob=ln_prob,
                tag=tag,
                modelpar=modelpar,
                attr_dict=attr_dict,
            )

        # Create plot

        if plot_filename is None:
            print("Plotting best-fit line profile...", end="", flush=True)
        else:
            print(
                f"Plotting best-fit line profile: {plot_filename}...",
                end="",
                flush=True,
            )

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams["axes.axisbelow"] = False

        plt.figure(figsize=(6, 6))
        gs = mpl.gridspec.GridSpec(2, 1)
        gs.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = ax1.twiny()
        ax4 = ax2.twiny()

        ax1.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
            labelbottom=False,
        )

        ax1.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
            labelbottom=False,
        )

        ax2.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
        )

        ax2.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=False,
            bottom=True,
            left=True,
            right=True,
        )

        ax3.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
        )

        ax3.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
        )

        ax4.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
            labeltop=False,
        )

        ax4.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=False,
            left=True,
            right=True,
            labeltop=False,
        )

        ax1.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax2.set_xlabel("Wavelength (µm)", fontsize=16)
        ax2.set_ylabel("Flux (W m$^{-2}$ µm$^{-1}$)", fontsize=16)
        ax3.set_xlabel("Velocity (km s$^{-1}$)", fontsize=16)

        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        ax2.get_xaxis().set_label_coords(0.5, -0.1)
        ax2.get_yaxis().set_label_coords(-0.1, 0.5)
        ax3.get_xaxis().set_label_coords(0.5, 1.12)

        ax1.plot(
            self.spectrum[:, 0],
            self.spectrum[:, 1] + self.continuum_flux,
            color="black",
            label=self.spec_name,
        )

        ax1.plot(
            best_model.wavelength,
            best_model.flux + cont_high_res,
            color="tab:blue",
            label="Best-fit model (continuum + line)",
        )

        ax2.plot(
            self.spectrum[:, 0],
            self.spectrum[:, 1],
            color="black",
            label=self.spec_name,
        )

        ax2.plot(
            best_model.wavelength,
            best_model.flux,
            color="tab:blue",
            label="Best-fit line profile",
        )

        ax3.plot(
            self.spec_vrad, self.spectrum[:, 1] + self.continuum_flux, ls="-", lw=0.0
        )
        ax4.plot(self.spec_vrad, self.spectrum[:, 1], ls="-", lw=0.0)

        ax1.legend(loc="upper left", frameon=False, fontsize=12.0)
        ax2.legend(loc="upper left", frameon=False, fontsize=12.0)

        print(" [DONE]")

        if plot_filename is None:
            plt.show()
        else:
            plt.savefig(plot_filename, bbox_inches="tight")

        plt.clf()
        plt.close()

    @typechecked
    def accretion_luminosity(
        self,
        line_lum: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Method for calculating the accretion luminosity from the
        (hydrogen) line luminosity with the relation from `Aoyama
        et al. (2021) <https://ui.adsabs.harvard.edu/abs/
        2021ApJ...917L..30A/abstract>`_ and extrapolated values
        from `Marleau & Aoyama (2022) <https://ui.adsabs.harvard.
        edu/abs/2022RNAAS...6..262M/abstract>`_.

        Parameters
        ----------
        line_lum : float, np.array
            Line luminosity (:math:`L_\\odot`) or array with line luminosities.

        Returns
        -------
        float, np.ndarray
            Accretion luminosity (:math:`L_\\odot`) or array with
            accretion luminosities.
        """

        with h5py.File(self.database, "r") as h5_file:
            line_names = np.array(h5_file["accretion/hydrogen_lines"], dtype=str)
            coefficients = np.array(h5_file["accretion/coefficients"])

            line_idx = np.argwhere(line_names == self.hydrogen_line)[0][0]
            a_coeff, b_coeff = coefficients[
                line_idx,
            ]

            # Equation C1 in Aoymama et al. (2021)
            log_acc_lum = a_coeff * np.log10(line_lum) + b_coeff

        return 10.**log_acc_lum
