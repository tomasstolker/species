"""
Utility functions for dust cross sections and extinction.
"""

import os
import configparser

from typing import List, Tuple, Union

import h5py
import numpy as np

from typeguard import typechecked
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import lognorm

from species.read.read_filter import ReadFilter
from species.data.misc_data.dust_data import add_cross_sections, add_optical_constants


@typechecked
def check_dust_database() -> str:
    """
    Function to check if the dust data is present in the
    database and add the data if needed.

    Returns
    -------
    str
        Path of the HDF5 database.
    """

    if "SPECIES_CONFIG" in os.environ:
        config_file = os.environ["SPECIES_CONFIG"]
    else:
        config_file = os.path.join(os.getcwd(), "species_config.ini")

    config = configparser.ConfigParser()
    config.read(config_file)

    database_path = config["species"]["database"]
    data_folder = config["species"]["data_folder"]

    with h5py.File(database_path, "r") as hdf5_file:
        # Check if the data are found in 'r' mode because the
        # 'a' mode is not possible when using multiprocessing
        data_found = "dust" in hdf5_file

    if not data_found:
        with h5py.File(database_path, "a") as hdf5_file:
            add_optical_constants(data_folder, hdf5_file)
            add_cross_sections(data_folder, hdf5_file)

    return database_path


@typechecked
def log_normal_distribution(
    radius_g: float, sigma_g: float, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function for returning a log-normal size distribution. See Eq. 9
    in Ackerman & Marley (2001).

    Parameters
    ----------
    radius_g : float
        Mean geometric radius (um).
    sigma_g : float
        Geometric standard deviation (dimensionless).
    n_bins : int
        Number of logarithmically-spaced radius bins.

    Returns
    -------
    np.ndarray
        Number of grains in each radius bin, normalized to a total of
        1 grain.
    np.ndarray
        Widths of the radius bins (um).
    np.ndarray
        Grain radii (um).
    """

    if sigma_g == 1.0:
        # The log-normal distribution is equal to a delta
        # function with sigma_g = 1
        radii = np.array([radius_g])
        r_width = np.array([np.nan])
        dn_grains = np.array([1.0])

    else:
        # Get the radius interval which contains 99.999%
        # of the distribution
        interval = lognorm.interval(
            1.0 - 1e-5, np.log(sigma_g), loc=0.0, scale=radius_g
        )

        # Create bin boundaries (um), so +1 because there
        # are n_bins+1 bin boundaries
        r_bins = np.logspace(np.log10(interval[0]), np.log10(interval[1]), n_bins + 1)

        # Width of the radius bins (um)
        r_width = np.diff(r_bins)

        # Grain radii (um) at which the size distribution is sampled
        radii = (r_bins[1:] + r_bins[:-1]) / 2.0

        # Number of grains per radius bin width, normalized to an
        # integrated value of 1 grain, that is,
        # np.sum(dn_dr*r_width) = 1
        # The log-normal distribution from Ackerman & Marley 2001
        # gives the same result as scipy.stats.lognorm.pdf with
        # s = log(sigma_g) and scale=radius_g
        dn_dr = lognorm.pdf(radii, s=np.log(sigma_g), loc=0.0, scale=radius_g)

        # Number of grains for each radius bin
        dn_grains = dn_dr * r_width

    return dn_grains, r_width, radii


@typechecked
def power_law_distribution(
    exponent: float, radius_min: float, radius_max: float, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function for returning a power-law size distribution.

    Parameters
    ----------
    exponent : float
        Exponent of the power-law size distribution,
        dn/dr = r**exponent.
    radius_min : float
        Minimum grain radius (um).
    radius_max : float
        Maximum grain radius (um).
    n_bins : int
        Number of logarithmically-spaced radius bins.

    Returns
    -------
    np.ndarray
        Number of grains in each radius bin, normalized to a total
        of 1 grain.
    np.ndarray
        Widths of the radius bins (um).
    np.ndarray
        Grain radii (um).
    """

    # Create bin boundaries (um), so +1 because there
    # are n_sizes+1 bin boundaries (um)
    r_bins = np.logspace(np.log10(radius_min), np.log10(radius_max), n_bins + 1)

    # Width of the radius bins (um)
    r_width = np.diff(r_bins)

    # Grains radii (um) at which the size distribution is sampled
    radii = (r_bins[1:] + r_bins[:-1]) / 2.0

    # Number of grains per radius bins size
    dn_dr = radii**exponent

    # Normalize the size distribution to 1 grain
    dn_dr /= np.sum(r_width * dn_dr)

    # Number of grains for each radius bin
    dn_grains = dn_dr * r_width

    return dn_grains, r_width, radii


@typechecked
def dust_cross_section(
    dn_grains: np.ndarray,
    radii: np.ndarray,
    wavelength: float,
    n_index: float,
    k_index: float,
) -> np.float64:
    """
    Function for calculating the extinction cross section for a size
    distribution of dust grains.

    Parameters
    ----------
    dn_grains : np.ndarray
        Number of grains in each radius bin, normalized to a total
        of 1 grain.
    radii : np.ndarray
        Grain radii (um).
    wavelength : float
        Wavelength (um).
    n_index : float
        Real part of the refractive index.
    k_index : float
        Imaginary part of the refractive index.

    Returns
    -------
    float
        Extinction cross section (um2)
    """

    # Importing here because it causes an error with Python 3.10
    import PyMieScatt

    c_ext = 0.0

    for i, item in enumerate(radii):
        # From the PyMieScatt documentation: When using PyMieScatt,
        # pay close attention to the units of the your inputs and
        # outputs. Wavelength and particle diameters are always in
        # nanometers, efficiencies are unitless, cross-sections are
        # in nm2, coefficients are in Mm-1, and size distribution
        # concentration is always in cm-3.
        mie = PyMieScatt.MieQ(
            complex(n_index, k_index),
            wavelength * 1e3,  # (nm)
            2.0 * item * 1e3,  # diameter (nm)
            asDict=True,
            asCrossSection=False,
        )

        if "Qext" in mie:
            c_ext += np.pi * item**2 * mie["Qext"] * dn_grains[i]  # (um2)

        else:
            raise ValueError("Qext not found in the PyMieScatt dictionary.")

    return c_ext  # (um2)


@typechecked
def calc_reddening(
    filters_color: Tuple[str, str],
    extinction: Tuple[str, float],
    composition: str = "MgSiO3",
    structure: str = "crystalline",
    radius_g: float = 1.0,
) -> Tuple[float, float]:
    """
    Function for calculating the reddening of a color given the
    extinction for a given filter. A log-normal size distribution with
    a geometric standard deviation of 2 is used as parametrization for
    the grain sizes (Ackerman & Marley 2001).

    Parameters
    ----------
    filters_color : tuple(str, str)
        Filter names for which the extinction is calculated.
    extinction : str
        Filter name and extinction (mag).
    composition : str
        Dust composition ('MgSiO3' or 'Fe').
    structure : str
        Grain structure ('crystalline' or 'amorphous').
    radius_g : float
        Geometric radius of the grain size distribution (um).

    Returns
    -------
    float
        Extinction (mag) for ``filters_color[0]``.
    float
        Extinction (mag) for ``filters_color[1]``.
    """

    database_path = check_dust_database()

    h5_file = h5py.File(database_path, "r")

    filters = [extinction[0], filters_color[0], filters_color[1]]

    dn_grains, _, radii = log_normal_distribution(radius_g, 2.0, 100)

    c_ext = {}

    for item in filters:
        h5_file.close()

        read_filt = ReadFilter(item)
        filter_wavel = read_filt.mean_wavelength()

        h5_file = h5py.File(database_path, "r")

        if composition == "MgSiO3" and structure == "crystalline":
            for i in range(3):
                data = h5_file[f"dust/mgsio3/crystalline/axis_{i+1}"]

                wavel_index = (np.abs(data[:, 0] - filter_wavel)).argmin()

                # Average cross section of the three axes

                if i == 0:
                    c_ext[item] = (
                        dust_cross_section(
                            dn_grains,
                            radii,
                            data[wavel_index, 0],
                            data[wavel_index, 1],
                            data[wavel_index, 2],
                        )
                        / 3.0
                    )

                else:
                    c_ext[item] += (
                        dust_cross_section(
                            dn_grains,
                            radii,
                            data[wavel_index, 0],
                            data[wavel_index, 1],
                            data[wavel_index, 2],
                        )
                        / 3.0
                    )

        else:
            if composition == "MgSiO3" and structure == "amorphous":
                data = h5_file["dust/mgsio3/amorphous/"]

            elif composition == "Fe" and structure == "crystalline":
                data = h5_file["dust/fe/crystalline/"]

            elif composition == "Fe" and structure == "amorphous":
                data = h5_file["dust/fe/amorphous/"]

            wavel_index = (np.abs(data[:, 0] - filter_wavel)).argmin()

            c_ext[item] += (
                dust_cross_section(
                    dn_grains,
                    radii,
                    data[wavel_index, 0],
                    data[wavel_index, 1],
                    data[wavel_index, 2],
                )
                / 3.0
            )

    h5_file.close()

    return (
        extinction[1] * c_ext[filters_color[0]] / c_ext[extinction[0]],
        extinction[1] * c_ext[filters_color[1]] / c_ext[extinction[0]],
    )


@typechecked
def interp_lognorm(verbose: bool = True) -> Tuple[
    RegularGridInterpolator,
    np.ndarray,
    np.ndarray,
]:
    """
    Function for interpolating the cross sections for dust grains with
    a log-normal size distribution. The returned dictionary contains
    the cross sections for each filter and spectrum.

    Parameters
    ----------
    verbose : bool
        Print information.

    Returns
    -------
    RegularGridInterpolator
        Interpolated extinction cross sections.
    np.ndarray
        Grid points of the geometric mean radius.
    np.ndarray
        Grid points of the geometric standard deviation.
    """

    database_path = check_dust_database()

    with h5py.File(database_path, "r") as h5_file:
        cross_section = np.asarray(
            h5_file["dust/lognorm/mgsio3/crystalline/cross_section"]
        )
        wavelength = np.asarray(h5_file["dust/lognorm/mgsio3/crystalline/wavelength"])
        radius_g = np.asarray(h5_file["dust/lognorm/mgsio3/crystalline/radius_g"])
        sigma_g = np.asarray(h5_file["dust/lognorm/mgsio3/crystalline/sigma_g"])

    if verbose:
        print("Grid boundaries of the dust opacities:")
        print(f"   - Wavelength (um) = {wavelength[0]:.2f} - {wavelength[-1]:.2f}")
        print(
            f"   - Geometric mean radius (um) = {radius_g[0]:.2e} - {radius_g[-1]:.2e}"
        )
        print(
            f"   - Geometric standard deviation = {sigma_g[0]:.2f} - {sigma_g[-1]:.2f}"
        )

    if verbose:
        print("Interpolating dust opacities...", end="")

    cross_sections = RegularGridInterpolator(
        (wavelength, radius_g, sigma_g),
        cross_section,
        method="linear",
        bounds_error=True,
    )

    if verbose:
        print(" [DONE]")

    return cross_sections, radius_g, sigma_g


@typechecked
def interp_powerlaw(verbose: bool = True) -> Tuple[
    RegularGridInterpolator,
    np.ndarray,
    np.ndarray,
]:
    """
    Function for interpolating the cross sections for dust grains with
    a power-law size distribution. The returned dictionary contains
    the cross sections for each filter and spectrum.

    Parameters
    ----------
    verbose : bool
        Print information.

    Returns
    -------
    RegularGridInterpolator
        Interpolate extinction cross sections.
    np.ndarray
        Grid points of the maximum radius.
    np.ndarray
        Grid points of the power-law exponent.
    """

    database_path = check_dust_database()

    with h5py.File(database_path, "r") as h5_file:
        cross_section = np.asarray(
            h5_file["dust/powerlaw/mgsio3/crystalline/cross_section"]
        )
        wavelength = np.asarray(h5_file["dust/powerlaw/mgsio3/crystalline/wavelength"])
        radius_max = np.asarray(h5_file["dust/powerlaw/mgsio3/crystalline/radius_max"])
        exponent = np.asarray(h5_file["dust/powerlaw/mgsio3/crystalline/exponent"])

    if verbose:
        print("Grid boundaries of the dust opacities:")
        print(f"   - Wavelength (um) = {wavelength[0]:.2f} - {wavelength[-1]:.2f}")
        print(f"   - Maximum radius (um) = {radius_max[0]:.2e} - {radius_max[-1]:.2e}")
        print(f"   - Power-law exponent = {exponent[0]:.2f} - {exponent[-1]:.2f}")

    if verbose:
        print("Interpolating dust opacities...", end="")

    cross_sections = RegularGridInterpolator(
        (wavelength, radius_max, exponent),
        cross_section,
        method="linear",
        bounds_error=True,
    )

    if verbose:
        print(" [DONE]")

    return cross_sections, radius_max, exponent


@typechecked
def ism_extinction(
    av_mag: float, rv_red: float, wavelengths: Union[np.ndarray, List[float], float]
) -> np.ndarray:
    """
    Function for calculating the optical and IR extinction
    with the empirical relation from `Cardelli et al. (1989)
    <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_.

    Parameters
    ----------
    av_mag : float
        Extinction (mag) in the $V$ band.
    rv_red : float
        Reddening in the $V$ band, ``R_V = A_V / E(B-V)``.
    wavelengths : np.ndarray, list(float), float
        Array or list with the wavelengths (um) for which
        the extinction is calculated. It is also possible
        to provide a single value as float.

    Returns
    -------
    np.ndarray
        Extinction (mag) at ``wavelengths``.
    """

    if isinstance(wavelengths, float):
        wavelengths = np.array([wavelengths])

    elif isinstance(wavelengths, list):
        wavelengths = np.array(wavelengths)

    x_wavel = 1.0 / wavelengths
    y_wavel = x_wavel - 1.82

    a_coeff = np.zeros(x_wavel.size)
    b_coeff = np.zeros(x_wavel.size)

    indices = np.where(x_wavel < 1.1)[0]

    if len(indices) > 0:
        a_coeff[indices] = 0.574 * x_wavel[indices] ** 1.61
        b_coeff[indices] = -0.527 * x_wavel[indices] ** 1.61

    indices = np.where(x_wavel >= 1.1)[0]

    if len(indices) > 0:
        a_coeff[indices] = (
            1.0
            + 0.17699 * y_wavel[indices]
            - 0.50447 * y_wavel[indices] ** 2
            - 0.02427 * y_wavel[indices] ** 3
            + 0.72085 * y_wavel[indices] ** 4
            + 0.01979 * y_wavel[indices] ** 5
            - 0.77530 * y_wavel[indices] ** 6
            + 0.32999 * y_wavel[indices] ** 7
        )

        b_coeff[indices] = (
            1.41338 * y_wavel[indices]
            + 2.28305 * y_wavel[indices] ** 2
            + 1.07233 * y_wavel[indices] ** 3
            - 5.38434 * y_wavel[indices] ** 4
            - 0.62251 * y_wavel[indices] ** 5
            + 5.30260 * y_wavel[indices] ** 6
            - 2.09002 * y_wavel[indices] ** 7
        )

    return av_mag * (a_coeff + b_coeff / rv_red)


@typechecked
def apply_ism_ext(
    wavelengths: np.ndarray, flux: np.ndarray, v_band_ext: float, v_band_red: float
) -> np.ndarray:
    """
    Function for applying ISM extinction to a spectrum.

    wavelengths : np.ndarray
        Wavelengths (um) of the spectrum.
    flux : np.ndarray
        Fluxes (W m-2 um-1) of the spectrum.
    v_band_ext : float
        Extinction (mag) in the $V$ band.
    v_band_red : float
        Reddening in the $V$ band.

    Returns
    -------
    np.ndarray
        Fluxes (W m-2 um-1) with the extinction applied.
    """

    ext_mag = ism_extinction(v_band_ext, v_band_red, wavelengths)

    return flux * 10.0 ** (-0.4 * ext_mag)


@typechecked
def convert_to_av(
    filter_name: str, filter_ext: float, v_band_red: float = 3.1
) -> float:
    """
    Function for converting the extinction in any filter from
    the `SVO Filter Profile Service <http://svo2.cab.inta-csic.
    es/svo/theory/fps/>`_ to a visual extinction, :math:`A_V`.
    This is done by simply scaling the extinction so at the
    mean wavelength of the filter.

    filter_name : str
        Filter name for which the extinction will be
        converted to a visual extinction (i.e. :math:`A_V`).
    filter_ext : float
        Extinction (mag) for the ``filter_name``.
    v_band_red : float
        Reddening in the $V$ band.

    Returns
    -------
    float
        Visual extinction (i.e. :math:`A_V`) for which the
        extinction in the ``filter_name`` band is ``filter_ext``.
    """

    av_test = 1.0

    # Mean wavelength for filter_name
    read_filt = ReadFilter(filter_name)
    filt_wavel = np.array([read_filt.mean_wavelength()])

    # Calculate test extinction for A_V = 1.0
    # at mean wavelength of filter_name
    ext_ref = ism_extinction(av_test, v_band_red, filt_wavel)[0]

    # Scaling for A_V = 1.0 to the A_V for which
    # extinction of filter_name is filter_ext
    scaling = filter_ext / ext_ref

    # Should be the same as filter_ext
    # filter_ext_test = ism_extinction(scaling * av_test, v_band_red, filt_wavel)[0]

    return scaling * av_test
