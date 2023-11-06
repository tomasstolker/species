"""
Utility functions for converting quantities.
"""

from typing import Optional, Tuple, Union

import numpy as np

from typeguard import typechecked

from species.core import constants


@typechecked
def apparent_to_absolute(
    app_mag: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
    distance: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting an apparent magnitude into an absolute
    magnitude. The uncertainty on the distance is propagated into the
    uncertainty on the absolute magnitude.

    Parameters
    ----------
    app_mag : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Apparent magnitude and uncertainty (mag). The returned error
        on the absolute magnitude is set to None if the error on the
        apparent magnitude is set to None, for example
        ``app_mag=(15., None)``.
    distance : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Distance and uncertainty (pc). The error is not propagated
        into the error on the absolute magnitude if set to None, for
        example ``distance=(20., None)``.

    Returns
    -------
    float, np.ndarray
        Absolute magnitude (mag).
    float, np.ndarray, None
        Uncertainty (mag).
    """

    abs_mag = app_mag[0] - 5.0 * np.log10(distance[0]) + 5.0

    if app_mag[1] is not None and distance[1] is not None:
        dist_err = distance[1] * (5.0 / (distance[0] * np.log(10.0)))
        abs_err = np.sqrt(app_mag[1] ** 2 + dist_err**2)

    elif app_mag[1] is not None and distance[1] is None:
        abs_err = app_mag[1]

    else:
        abs_err = None

    return abs_mag, abs_err


@typechecked
def absolute_to_apparent(
    abs_mag: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
    distance: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting an absolute magnitude
    into an apparent magnitude.

    Parameters
    ----------
    abs_mag : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Tuple with the absolute magnitude and uncertainty (mag).
        The uncertainty on the returned apparent magnitude is
        simply adopted from the absolute magnitude. Providing the
        uncertainty is optional and can be set to ``None``.
    distance : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Tuple with the distance and uncertainty (pc). The uncertainty
        is optional and can be set to ``None``. The distance
        uncertainty is currently not used by this function but
        included so it can be implemented at some point into the
        error budget.

    Returns
    -------
    float, np.ndarray
        Apparent magnitude (mag).
    float, np.ndarray, None
        Uncertainty (mag).
    """

    app_mag = abs_mag[0] + 5.0 * np.log10(distance[0]) - 5.0

    return app_mag, abs_mag[1]


@typechecked
def parallax_to_distance(
    parallax: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting from parallax to distance.

    Parameters
    ----------
    parallax : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Parallax and optional uncertainty (mas). The
        uncertainty is not used if set to ``None``,
        for example, ``parallax=(2., None)``.

    Returns
    -------
    float, np.ndarray
        Distance (pc).
    float, np.ndarray, None
        Uncertainty (pc).
    """

    # From parallax (mas) to distance (pc)
    distance = 1e3 / parallax[0]

    if parallax[1] is None:
        distance_error = None

    else:
        distance_minus = distance - 1.0 / ((parallax[0] + parallax[1]) * 1e-3)
        distance_plus = 1.0 / ((parallax[0] - parallax[1]) * 1e-3) - distance
        distance_error = (distance_plus + distance_minus) / 2.0

    return distance, distance_error


@typechecked
def logg_to_mass(
    logg: Union[float, np.ndarray], radius: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Function for converting :math:`\\log(g)` and a radius into a mass.

    Parameters
    ----------
    logg : float, np.ndarray
        Log10 of the surface gravity (cgs).
    radius : float, np.ndarray
        Radius (Rjup).

    Returns
    -------
    float, np.ndarray
        Mass (Mjup).
    """

    surface_grav = 1e-2 * 10.0**logg  # (m s-2)
    radius *= constants.R_JUP  # (m)
    mass = surface_grav * radius**2 / constants.GRAVITY  # (kg)

    return mass / constants.M_JUP


@typechecked
def logg_to_radius(
    logg: Union[float, np.ndarray], mass: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Function for converting :math:`\\log(g)` and a mass into a radius.

    Parameters
    ----------
    logg : float, np.ndarray
        Log10 of the surface gravity (cgs).
    mass : float, np.ndarray
        Mass (Mjup).

    Returns
    -------
    float, np.ndarray
        Radius (Rjup).
    """

    surface_grav = 1e-2 * 10.0**logg  # (m s-2)
    mass_kg = mass * constants.M_JUP  # (kg)
    radius = np.sqrt(mass_kg * constants.GRAVITY / surface_grav)  # (m)

    return radius / constants.R_JUP


@typechecked
def mass_to_logg(
    mass: Union[float, np.ndarray], radius: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Function for converting a mass and radius into :math:`\\log(g)`.

    Parameters
    ----------
    mass : float, np.ndarray
        Mass ($M_\\mathrm{J}$).
    radius : float, np.ndarray
        Radius ($R_\\mathrm{J}$).
    Returns
    -------
    float, np.ndarray
        Surface gravity :math:`\\log(g)`.
    """

    mass *= constants.M_JUP  # (kg)
    radius *= constants.R_JUP  # (m)
    gravity = 1e2 * mass * constants.GRAVITY / radius**2  # (cm s-2)

    return np.log10(gravity)


@typechecked
def luminosity_to_teff(
    luminosity: Union[float, np.ndarray], radius: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Function for converting a luminosity and radius into :math:`T_\\mathrm{eff}`.
    Parameters
    ----------
    luminosity : float, np.ndarray
        Bolometric luminosity ($L_\\odot$).
    radius : float, np.ndarray
        Radius ($R_\\mathrm{J}$).
    Returns
    -------
    float, np.ndarray
        Effective temperature (K).
    """

    radius *= constants.R_JUP  # (Rjup)
    teff = (luminosity / (4.0 * np.pi * radius**2 * constants.SIGMA_SB)) ** 0.25

    return teff
