"""
Module with functions for plotting the results obtained with the
:class:`~species.fit.fit_evolution.FitEvolution` class.
"""

from typing import List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import AutoMinorLocator

from species.read.read_isochrone import ReadIsochrone
from species.util.core_util import print_section


@typechecked
def plot_cooling(
    tag: str,
    n_samples: int = 50,
    cooling_param: str = "log_lum",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xscale: Optional[str] = "linear",
    yscale: Optional[str] = "linear",
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (4.0, 2.5),
    output: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, List[List[List[np.ndarray]]], np.ndarray]:
    """
    Function for plotting samples of cooling tracks that are
    randomly drawn from the posterior distributions of the
    age and mass parameters that have been estimated with
    :class:`~species.fit.fit_evolution.FitEvolution`.

    Parameters
    ----------
    tag : str
        Database tag where the samples are stored
    n_samples : int
        Number of randomly drawn cooling tracks that will be plotted.
    cooling_param : str
        Type of cooling parameter that will be plotted
        ('log_lum', 'radius', 'teff', or 'logg').
    xlim : tuple(float, float), None
        Limits of the wavelength axis. Automatic limits are used if
        the argument is set to ``None``.
    ylim : tuple(float, float), None
        Limits of the flux axis. Automatic limits are used if
        the argument is set to ``None``.
    xscale : str, None
        Scale of the x axis ('linear' or 'log'). The scale is set
        to ``'linear'`` if the argument is set to ``None``.
    yscale : str, None
        Scale of the x axis ('linear' or 'log'). The scale is set
        to ``'linear'`` if the argument is set to ``None``.
    title : str
        Title to show at the top of the plot.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    np.ndarray
        Array with the cooling tracks. The array contains
        :math:`L/L_\\odot` or radius as function of time
        for each companion and sample, so the shape is
        (n_companions, n_samples, n_ages).
    np.ndarray
        Array with the random indices that have been
        sampled from the posterior distribution.
    """

    print_section("Plot cooling tracks")

    print(f"Database tag: {tag}")
    print(f"Number of samples: {n_samples}")
    print(f"Model parameters: {cooling_param}")

    plt.close()

    if cooling_param not in ["log_lum", "luminosity", "radius", "teff", "logg"]:
        raise ValueError(
            "The argument of 'cooling_parameter' is "
            "not valid and should be set to "
            "'log_lum', 'radius', 'teff', or 'logg'."
        )

    from species.data.database import Database

    species_db = Database()
    samples_box = species_db.get_samples(tag)

    samples = samples_box.samples
    attr = samples_box.attributes
    n_param = attr["n_param"]
    n_planets = attr["n_planets"]
    model_name = attr["model_name"]
    log_lum = attr["log_lum"]
    age_prior = attr["age_prior"]
    radius_prior = attr["radius_prior"]

    param_indices = {}
    for i in range(n_param):
        param_indices[samples_box.attributes[f"parameter{i}"]] = i

    if np.isnan(age_prior[0]):
        param_idx = samples_box.parameters.index("age")
        age_prior = np.percentile(samples[:, param_idx], [50.0, 16.0, 84.0])
    else:
        # Asymmetric normal prior set in FitEvolution
        age_prior = [
            age_prior[0],
            age_prior[0] + age_prior[1],
            age_prior[0] + age_prior[2],
        ]

    read_iso = ReadIsochrone(model_name)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig = plt.figure(1, figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(n_planets, 1)
    gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

    ax = []
    for i in range(n_planets):
        ax.append(plt.subplot(gridsp[i, 0]))

    if xscale is None:
        xscale = "linear"

    if yscale is None:
        yscale = "linear"

    cool_tracks = []
    for i in range(n_planets):
        cool_tracks.append([])

    for i in range(n_planets):
        if not isinstance(radius_prior[i], np.ndarray) and np.isnan(radius_prior[i]):
            param_idx = samples_box.parameters.index(f"radius_{i}")
            radius_tmp = np.percentile(samples[:, param_idx], [50.0, 16.0, 84.0])
        else:
            radius_tmp = [
                radius_prior[i][0],
                radius_prior[i][0] - radius_prior[i][1],
                radius_prior[i][0] + radius_prior[i][1],
            ]

        param_idx = samples_box.parameters.index(f"teff_{i}")
        teff_tmp = np.percentile(samples[:, param_idx], [50.0, 16.0, 84.0])

        param_idx = samples_box.parameters.index(f"logg_{i}")
        logg_tmp = np.percentile(samples[:, param_idx], [50.0, 16.0, 84.0])

        ax[i].set_xscale(xscale)
        ax[i].set_yscale(yscale)

        labelbottom = bool(i == n_planets - 1)

        ax[i].tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelbottom=labelbottom,
        )

        ax[i].tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelbottom=labelbottom,
        )

        if xscale == "linear":
            ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))

        if i == n_planets - 1:
            ax[i].set_xlabel("Age (Myr)", fontsize=13)

        if cooling_param in ["luminosity", "log_lum"]:
            ax[i].set_ylabel(r"$\log(L/L_\odot)$", fontsize=13)

        elif cooling_param == "radius":
            ax[i].set_ylabel(r"Radius ($R_\mathrm{J}$)", fontsize=13)

        elif cooling_param == "teff":
            ax[i].set_ylabel(r"$T_\mathrm{eff}$ (K)", fontsize=13)

        elif cooling_param == "logg":
            ax[i].set_ylabel(r"$\log\,g$", fontsize=13)

        if xlim is not None:
            ax[i].set_xlim(xlim[0], xlim[1])

        if ylim is not None:
            ax[i].set_ylim(ylim[0], ylim[1])

        if offset is not None:
            ax[i].get_xaxis().set_label_coords(0.5, offset[0])
            ax[i].get_yaxis().set_label_coords(offset[1], 0.5)

        ran_indices = np.random.randint(low=0, high=samples.shape[0], size=n_samples)

        for sample_idx in ran_indices:
            for planet_idx in range(n_planets):
                mass = samples[sample_idx, param_indices[f"mass_{planet_idx}"]]

                if f"s_init_{planet_idx}" in param_indices:
                    s_init = samples[sample_idx, param_indices[f"s_init_{planet_idx}"]]
                else:
                    s_init = None

                cool_box = read_iso.get_cooling_track(
                    mass=mass, ages=None, s_init=s_init
                )

                if cooling_param in ["luminosity", "log_lum"]:
                    cool_tracks[planet_idx].append([cool_box.age, cool_box.log_lum])

                elif cooling_param == "radius":
                    cool_tracks[planet_idx].append([cool_box.age, cool_box.radius])

                elif cooling_param == "teff":
                    cool_tracks[planet_idx].append([cool_box.age, cool_box.teff])

                elif cooling_param == "logg":
                    cool_tracks[planet_idx].append([cool_box.age, cool_box.logg])

                if cool_tracks[planet_idx][-1][1] is None:
                    raise ValueError(
                        f"The selected parameter, '{cooling_param}', "
                        f"is not part of the '{model_name}' "
                        "evolutionary model grid.")

                ax[planet_idx].plot(
                    cool_tracks[planet_idx][-1][0],
                    cool_tracks[planet_idx][-1][1],
                    lw=0.5,
                    color="gray",
                    alpha=0.5,
                )

    for i in range(n_planets):
        if cooling_param in ["luminosity", "log_lum"]:
            ax[i].errorbar(
                [age_prior[0]],
                [log_lum[i][0]],
                xerr=[
                    [age_prior[0] - np.abs(age_prior[1])],
                    [age_prior[2] - age_prior[0]],
                ],
                yerr=[log_lum[i][1]],
                color="tab:orange",
            )

        elif cooling_param == "radius":
            ax[i].errorbar(
                [age_prior[0]],
                [radius_tmp[0]],
                xerr=[
                    [age_prior[0] - np.abs(age_prior[1])],
                    [age_prior[2] - age_prior[0]],
                ],
                yerr=[[radius_tmp[0] - radius_tmp[1]], [radius_tmp[2] - radius_tmp[0]]],
                color="tab:orange",
            )

        elif cooling_param == "teff":
            ax[i].errorbar(
                [age_prior[0]],
                [teff_tmp[0]],
                xerr=[
                    [age_prior[0] - np.abs(age_prior[1])],
                    [age_prior[2] - age_prior[0]],
                ],
                yerr=[[teff_tmp[0] - teff_tmp[1]], [teff_tmp[2] - teff_tmp[0]]],
                color="tab:orange",
            )

        elif cooling_param == "logg":
            ax[i].errorbar(
                [age_prior[0]],
                [logg_tmp[0]],
                xerr=[
                    [age_prior[0] - np.abs(age_prior[1])],
                    [age_prior[2] - age_prior[0]],
                ],
                yerr=[[logg_tmp[0] - logg_tmp[1]], [logg_tmp[2] - logg_tmp[0]]],
                color="tab:orange",
            )

    if title is not None:
        ax[0].set_title(title, fontsize=18.0)

    if output is None:
        plt.show()
    else:
        print(f"\nOutput: {output}")
        plt.savefig(output, bbox_inches="tight")

    return fig, cool_tracks, ran_indices


@typechecked
def plot_isochrones(
    tag: str,
    n_samples: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xscale: Optional[str] = "linear",
    yscale: Optional[str] = "linear",
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (4.0, 2.5),
    output: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, List[List[List[np.ndarray]]], np.ndarray]:
    """
    Function for plotting samples of isochrones that are
    randomly drawn from the posterior distributions of the
    age and mass parameters that have been estimated with
    :class:`~species.fit.fit_evolution.FitEvolution`.
    For each isochrone, the parameters from a single
    posterior sample are used.

    Parameters
    ----------
    tag : str
        Database tag where the samples are stored
    n_samples : int
        Number of randomly drawn cooling tracks that will be plotted.
    xlim : tuple(float, float), None
        Limits of the wavelength axis. Automatic limits are used if
        the argument is set to ``None``.
    ylim : tuple(float, float), None
        Limits of the flux axis. Automatic limits are used if
        the argument is set to ``None``.
    xscale : str, None
        Scale of the x axis ('linear' or 'log'). The scale is set
        to ``'linear'`` if the argument is set to ``None``.
    yscale : str, None
        Scale of the x axis ('linear' or 'log'). The scale is set
        to ``'linear'`` if the argument is set to ``None``.
    title : str
        Title to show at the top of the plot.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    np.ndarray
        Array with the isochrones. The array contains
        :math:`L/L_\\odot` as function of time for each companions
        and sample, so the shape is (n_companions, n_samples, n_masses).
    np.ndarray
        Array with the random indices that have been
        sampled from the posterior distribution.
    """

    print_section("Plot isochrones")

    print(f"Database tag: {tag}")
    print(f"Number of samples: {n_samples}")

    plt.close()

    from species.data.database import Database

    species_db = Database()
    samples_box = species_db.get_samples(tag)

    samples = samples_box.samples
    attr = samples_box.attributes
    n_param = samples_box.attributes["n_param"]
    n_planets = attr["n_planets"]
    model_name = attr["model_name"]
    log_lum = attr["log_lum"]
    age_prior = attr["age_prior"]
    mass_prior = attr["mass_prior"]

    param_indices = {}
    for i in range(n_param):
        param_indices[samples_box.attributes[f"parameter{i}"]] = i

    if np.isnan(age_prior[0]):
        param_idx = samples_box.parameters.index("age")
        age_prior = np.percentile(samples[:, param_idx], [50.0, 16.0, 84.0])
    else:
        # Asymmetric normal prior set in FitEvolution
        age_prior = [
            age_prior[0],
            age_prior[0] + age_prior[1],
            age_prior[0] + age_prior[2],
        ]

    read_iso = ReadIsochrone(model_name)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig = plt.figure(1, figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(n_planets, 1)
    gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

    ax = []
    for i in range(n_planets):
        ax.append(plt.subplot(gridsp[i, 0]))

    if xscale is None:
        xscale = "linear"

    if yscale is None:
        yscale = "linear"

    isochrones = []
    for i in range(n_planets):
        isochrones.append([])

    for i in range(n_planets):
        labelbottom = bool(i == n_planets - 1)

        ax[i].tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelbottom=labelbottom,
        )

        ax[i].tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelbottom=labelbottom,
        )

        if xscale == "linear":
            ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))

        if i == n_planets - 1:
            ax[i].set_xlabel(r"Mass ($M_\mathrm{J}$)", fontsize=13)

        ax[i].set_ylabel(r"$\log(L/L_\odot)$", fontsize=13)

        ax[i].set_xscale(xscale)
        ax[i].set_yscale(yscale)

        if xlim is not None:
            ax[i].set_xlim(xlim[0], xlim[1])

        if ylim is not None:
            ax[i].set_ylim(ylim[0], ylim[1])

        if offset is not None:
            ax[i].get_xaxis().set_label_coords(0.5, offset[0])
            ax[i].get_yaxis().set_label_coords(offset[1], 0.5)

    ran_indices = np.random.randint(low=0, high=samples.shape[0], size=n_samples)

    for sample_idx in ran_indices:
        for planet_idx in range(n_planets):
            age = samples[sample_idx, param_indices["age"]]

            if f"s_init_{planet_idx}" in param_indices:
                s_init = samples[sample_idx, param_indices[f"s_init_{planet_idx}"]]
            else:
                s_init = None

            iso_box = read_iso.get_isochrone(
                age=age, masses=None, s_init=s_init, param_interp=["log_lum"]
            )

            isochrones[planet_idx].append([iso_box.mass, iso_box.log_lum])

            ax[planet_idx].plot(
                isochrones[planet_idx][-1][0],
                isochrones[planet_idx][-1][1],
                lw=0.5,
                color="gray",
                alpha=0.5,
            )

    for i in range(n_planets):
        if not isinstance(mass_prior[i], np.ndarray) and np.isnan(mass_prior[i]):
            param_idx = samples_box.parameters.index(f"mass_{i}")
            mass_tmp = np.percentile(samples[:, param_idx], [50.0, 16.0, 84.0])
        else:
            mass_tmp = [
                mass_prior[i][0],
                mass_prior[i][0] - mass_prior[i][1],
                mass_prior[i][0] + mass_prior[i][1],
            ]

        ax[i].errorbar(
            [mass_tmp[0]],
            [log_lum[i][0]],
            xerr=[[mass_tmp[0] - mass_tmp[1]], [mass_tmp[2] - mass_tmp[0]]],
            yerr=[log_lum[i][1]],
            color="tab:orange",
        )

    if title is not None:
        ax[0].set_title(title, fontsize=18.0)

    if output is None:
        print(f"\nOutput: {output}")
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    return fig, isochrones, ran_indices
