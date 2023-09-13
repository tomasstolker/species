"""
Module with functions for plotting the results obtained with the
:class:`~species.analysis.fit_evolution.FitEvolution` class.
"""

from typing import List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import AutoMinorLocator

from species.data import database
from species.read import read_isochrone


@typechecked
def plot_cooling(
    tag: str,
    n_samples: int = 50,
    cooling_param: str = "luminosity",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xscale: Optional[str] = "linear",
    yscale: Optional[str] = "linear",
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (4.0, 2.5),
    output: Optional[str] = None,
) -> Tuple[mpl.figure.Figure, List[List[np.ndarray]], np.ndarray]:
    """
    Function for plotting samples of cooling curves that are
    randomly drawn from the posterior distributions of the
    age and mass parameters that have been estimated with
    :class:`~species.analysis.fit_evolution.FitEvolution`.

    Parameters
    ----------
    tag : str
        Database tag where the samples are stored
    n_samples : int
        Number of randomly drawn cooling curves that will be plotted.
    cooling_param : str
        Type of cooling parameter that will be plotted
        ('luminosity' or 'radius').
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
        Array with the cooling curves. The array contains
        :math:`L/L_\\odot` or radius as function of time
        for each companion and sample, so the shape is
        (n_companions, n_samples, n_ages).
    np.ndarray
        Array with the random indices that have been
        sampled from the posterior distribution.
    """

    plt.close()

    if cooling_param not in ["luminosity", "radius"]:
        raise ValueError(
            "The argument of 'cooling_parameter' is "
            "not valid and should be set to "
            "'luminosity' or 'radius'."
        )

    species_db = database.Database()
    samples_box = species_db.get_samples(tag)

    samples = samples_box.samples
    attr = samples_box.attributes
    n_planets = attr["n_planets"]
    evolution_model = attr["evolution_model"]
    object_lbol = attr["object_lbol"]
    object_radius = attr["object_radius"]
    object_age = (np.mean(samples[:, 0]), np.std(samples[:, 0]))

    read_iso = read_isochrone.ReadIsochrone(evolution_model)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    if output is None:
        print("Plotting cooling curves...", end="", flush=True)
    else:
        print(
            f"Plotting cooling curves: {output}...",
            end="",
            flush=True,
        )

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

    cool_curves = []
    for i in range(n_planets):
        cool_curves.append([])

    for i in range(n_planets):
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

        if cooling_param == "luminosity":
            ax[i].set_ylabel("$\\log(L/L_\\odot)$", fontsize=13)

        elif cooling_param == "radius":
            ax[i].set_ylabel("Radius ($R_\\mathrm{J}$)", fontsize=13)

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
                mass = samples[sample_idx, 1 + planet_idx]

                cool_box = read_iso.get_cooling_curve(mass=mass, ages=None)

                if cooling_param == "luminosity":
                    cool_curves[planet_idx].append(cool_box.log_lum)

                elif cooling_param == "radius":
                    cool_curves[planet_idx].append(cool_box.radius)

                ax[planet_idx].plot(
                    cool_box.age,
                    cool_curves[planet_idx][-1],
                    lw=0.5,
                    color="gray",
                    alpha=0.5,
                )

    for i in range(n_planets):
        if cooling_param == "luminosity":
            ax[i].errorbar(
                object_age[0],
                object_lbol[i][0],
                xerr=object_age[1],
                yerr=object_lbol[i][1],
                color="tab:orange",
            )

        elif cooling_param == "radius" and isinstance(object_radius[i], np.ndarray):
            # Only plot the data if these were provided as optional
            # argument of object_radius when using FitEvolution
            ax[i].errorbar(
                object_age[0],
                object_radius[i][0],
                xerr=object_age[1],
                yerr=object_radius[i][1],
                color="tab:orange",
            )

    if title is not None:
        ax[0].set_title(title, fontsize=18.0)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig, cool_curves, ran_indices


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
) -> Tuple[mpl.figure.Figure, List[List[np.ndarray]], np.ndarray]:
    """
    Function for plotting samples of isochrones that are
    randomly drawn from the posterior distributions of the
    age and mass parameters that have been estimated with
    :class:`~species.analysis.fit_evolution.FitEvolution`.
    For each isochrone, the parameters from a single
    posterior sample are used.

    Parameters
    ----------
    tag : str
        Database tag where the samples are stored
    n_samples : int
        Number of randomly drawn cooling curves that will be plotted.
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

    plt.close()

    species_db = database.Database()
    samples_box = species_db.get_samples(tag)

    samples = samples_box.samples
    attr = samples_box.attributes
    n_planets = attr["n_planets"]
    evolution_model = attr["evolution_model"]
    object_lbol = attr["object_lbol"]
    object_mass = attr["object_mass"]

    read_iso = read_isochrone.ReadIsochrone(evolution_model)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    if output is None:
        print("Plotting isochrones...", end="", flush=True)
    else:
        print(
            f"Plotting isochrones: {output}...",
            end="",
            flush=True,
        )

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
            ax[i].set_xlabel("Mass ($M_\\mathrm{J}$)", fontsize=13)

        ax[i].set_ylabel("$\\log(L/L_\\odot)$", fontsize=13)

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
            age = samples[sample_idx, 0]

            iso_box = read_iso.get_isochrone(age=age, masses=None)

            isochrones[planet_idx].append(iso_box.log_lum)

            ax[planet_idx].plot(
                iso_box.mass,
                isochrones[planet_idx][-1],
                lw=0.5,
                color="gray",
                alpha=0.5,
            )

    for i in range(n_planets):
        ax[i].errorbar(
            object_mass[i][0],
            object_lbol[i][0],
            xerr=object_mass[i][1],
            yerr=object_lbol[i][1],
            color="tab:orange",
        )

    if title is not None:
        ax[0].set_title(title, fontsize=18.0)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig, isochrones, ran_indices
