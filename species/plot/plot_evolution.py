"""
Module with functions for plotting the results obtained with the
:class:`~species.analysis.evolution.PlanetEvolution` class.
"""

from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import AutoMinorLocator

from species.data import database
from species.read import read_evolution


@typechecked
def plot_cooling(
    tag: str,
    n_samples: int = 50,
    age_min: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xscale: Optional[str] = "linear",
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (5.0, 5.0),
    output: Optional[str] = "cooling.pdf",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function for plotting samples of cooling curves that are
    randomly drawn from the posterior distribution of age and
    bulk parameters that has been estimated with
    :class:`~species.analysis.evolution.PlanetEvolution`.

    Parameters
    ----------
    tag : str
        Database tag where the samples are stored
    n_samples : int
        Number of randomly drawn cooling curves that will be plotted.
    age_min float, None

    xlim : tuple(float, float)
        Limits of the wavelength axis.
    ylim : tuple(float, float)
        Limits of the flux axis.
    xscale : str, None
        Scale of the x axis ('linear' or 'log'). The scale is set
        to ``'linear'`` if the argument is set to ``None``.
    title : str
        Title to show at the top of the plot.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str
        Name of the output file.

    Returns
    -------
    np.ndarray
        Array with the cooling curves. The array contains
        :math:`L/L_\\odot` as function of time for each companions
        and sample, so the shape is (n_companions, n_samples, n_ages).
    np.ndarray
        Array with the cooling curves. The array contains
        the radius as function of time for each companions
        and sample, so the shape is (n_companions, n_samples, n_ages).
    np.ndarray
        Array with the random indices that have been
        sampled from the posterior distribution.
    """

    species_db = database.Database()
    samples_box = species_db.get_samples(tag)

    samples = samples_box.samples
    attr = samples_box.attributes
    n_planets = attr["n_planets"]

    read_evol = read_evolution.ReadEvolution()
    read_evol.interpolate_grid()

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["mathtext.fontset"] = "dejavuserif"

    plt.rc("axes", edgecolor="black", linewidth=2.2)
    plt.rcParams["axes.axisbelow"] = False

    for cool_item in ["Lbol", "radius"]:
        if output is None:
            print("Plotting cooling curves...", end="", flush=True)
        else:
            output_split = output.split(".")
            output_update = output_split[0] + f"_{cool_item.lower()}." + output_split[1]

            print(
                f"Plotting cooling curves: {output_update}...",
                end="",
                flush=True,
            )

        plt.figure(1, figsize=figsize)
        gridsp = mpl.gridspec.GridSpec(n_planets, 1)
        gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

        ax = []
        for i in range(n_planets):
            ax.append(plt.subplot(gridsp[i, 0]))

        if xscale is None:
            xscale = "linear"

        for i in range(n_planets):
            ax[i].set_xscale(xscale)

            if cool_item == "Lbol":
                ax[i].set_yscale("log")
            elif cool_item == "radius":
                ax[i].set_yscale("linear")

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

            if cool_item == "Lbol":
                ax[i].set_ylabel("$\\log(L/L_\\odot)$", fontsize=13)
            elif cool_item == "radius":
                ax[i].set_ylabel("Radius ($R_\\mathrm{J}$)", fontsize=13)

            if xlim is not None:
                ax[i].set_xlim(xlim[0], xlim[1])

            if ylim is not None:
                ax[i].set_ylim(ylim[0], ylim[1])

            if offset is not None:
                ax[i].get_xaxis().set_label_coords(0.5, offset[0])
                ax[i].get_yaxis().set_label_coords(offset[1], 0.5)

        indices = np.random.randint(low=0, high=samples.shape[0], size=n_samples)

        # np.savetxt('random_indices.dat', indices, header='Index', fmt='%d')
        # param = np.zeros((50, len(points)-1))

        ages = np.array(read_evol.grid_points["age"])  # (Myr)

        if age_min is not None:
            ages = ages[ages >= age_min]

        cool_curves = np.zeros((n_planets, n_samples, ages.size))

        for k, idx in enumerate(indices):
            for j, item in enumerate(ages):
                for i in range(n_planets):
                    mass = samples[idx, (i * 5) + 1]
                    s_i = samples[idx, (i * 5) + 2]
                    d_frac = samples[idx, (i * 5) + 3]
                    y_frac = samples[idx, (i * 5) + 4]
                    m_core = samples[idx, (i * 5) + 5]

                    if cool_item == "Lbol":
                        cool_curves[i, k, j] = 10.0 ** read_evol.interp_lbol(
                            [item, mass, s_i, d_frac, y_frac, m_core]
                        )
                    elif cool_item == "radius":
                        cool_curves[i, k, j] = read_evol.interp_radius(
                            [item, mass, s_i, d_frac, y_frac, m_core]
                        )

                    # param[k, :] = np.array([mass, s_i, d_frac, y_frac, m_core])

                    if np.isnan(cool_curves[i, k, j]):
                        raise ValueError(
                            f"The interpolated luminosity is "
                            f"NaN for the following "
                            f"parameters: {item}, {mass},"
                            f"{s_i}, {d_frac}, {y_frac}"
                        )

            for i in range(n_planets):
                ax[i].plot(
                    ages,
                    cool_curves[i, k, :],
                    lw=0.5,
                    color="gray",
                    alpha=0.5,
                )

            if cool_item == "Lbol":
                cool_lbol = np.copy(cool_curves)
            elif cool_item == "radius":
                cool_radius = np.copy(cool_curves)

        object_age = (np.mean(samples[:, 0]), np.std(samples[:, 0]))
        object_lbol = attr["object_lbol"]
        object_radius = attr["object_radius"]

        for i in range(n_planets):
            if cool_item == "Lbol":
                lbol_err = (
                    10.0 ** (object_lbol[i][0] + object_lbol[i][1])
                    - 10.0 ** object_lbol[i][0]
                )
                ax[i].errorbar(
                    object_age[0],
                    10.0 ** object_lbol[i][0],
                    xerr=object_age[1],
                    yerr=lbol_err,
                    color="tab:orange",
                )

            elif cool_item == "radius" and isinstance(object_radius[i], np.ndarray):
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
            plt.savefig(output_update, bbox_inches="tight")

        print(" [DONE]")

        plt.clf()
        plt.close()

    return cool_lbol, cool_radius, indices


@typechecked
def plot_isochrones(
    tag: str,
    n_samples: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (5.0, 5.0),
    output: Optional[str] = "isochrones.pdf",
) -> Tuple[np.array, np.array]:
    """
    Function for plotting samples of isochrones that are
    randomly drawn from the posterior distribution of age
    and bulk parameters that has been estimated with
    :class:`~species.analysis.evolution.PlanetEvolution`.
    For each isochrone, the parameters from a single
    posterior sample are used.


    Parameters
    ----------
    tag : str
        Database tag where the samples are stored
    n_samples : int
        Number of randomly drawn cooling curves that will be plotted.
    xlim : tuple(float, float)
        Limits of the wavelength axis.
    ylim : tuple(float, float)
        Limits of the flux axis.
    title : str
        Title to show at the top of the plot.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str
        Name of the output file.

    Returns
    -------
    np.ndarray
        Array with the isochrones. The array contains
        :math:`L/L_\\odot` as function of time for each companions
        and sample, so the shape is (n_companions, n_samples, n_masses).
    np.ndarray
        Array with the random indices that have been
        sampled from the posterior distribution.
    """

    species_db = database.Database()
    samples_box = species_db.get_samples(tag)

    samples = samples_box.samples
    attr = samples_box.attributes
    n_planets = attr["n_planets"]

    read_evol = read_evolution.ReadEvolution()
    read_evol.interpolate_grid()

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["mathtext.fontset"] = "dejavuserif"

    plt.rc("axes", edgecolor="black", linewidth=2.2)
    plt.rcParams["axes.axisbelow"] = False

    if output is None:
        print("Plotting isochrones...", end="", flush=True)
    else:
        print(
            f"Plotting isochrones: {output}...",
            end="",
            flush=True,
        )

    plt.figure(1, figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(n_planets, 1)
    gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

    ax = []
    for i in range(n_planets):
        ax.append(plt.subplot(gridsp[i, 0]))

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

        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))

        if i == n_planets - 1:
            ax[i].set_xlabel("Mass ($M_\\mathrm{J}$)", fontsize=13)

        ax[i].set_ylabel("$\\log(L/L_\\odot)$", fontsize=13)

        ax[i].set_xscale("linear")
        ax[i].set_yscale("log")

        if xlim is not None:
            ax[i].set_xlim(xlim[0], xlim[1])

        if ylim is not None:
            ax[i].set_ylim(ylim[0], ylim[1])

        if offset is not None:
            ax[i].get_xaxis().set_label_coords(0.5, offset[0])
            ax[i].get_yaxis().set_label_coords(offset[1], 0.5)

    indices = np.random.randint(low=0, high=samples.shape[0], size=n_samples)

    masses = np.array(read_evol.grid_points["mass"])  # (Myr)

    isochrones = np.zeros((n_planets, n_samples, masses.size))

    for k, idx in enumerate(indices):
        for j, item in enumerate(masses):
            for i in range(n_planets):
                age = samples[idx, 0]
                s_i = samples[idx, (i * 5) + 2]
                d_frac = samples[idx, (i * 5) + 3]
                y_frac = samples[idx, (i * 5) + 4]
                m_core = samples[idx, (i * 5) + 5]

                isochrones[i, k, j] = 10.0 ** read_evol.interp_lbol(
                    [age, item, s_i, d_frac, y_frac, m_core]
                )

                if np.isnan(isochrones[i, k, j]):
                    raise ValueError(
                        f"The interpolated luminosity is "
                        f"NaN for the following "
                        f"parameters: {age}, {item},"
                        f"{s_i}, {d_frac}, {y_frac}"
                    )

        for i in range(n_planets):
            ax[i].plot(
                masses,
                isochrones[i, k, :],
                lw=0.5,
                color="gray",
                alpha=0.5,
            )

    
    object_lbol = attr["object_lbol"]
    object_mass = attr["object_mass"]

    for i in range(n_planets):
        lbol_err = (
            10.0 ** (object_lbol[i][0] + object_lbol[i][1]) - 10.0 ** object_lbol[i][0]
        )

        ax[i].errorbar(
            object_mass[i][0],
            10.0 ** object_lbol[i][0],
            xerr=object_mass[i][1],
            yerr=lbol_err,
            color="tab:orange",
        )

    if title is not None:
        ax[0].set_title(title, fontsize=18.0)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    plt.clf()
    plt.close()

    return isochrones, indices
