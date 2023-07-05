"""
Module with functions for plotting results from a spectral
analysis that compares data with a library of empirical
spectra or a grid of model spectra.
"""

import configparser
import os
import warnings

from typing import List, Optional, Tuple

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d, RegularGridInterpolator
from typeguard import typechecked

from species.core import constants
from species.read import read_model, read_object
from species.util import dust_util, plot_util, read_util


@typechecked
def plot_statistic(
    tag: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (4.0, 2.5),
    output: Optional[str] = None,
) -> mpl.figure.Figure:
    """
    Function for plotting the goodness-of-fit statistic from a
    comparison with an empirical spectral library with
    :class:`~species.analysis.compare_spectra.CompareSpectra.spectral_type`
    that enables a determination of the spectral type

    Parameters
    ----------
    tag : str
        Database tag where the results from the empirical comparison with
        :class:`~species.analysis.compare_spectra.CompareSpectra.spectral_type`
        are stored.
    xlim : tuple(float, float)
        Limits of the spectral type axis in numbers (i.e.
        0=M0, 5=M5, 10=L0, etc.).
    ylim : tuple(float, float)
        Limits of the goodness-of-fit axis.
    title : str
        Plot title.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.

    Returns
    -------
    NoneType
        None
    """

    if output is None:
        print("Plotting goodness-of-fit statistic...", end="")
    else:
        print(f"Plotting goodness-of-fit statistic: {output}...", end="")

    config_file = os.path.join(os.getcwd(), "species_config.ini")

    config = configparser.ConfigParser()
    config.read(config_file)

    db_path = config["species"]["database"]

    h5_file = h5py.File(db_path, "r")

    dset = h5_file[f"results/empirical/{tag}/names"]

    names = np.array(dset)
    sptypes = np.array(h5_file[f"results/empirical/{tag}/sptypes"])
    g_fit = np.array(h5_file[f"results/empirical/{tag}/goodness_of_fit"])

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.axisbelow"] = False

    fig = plt.figure(figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    ax.tick_params(
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
    )

    ax.tick_params(
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
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel("Spectral type", fontsize=13)
    ax.set_ylabel(r"G$_\mathregular{k}$", fontsize=13)

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

    if title is not None:
        ax.set_title(title, y=1.02, fontsize=13)

    ax.set_xticks(np.linspace(0.0, 30.0, 7, endpoint=True))
    ax.set_xticklabels(["M0", "M5", "L0", "L5", "T0", "T5", "Y0"])

    if xlim is None:
        ax.set_xlim(0.0, 30.0)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    sptype_num = np.zeros(names.shape[0])

    for i, item in enumerate(sptypes):
        for j in range(10):
            if not isinstance(item, str):
                item = item.decode("utf-8")

            if item == f"M{j}":
                sptype_num[i] = float(j)

            elif item == f"L{j}":
                sptype_num[i] = float(10 + j)

            elif item == f"T{j}":
                sptype_num[i] = float(20 + j)

    ax.plot(
        sptype_num,
        g_fit,
        "s",
        ms=3.0,
        mew=0.5,
        color="lightgray",
        markeredgecolor="darkgray",
    )

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    h5_file.close()

    print(" [DONE]")

    return fig


@typechecked
def plot_empirical_spectra(
    tag: str,
    n_spectra: Optional[int] = None,
    flux_offset: Optional[float] = None,
    label_pos: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (4.0, 2.5),
    output: Optional[str] = None,
) -> mpl.figure.Figure:
    """
    Function for plotting the results from the empirical
    spectrum comparison.

    Parameters
    ----------
    tag : str
        Database tag where the results from the empirical
        comparison with
        :class:`~species.analysis.compare_spectra.CompareSpectra.spectral_type`
        are stored.
    n_spectra : int, None
        The number of spectra with the lowest goodness-of-fit
        statistic that will be plotted in comparison with the data.
        All spectra are selected if the argument is set to ``None``.
    label_pos : tuple(float, float), None
        Position for the name labels. Should be provided as (x, y)
        for the lowest spectrum. The ``flux_offset`` will be applied
        to the remaining spectra. The labels are only plotted if the
        argument of both ``label_pos`` and ``flux_offset`` are not
        ``None``.
    flux_offset : float, None
        Offset to be applied such that the spectra do not overlap. No
        offset is applied if the argument is set to ``None``.
    xlim : tuple(float, float)
        Limits of the spectral type axis.
    ylim : tuple(float, float)
        Limits of the goodness-of-fit axis.
    title : str
        Plot title.
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
    """

    if output is None:
        print("Plotting empirical spectra comparison...", end="")
    else:
        print(f"Plotting empirical spectra comparison: {output}...", end="")

    if flux_offset is None:
        flux_offset = 0.0

    config_file = os.path.join(os.getcwd(), "species_config.ini")

    config = configparser.ConfigParser()
    config.read(config_file)

    db_path = config["species"]["database"]

    h5_file = h5py.File(db_path, "r")

    dset = h5_file[f"results/empirical/{tag}/names"]

    object_name = dset.attrs["object_name"]
    spec_library = dset.attrs["spec_library"]
    n_spec_name = dset.attrs["n_spec_name"]

    spec_name = []
    for i in range(n_spec_name):
        spec_name.append(dset.attrs[f"spec_name{i}"])

    names = np.array(dset)
    flux_scaling = np.array(h5_file[f"results/empirical/{tag}/flux_scaling"])
    av_ext = np.array(h5_file[f"results/empirical/{tag}/av_ext"])

    rad_vel = np.array(h5_file[f"results/empirical/{tag}/rad_vel"])
    rad_vel *= 1e3  # (m s-1)

    if n_spectra is None:
        n_spectra = names.size

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.axisbelow"] = False

    fig = plt.figure(figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    ax.tick_params(
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
    )

    ax.tick_params(
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
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel("Wavelength (\N{GREEK SMALL LETTER MU}m)", fontsize=13)

    if flux_offset == 0.0:
        ax.set_ylabel(
            r"$\mathregular{F}_\lambda$"
            + " (W m$^{-2}$ \N{GREEK SMALL LETTER MU}m$^{-1}$)",
            fontsize=11,
        )
    else:
        ax.set_ylabel(
            r"$\mathregular{F}_\lambda$ (W m$^{-2}$"
            + " \N{GREEK SMALL LETTER MU}m$^{-1}$) + offset",
            fontsize=11,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

    if title is not None:
        ax.set_title(title, y=1.02, fontsize=13)

    read_obj = read_object.ReadObject(object_name)

    obj_spec = []
    obj_res = []

    for item in spec_name:
        obj_spec.append(read_obj.get_spectrum()[item][0])
        obj_res.append(read_obj.get_spectrum()[item][3])

    if flux_offset == 0.0:
        for spec_item in obj_spec:
            ax.plot(spec_item[:, 0], spec_item[:, 1], "-", lw=0.5, color="black")

    for i in range(n_spectra):
        if isinstance(names[i], str):
            name_item = names[i]
        else:
            name_item = names[i].decode("utf-8")

        dset = h5_file[f"spectra/{spec_library}/{name_item}"]
        sptype = dset.attrs["sptype"]
        spectrum = np.asarray(dset)

        if flux_offset != 0.0:
            for spec_item in obj_spec:
                ax.plot(
                    spec_item[:, 0],
                    (n_spectra - i - 1) * flux_offset + spec_item[:, 1],
                    "-",
                    lw=0.5,
                    color="black",
                )

        for j, spec_item in enumerate(obj_spec):
            ism_ext = dust_util.ism_extinction(av_ext[i], 3.1, spectrum[:, 0])
            ext_scaling = 10.0 ** (-0.4 * ism_ext)

            wavel_shifted = (
                spectrum[:, 0] + spectrum[:, 0] * rad_vel[i] / constants.LIGHT
            )

            flux_smooth = read_util.smooth_spectrum(
                wavel_shifted,
                spectrum[:, 1] * ext_scaling,
                spec_res=obj_res[j],
                force_smooth=True,
            )

            interp_spec = interp1d(
                spectrum[:, 0], flux_smooth, fill_value="extrapolate"
            )

            indices = np.where(
                (spec_item[:, 0] > np.amin(spectrum[:, 0]))
                & (spec_item[:, 0] < np.amax(spectrum[:, 0]))
            )[0]

            flux_resample = interp_spec(spec_item[indices, 0])

            ax.plot(
                spec_item[indices, 0],
                (n_spectra - i - 1) * flux_offset + flux_scaling[i][j] * flux_resample,
                color="tomato",
                lw=0.5,
            )

        if label_pos is not None and flux_offset != 0.0:
            label_text = name_item + ", " + sptype

            if av_ext[i] != 0.0:
                label_text += r", A$_\mathregular{V}$ = " + f"{av_ext[i]:.1f}"

            ax.text(
                label_pos[0],
                label_pos[1] + (n_spectra - i - 1) * flux_offset,
                label_text,
                fontsize=8.0,
                ha="left",
            )

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    h5_file.close()

    print(" [DONE]")

    return fig


@typechecked
def plot_grid_statistic(
    tag: str,
    upsample: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (4.0, 2.5),
    output: Optional[str] = None,
    extra_param: Optional[str] = None,
    nlevels_main: int = 20,
    nlevels_extra: int = 10,
) -> mpl.figure.Figure:
    """
    Function for plotting the results from the comparison with
    a grid of empirical or model spectra

    Parameters
    ----------
    tag : str
        Database tag where the results from the comparison with
        :class:`~species.analysis.compare_spectra.CompareSpectra` are stored.
    upsample : bool
        Upsample the goodness-of-fit grid to a higher resolution
        for a smoother appearance.
    xlim : tuple(float, float), None
        Limits of the x-axis (spectral type or effective temperature).
    ylim : tuple(float, float), None
        Limits of the y-axis.
    title : str, None
        Title that is shown above the plot.
    offset : tuple(float, float), None
        Offset for the label for the x- and y-axis.
    figsize : tuple(float, float), None
        Figure size.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.
    extra_param : str, None
        Extra parameter to be overplotted with contours. The argument
        can be set to any of the atmospheric parameters that were used
        for the comparison, for example, 'teff', 'logg', 'feh', or
        'radius'. Optionally, the argument can be set to 'ism_ext' in
        case the ``av_points`` parameter of the
        :func:`~species.analysis.compare_spectra.CompareSpectra.compare_model`
        method was used. Extra contours are not plotted if the
        argument is set to ``None``.
    nlevels_main : int
        Number of contour levels for the main plot.
    nlevels_extra : int
        Number of contour levels for the optional extra parameter.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    """

    if output is None:
        print("Plotting goodness-of-fit of model grid...", end="")
    else:
        print(f"Plotting goodness-of-fit of model grid: {output}...", end="")

    config_file = os.path.join(os.getcwd(), "species_config.ini")

    config = configparser.ConfigParser()
    config.read(config_file)

    db_path = config["species"]["database"]

    h5_file = h5py.File(db_path, "r")

    dset = h5_file[f"results/comparison/{tag}/goodness_of_fit"]

    n_param = dset.attrs["n_param"]
    parallax = dset.attrs["parallax"]

    flux_scaling = np.array(h5_file[f"results/comparison/{tag}/flux_scaling"])

    radius = (
        np.sqrt(flux_scaling * (constants.PARSEC / (1e-3 * parallax)) ** 2)
        / constants.R_JUP
    )

    # if 'extra_scaling' in h5_file[f'results/comparison/{tag}']:
    #     extra_scaling = np.array(h5_file[f'results/comparison/{tag}/extra_scaling'])
    # else:
    #     extra_scaling = None

    read_obj = read_object.ReadObject(dset.attrs["object_name"])

    n_wavel = 0
    for item in read_obj.get_spectrum().values():
        n_wavel += item[0].shape[0]

    goodness_fit = np.array(dset)

    model_param = []
    coord_points = []

    for i in range(n_param):
        model_param.append(dset.attrs[f"parameter{i}"])
        coord_points.append(
            np.array(h5_file[f"results/comparison/{tag}/coord_points{i}"])
        )

    # Set the coordinate for the x-axis to Teff
    coord_x = coord_points[0]
    coord_y = None
    param_y = None

    for i, item in enumerate(coord_points[1:]):
        if len(item) > 1:
            # Set the coordinate for the y-axis to the 1st axis after
            # Teff that has a length larger than 1
            coord_y = item
            param_y = model_param[i + 1]
            break

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.axisbelow"] = False

    fig = plt.figure(figsize=figsize)

    if coord_y is None:
        # Create a line plot if there is not a parameter for the y-axis
        gridsp = mpl.gridspec.GridSpec(1, 1)
        gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax = plt.subplot(gridsp[0, 0])

    else:
        # Create a contour plot if there is a second parameter to show
        gridsp = mpl.gridspec.GridSpec(1, 2, width_ratios=[4.0, 0.25])
        gridsp.update(wspace=0.07, hspace=0, left=0, right=1, bottom=0, top=1)

        ax = plt.subplot(gridsp[0, 0])
        ax_cb = plt.subplot(gridsp[0, 1])

    ax.tick_params(
        axis="both",
        which="major",
        colors="black",
        labelcolor="black",
        direction="in",
        width=1,
        length=5,
        labelsize=11,
        top=True,
        bottom=True,
        left=True,
        right=True,
        pad=5,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        colors="black",
        labelcolor="black",
        direction="in",
        width=1,
        length=3,
        labelsize=11,
        top=True,
        bottom=True,
        left=True,
        right=True,
        pad=5,
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel(r"$T_\mathregular{eff}$ (K)", fontsize=13.0)

    if param_y is None:
        ax.set_ylabel(r"$\Delta\log\,G_k$", fontsize=13.0)

    elif param_y == "ism_ext":
        ax.set_ylabel(r"$\mathregular{A}_\mathregular{V}$", fontsize=13.0)

    elif param_y == "logg":
        ax.set_ylabel(r"$\mathregular{log}\,g$", fontsize=13.0)

    elif param_y == "feh":
        ax.set_ylabel("[Fe/H]", fontsize=13.0)

    elif param_y == "c_o_ratio":
        ax.set_ylabel("C/O", fontsize=13.0)

    elif param_y == "fsed":
        ax.set_ylabel(r"$f_\mathregular{sed}$", fontsize=13.0)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.11)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

    if title is not None:
        ax.set_title(title, y=1.02, fontsize=14.0)

    # Sum/collapse over parameters with a single value
    for i, item in enumerate(coord_points):
        if len(item) == 1:
            goodness_fit = np.sum(goodness_fit, axis=i)

    # Indices of the best-fit model
    best_index = np.unravel_index(np.nanargmin(goodness_fit), goodness_fit.shape)

    if len(model_param) > 2 or extra_param == "radius":
        if extra_param is None:
            # Select all axes beyond the 1st and 2nd axis
            ax_list = []
            for i in range(len(coord_points) - 2):
                ax_list.append(2 + i)

            # Select minimum G_k along the axes of ax_list
            # This creates a 3D array of goodness_fit
            goodness_fit = np.nanmin(goodness_fit, axis=tuple(ax_list))

        else:
            if extra_param == "radius":
                goodness_full = goodness_fit.copy()

                # Select all axes beyond the 2nd axis
                ax_list = []
                for i in range(len(coord_points) - 2):
                    ax_list.append(2 + i)

                # Select minimum G_k along the axes of ax_list
                # This creates a 2D array from goodness_fit
                goodness_fit = np.nanmin(goodness_fit, axis=tuple(ax_list))

                extra_map = np.zeros(goodness_fit.shape[:2])
                for i in range(goodness_fit.shape[0]):
                    for j in range(goodness_fit.shape[1]):
                        # Get the indices in the goodness_full array
                        # for the values from the collapsed (2D)
                        # goodness_fit array
                        min_idx = np.argwhere(goodness_fit[i, j] == goodness_full)

                        if len(min_idx) > 1:
                            warnings.warn(
                                f"Found {len(min_idx)} positions in the "
                                "goodness-of-fit grid with the value "
                                f"{goodness_fit[i, j]:.2f}. Using the "
                                "first position in the list but this "
                                "warning is not expected to have occurred."
                            )

                        extra_map[i, j] = radius[tuple(min_idx[0])]

            else:
                extra_idx = model_param.index(extra_param)

                if extra_idx != 2:
                    goodness_fit = np.swapaxes(goodness_fit, extra_idx, 2)

                    coord_points_new = [
                        coord_points[0],
                        coord_points[1],
                        coord_points[extra_idx],
                    ]

                    for coord_idx in range(len(coord_points)):
                        if coord_idx in [0, 1, extra_idx]:
                            continue

                        # Add remaining coordinate points but skip
                        # 1st, 2nd, and extra axis that have already
                        # been added before the for loop
                        coord_points_new.append(coord_points[coord_idx])

                    coord_points = coord_points_new.copy()

                    # Set the extra axis to the 3rd axis after the swap
                    extra_idx = 2

                if len(model_param) > 3:
                    # Select all axes beyond the axis of extra_param
                    ax_list = []
                    for i in range(len(coord_points) - 3):
                        ax_list.append(3 + i)

                    # Select minimum G_k along the axes of ax_list
                    # This creates a 3D array of goodness_fit
                    goodness_fit = np.nanmin(goodness_fit, axis=tuple(ax_list))

                # Indices with the minimum G_k for the tested values
                # of the 3rd axis, that is, the extra_param axis
                # This creates a 2D array with the shape of the
                # 1st and 2nd axis of goodness_fit
                indices = np.argmin(goodness_fit, axis=extra_idx)

                # Select minimum G_k along the 3rd axis
                # This creates a 2D array of goodness_fit
                goodness_fit = np.nanmin(goodness_fit, axis=extra_idx)

                # Create 2D array with the extra map that will
                # be plotted as contours over the main map
                extra_map = np.zeros(goodness_fit.shape)
                for i in range(extra_map.shape[0]):
                    for j in range(extra_map.shape[1]):
                        extra_map[i, j] = coord_points[extra_idx][indices[i, j]]

    # Transpose for plot so make Teff the x axis

    goodness_fit = np.transpose(goodness_fit)

    if extra_param is not None:
        extra_map = np.transpose(extra_map)

    if coord_y is not None:
        if upsample:
            fit_interp = RegularGridInterpolator((coord_y, coord_x), goodness_fit)

            x_new = np.linspace(coord_x[0], coord_x[-1], 50)
            y_new = np.linspace(coord_y[0], coord_y[-1], 50)

            x_grid, y_grid = np.meshgrid(x_new, y_new)

            goodness_fit = fit_interp((y_grid, x_grid))

        else:
            x_grid, y_grid = np.meshgrid(coord_x, coord_y)

    goodness_fit = np.log10(goodness_fit)
    goodness_fit -= np.nanmin(goodness_fit)

    nan_points = np.sum(np.isnan(goodness_fit))
    if nan_points > 0:
        warnings.warn(
            f"Found {nan_points} NaN values in the "
            "goodness-of-fit grid. These points will be set "
            "to zero in the contour map."
        )

        goodness_fit = np.nan_to_num(goodness_fit)

    if coord_y is None:
        ax.plot(
            coord_x,
            goodness_fit[0,],
        )

    else:
        c = ax.contourf(x_grid, y_grid, goodness_fit, levels=nlevels_main)

        cb = mpl.colorbar.Colorbar(
            ax=ax_cb,
            mappable=c,
            orientation="vertical",
            ticklocation="right",
            format="%.1f",
        )

        cb.ax.minorticks_on()

        cb.ax.tick_params(
            which="major",
            width=0.8,
            length=5,
            labelsize=12,
            direction="in",
            color="black",
        )

        cb.ax.tick_params(
            which="minor",
            width=0.8,
            length=3,
            labelsize=12,
            direction="in",
            color="black",
        )

        cb.ax.set_ylabel(
            r"$\Delta\log\,G_k$",
            rotation=270,
            labelpad=22,
            fontsize=13.0,
        )

        if extra_param is not None:
            if upsample:
                extra_interp = RegularGridInterpolator((coord_y, coord_x), extra_map)
                extra_map = extra_interp((y_grid, x_grid))

                cs = ax.contour(
                    x_grid,
                    y_grid,
                    extra_map,
                    levels=nlevels_extra,
                    colors="white",
                    linewidths=0.7,
                )

            else:
                cs = ax.contour(
                    coord_x,
                    coord_y,
                    extra_map,
                    levels=nlevels_extra,
                    colors="white",
                    linewidths=0.7,
                )

            # manual = [(2350, 0.8), (2500, 0.8), (2600, 0.8), (2700, 0.8),
            #           (2800, 0.8), (2950, 0.8), (3100, 0.8), (3300, 0.8)]

            ax.clabel(cs, cs.levels, inline=True, fontsize=8, fmt="%1.1f")

        # if extra_scaling is not None and len(coord_points[2]) > 1:
        #     ratio = np.transpose(flux_scaling[:, 0, :])/np.transpose(extra_scaling[:, 0, :, 0])
        #
        #     cs = ax.contour(coord_x, coord_y, ratio, levels=nlevels_extra, colors='white',
        #                     linestyles='-', linewidths=0.7)
        #
        #     ax.clabel(cs, cs.levels, inline=True, fontsize=8, fmt='%1.1f')

        ax.plot(
            coord_x[best_index[0]],
            coord_y[best_index[1]],
            marker="X",
            ms=10.0,
            color="#eb4242",
            mfc="#eb4242",
            mec="black",
        )

        # best_param = (coord_x[best_index[0]], coord_y[best_index[1]])
        #
        # par_key, par_unit, par_label = plot_util.quantity_unit(model_param, object_type='planet')
        #
        # par_text = f'{par_label[0]} = {best_param[0]:.0f} {par_unit[0]}\n' \
        #            f'{par_label[1]} = {best_param[1]:.1f}'
        #
        # ax.annotate(par_text, (best_param[0]+50., best_param[1]), ha='left', va='center',
        #             color='white', fontsize=12.)

    if extra_param is not None:
        extra_label = plot_util.update_labels([extra_param])[0]
        ax.plot([], [], ls="-", lw=1.2, color="white", label=extra_label)
        ax.legend(loc="best", frameon=False, labelcolor="linecolor", fontsize=12.0)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    h5_file.close()

    print(" [DONE]")

    return fig


@typechecked
def plot_model_spectra(
    tag: str,
    n_spectra: Optional[int] = None,
    flux_offset: Optional[float] = None,
    label_pos: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = (4.0, 2.5),
    output: Optional[str] = None,
    leg_param: Optional[List[str]] = None,
) -> mpl.figure.Figure:
    """
    Function for plotting the results from comparing a spectrum
    with a grid of model spectra.

    Parameters
    ----------
    tag : str
        Database tag where the results from the model
        comparison with
        :func:`~species.analysis.compare_spectra.CompareSpectra.compare_model`
        are stored.
    n_spectra : int, None
        The number of spectra with the lowest goodness-of-fit
        statistic that will be plotted in comparison with the data.
        All spectra are selected if the argument is set to ``None``.
    label_pos : tuple(float, float), None
        Position for the name labels. Should be provided as (x, y)
        for the lowest spectrum. The ``flux_offset`` will be applied
        to the remaining spectra. The labels are only plotted if the
        argument of both ``label_pos`` and ``flux_offset`` are not
        ``None``.
    flux_offset : float, None
        Offset to be applied such that the spectra do not overlap. No
        offset is applied if the argument is set to ``None``.
    xlim : tuple(float, float)
        Limits of the spectral type axis.
    ylim : tuple(float, float)
        Limits of the goodness-of-fit axis.
    title : str
        Plot title.
    offset : tuple(float, float)
        Offset for the label of the x- and y-axis.
    figsize : tuple(float, float)
        Figure size.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.
    leg_param : list(str), None
        List with the parameters to include in the legend of the
        model spectra. Apart from atmospheric parameters (e.g.
        'teff', 'logg', 'radius') also parameters such as 'mass'
        and 'luminosity' can be included. The default atmospheric
        parameters are included in the legend if the argument is
        set to ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    """

    if output is None:
        print("Plotting model spectra comparison...", end="")
    else:
        print(f"Plotting model spectra comparison: {output}...", end="")

    if flux_offset is None:
        flux_offset = 0.0

    config_file = os.path.join(os.getcwd(), "species_config.ini")

    config = configparser.ConfigParser()
    config.read(config_file)

    db_path = config["species"]["database"]

    h5_file = h5py.File(db_path, "r")

    dset = h5_file[f"results/comparison/{tag}/goodness_of_fit"]

    object_name = dset.attrs["object_name"]
    n_spec_name = dset.attrs["n_spec_name"]
    model_name = dset.attrs["model"]
    n_param = dset.attrs["n_param"]
    n_scale_spec = dset.attrs["n_scale_spec"]
    parallax = dset.attrs["parallax"]

    spec_name = []
    for i in range(n_spec_name):
        spec_name.append(dset.attrs[f"spec_name{i}"])

    model_param = []
    coord_points = []
    for i in range(n_param):
        model_param.append(dset.attrs[f"parameter{i}"])
        coord_points.append(
            np.array(h5_file[f"results/comparison/{tag}/coord_points{i}"])
        )

    scale_spec = []
    for i in range(n_scale_spec):
        scale_spec.append(dset.attrs[f"scale_spec{i}"])

    goodness_fit = np.array(dset)
    goodness_fit = np.log10(goodness_fit)
    goodness_fit -= np.nanmin(goodness_fit)

    sort_idx = np.unravel_index(np.argsort(goodness_fit, axis=None), goodness_fit.shape)
    sort_idx = list(sort_idx)

    for i, item in enumerate(sort_idx):
        sort_idx[i] = item[:n_spectra]

    flux_scaling = np.array(h5_file[f"results/comparison/{tag}/flux_scaling"])

    if n_scale_spec > 0:
        extra_scaling = np.array(h5_file[f"results/comparison/{tag}/extra_scaling"])
    else:
        extra_scaling = None

    radius = (
        np.sqrt(flux_scaling * (constants.PARSEC / (1e-3 * parallax)) ** 2)
        / constants.R_JUP
    )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.axisbelow"] = False

    fig = plt.figure(figsize=figsize)
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    ax.tick_params(
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
    )

    ax.tick_params(
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
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel("Wavelength (\N{GREEK SMALL LETTER MU}m)", fontsize=13)

    if flux_offset == 0.0:
        ax.set_ylabel(
            r"$\mathregular{F}_\lambda$"
            + " (W m$^{-2}$ \N{GREEK SMALL LETTER MU}m$^{-1}$)",
            fontsize=11,
        )
    else:
        ax.set_ylabel(
            r"$\mathregular{F}_\lambda$ (W m$^{-2}$"
            + " \N{GREEK SMALL LETTER MU}m$^{-1}$) + offset",
            fontsize=11,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.1)
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

    if title is not None:
        ax.set_title(title, y=1.02, fontsize=13)

    read_obj = read_object.ReadObject(object_name)
    obj_spec = read_obj.get_spectrum()

    if flux_offset == 0.0:
        for spec_key, spec_item in obj_spec.items():
            if spec_key not in spec_name:
                continue

            ax.plot(spec_item[0][:, 0], spec_item[0][:, 1], "-", lw=0.5, color="black")

    model_reader = read_model.ReadModel(model_name)

    for i in range(n_spectra):
        param_select = {"parallax": parallax}
        idx_select = []

        for param_idx, param_item in enumerate(model_param):
            param_select[param_item] = coord_points[param_idx][sort_idx[param_idx][i]]
            idx_select.append(sort_idx[param_idx][i])

        param_select["radius"] = radius[tuple(idx_select)]

        if flux_offset != 0.0:
            for spec_key, spec_item in obj_spec.items():
                if spec_key not in spec_name:
                    continue

                if spec_key in scale_spec:
                    spec_idx = scale_spec.index(spec_key)
                    scaling_idx = np.append(idx_select, spec_idx)
                    data_scaling = extra_scaling[tuple(scaling_idx)]

                else:
                    data_scaling = 1.0

                ax.plot(
                    spec_item[0][:, 0],
                    (n_spectra - i - 1) * flux_offset
                    + data_scaling * spec_item[0][:, 1],
                    "-",
                    lw=0.5,
                    color="black",
                )

        for spec_key, spec_item in obj_spec.items():
            if spec_key not in spec_name:
                continue

            model_box = model_reader.get_data(
                model_param=param_select,
                spec_res=spec_item[3],
                wavel_resample=spec_item[0][:, 0],
            )

            ax.plot(
                model_box.wavelength,
                (n_spectra - i - 1) * flux_offset + model_box.flux,
                color="tomato",
                lw=0.5,
            )

        if label_pos is not None and flux_offset != 0.0:
            if leg_param is None:
                leg_param = model_param.copy()
                leg_param.append("radius")

            label_text = plot_util.create_model_label(
                model_param=param_select,
                model_name=model_name,
                inc_model_name=False,
                object_type="planet",
                leg_param=leg_param,
            )

            label_text = (
                rf"$\Delta\log\,G_k = "
                rf"{goodness_fit[tuple(idx_select)]:.2f}$: {label_text}"
            )

            ax.text(
                label_pos[0],
                label_pos[1] + (n_spectra - i - 1) * flux_offset,
                label_text,
                fontsize=8.0,
                ha="left",
            )

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    h5_file.close()

    print(" [DONE]")

    return fig
