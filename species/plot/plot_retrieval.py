"""
Module for plotting atmospheric retrieval results.
"""

# import copy
import sys
import warnings

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from scipy.stats import lognorm
from typeguard import typechecked

from species.data import database
from species.read import read_radtrans
from species.util import retrieval_util


@typechecked
def plot_pt_profile(
    tag: str,
    random: Optional[int] = 100,
    envelope: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
    extra_axis: Optional[str] = None,
    rad_conv_bound: bool = False,
) -> mpl.figure.Figure:
    """
    Function to plot the posterior distribution.

    Parameters
    ----------
    tag : str
        Database tag with the posterior samples.
    random : int, None
        Number of randomly selected samples from the posterior. All
        samples are selected if the argument is set to ``None``.
    envelope : bool
        Plot an envelope instead of the individual samples. The
        envelopes show the 68 and 99.7 percent confidence intervals,
        so :math:`1\\sigma` and :math:`3\\sigma` in case of
        Gaussian distributions.
    xlim : tuple(float, float), None
        Limits of the temperature axis. Default values are used if
        set to ``None``.
    ylim : tuple(float, float), None
        Limits of the pressure axis. Default values are used if set
        to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values are used
        if set to ``None``.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        Not used if set to ``None``.
    extra_axis : str, None
        The quantify that is plotted at the top axis ('photosphere',
        'grains'). The top axis is not used if the argument is set
        to ``None``.
    rad_conv_bound : bool
        Plot the range of pressures (:math:`\\pm 1\\sigma`) of the
        radiative-convective boundary.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    """

    if output is None:
        print("Plotting the P-T profiles...", end="", flush=True)
    else:
        print(f"Plotting the P-T profiles: {output}...", end="", flush=True)

    cloud_species = ["Fe(c)", "MgSiO3(c)", "Al2O3(c)", "Na2S(c)", "KCL(c)"]

    cloud_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:cyan",
        "tab:pink",
        "tab:brown",
        "tab:olive",
    ]

    color_iter = iter(cloud_colors)

    species_db = database.Database()
    box = species_db.get_samples(tag)

    parameters = np.asarray(box.parameters)
    samples = box.samples
    model_param = box.prob_sample

    if random is not None:
        indices = np.random.randint(samples.shape[0], size=random)
        samples = samples[indices,]

    param_index = {}
    for item in parameters:
        param_index[item] = np.argwhere(parameters == item)[0][0]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig = plt.figure(figsize=(4.0, 5.0))
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

    ax.set_xlabel("Temperature (K)", fontsize=13)
    ax.set_ylabel("Pressure (bar)", fontsize=13)

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.06)
        ax.get_yaxis().set_label_coords(-0.14, 0.5)

    if "temp_nodes" in box.attributes:
        temp_nodes = box.attributes["temp_nodes"]
    else:
        # For backward compatibility
        temp_nodes = 15

    if "abund_nodes" in box.attributes:
        if box.attributes["abund_nodes"] == "None":
            abund_nodes = None
        else:
            abund_nodes = box.attributes["abund_nodes"]
    else:
        # For backward compatibility
        abund_nodes = None

    if "max_press" in box.attributes:
        max_press = box.attributes["max_press"]
    else:
        # For backward compatibility
        max_press = 1e3  # (bar)

    if xlim is None:
        ax.set_xlim(1000.0, 5000.0)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is None:
        ax.set_ylim(max_press, 1e-6)
    else:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_yscale("log")

    # Create the pressure points (bar)
    pressure = np.logspace(-6.0, np.log10(max_press), 180)

    if "tint" in parameters and "log_delta" in parameters and "alpha" in parameters:
        pt_profile = "molliere"

    elif "tint" in parameters and "log_delta" in parameters:
        pt_profile = "eddington"

    else:
        pt_profile = "free"

        temp_index = []
        for i in range(temp_nodes):
            temp_index.append(np.argwhere(parameters == f"t{i}")[0][0])

        knot_press = np.logspace(
            np.log10(pressure[0]), np.log10(pressure[-1]), temp_nodes
        )

    if pt_profile == "molliere":
        conv_press = np.zeros(samples.shape[0])

    if envelope:
        temp_list = np.zeros((samples.shape[0], pressure.shape[0]))
    else:
        temp_list = None

    for i, item in enumerate(samples):
        # C/O and [Fe/H]

        if box.attributes["chemistry"] == "equilibrium":
            metallicity = item[param_index["metallicity"]]
            c_o_ratio = item[param_index["c_o_ratio"]]

        elif box.attributes["chemistry"] == "free":
            # TODO Set [Fe/H] = 0
            metallicity = 0.0

            # Create a dictionary with the mass fractions

            if abund_nodes is None:
                log_x_abund = {}
                line_species = []

                for line_idx in range(box.attributes["n_line_species"]):
                    line_item = box.attributes[f"line_species{line_idx}"]
                    log_x_abund[line_item] = item[param_index[line_item]]
                    line_species.append(line_item)

                # Check if the C/H and O/H ratios are within the prior boundaries

                _, _, c_o_ratio = retrieval_util.calc_metal_ratio(
                    log_x_abund, line_species
                )

            else:
                log_x_abund = {}
                for line_idx in range(box.attributes["n_line_species"]):
                    line_item = box.attributes[f"line_species{line_idx}"]
                    for node_idx in range(abund_nodes):
                        log_x_abund[f"{line_item}_{node_idx}"] = model_param[
                            f"{line_item}_{node_idx}"
                        ]

                # TODO Set C/O = 0.55 for Molliere P-T profile
                # and cloud condensation profiles
                c_o_ratio = 0.55

        if pt_profile == "molliere":
            t3_param = np.array(
                [
                    item[param_index["t1"]],
                    item[param_index["t2"]],
                    item[param_index["t3"]],
                ]
            )

            temp, _, conv_press[i] = retrieval_util.pt_ret_model(
                t3_param,
                10.0 ** item[param_index["log_delta"]],
                item[param_index["alpha"]],
                item[param_index["tint"]],
                pressure,
                metallicity,
                c_o_ratio,
            )

        elif pt_profile == "eddington":
            tau = pressure * 1e6 * 10.0 ** item[param_index["log_delta"]]
            temp = (0.75 * item[param_index["tint"]] ** 4.0 * (2.0 / 3.0 + tau)) ** 0.25

        elif pt_profile == "free":
            knot_temp = []
            for j in range(temp_nodes):
                knot_temp.append(item[temp_index[j]])

            knot_temp = np.asarray(knot_temp)

            if "pt_smooth" in parameters:
                pt_smooth = item[param_index["pt_smooth"]]

            elif "pt_smooth_0" in parameters:
                pt_smooth = {}
                for i in range(temp_nodes - 1):
                    pt_smooth[f"pt_smooth_{i}"] = item[param_index[f"pt_smooth_{i}"]]

            elif "pt_turn" in parameters:
                pt_smooth = {
                    "pt_smooth_1": item[param_index["pt_smooth_1"]],
                    "pt_smooth_2": item[param_index["pt_smooth_2"]],
                    "pt_turn": item[param_index["pt_turn"]],
                    "pt_index": item[param_index["pt_index"]],
                }

            else:
                pt_smooth = box.attributes["pt_smooth"]

            temp = retrieval_util.pt_spline_interp(
                knot_press, knot_temp, pressure, pt_smooth=pt_smooth
            )

        # if pt_profile == "free":
        #     temp = temp[:, 0]
        #
        #     if "poor_mans_nonequ_chem" in sys.modules:
        #         from poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances
        #     else:
        #         from petitRADTRANS.poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances
        #
        #     ab = interpol_abundances(
        #         np.full(temp.shape[0], c_o_ratio),
        #         np.full(temp.shape[0], metallicity),
        #         temp,
        #         pressure,
        #     )
        #
        #     nabla_ad = ab["nabla_ad"]
        #
        #     # Convert pressures from bar to cgs units
        #     press_cgs = pressure * 1e6
        #
        #     # Calculate the current, radiative temperature gradient
        #     nab_rad = np.diff(np.log(temp)) / np.diff(np.log(press_cgs))
        #
        #     # Extend to array of same length as pressure structure
        #     nabla_rad = np.ones_like(temp)
        #     nabla_rad[0] = nab_rad[0]
        #     nabla_rad[-1] = nab_rad[-1]
        #     nabla_rad[1:-1] = (nab_rad[1:] + nab_rad[:-1]) / 2.0
        #
        #     # Where is the atmosphere convectively unstable?
        #     conv_index = nabla_rad > nabla_ad
        #
        #     tfinal = None
        #
        #     for i in range(10):
        #         if i == 0:
        #             t_take = copy.copy(temp)
        #         else:
        #             t_take = copy.copy(tfinal)
        #
        #         ab = interpol_abundances(
        #             np.full(t_take.shape[0], c_o_ratio),
        #             np.full(t_take.shape[0], metallicity),
        #             t_take,
        #             pressure,
        #         )
        #
        #         nabla_ad = ab["nabla_ad"]
        #
        #         # Calculate the average nabla_ad between the layers
        #         nabla_ad_mean = nabla_ad
        #         nabla_ad_mean[1:] = (nabla_ad[1:] + nabla_ad[:-1]) / 2.0
        #
        #         # What are the increments in temperature due to convection
        #         tnew = nabla_ad_mean[conv_index] * np.mean(np.diff(np.log(press_cgs)))
        #
        #         # What is the last radiative temperature?
        #         tstart = np.log(t_take[~conv_index][-1])
        #
        #         # Integrate and translate to temperature
        #         # from log(temperature)
        #         tnew = np.exp(np.cumsum(tnew) + tstart)
        #
        #         # Add upper radiative and lower covective
        #         # part into one single array
        #         tfinal = copy.copy(t_take)
        #         tfinal[conv_index] = tnew
        #
        #         if np.max(np.abs(t_take - tfinal) / t_take) < 0.01:
        #             break
        #
        #     temp = copy.copy(tfinal)

        if envelope:
            temp_list[i] = temp
        else:
            ax.plot(temp, pressure, "-", lw=0.3, color="gray", zorder=1)

    if box.attributes["chemistry"] == "free":
        # TODO Set [Fe/H] = 0
        model_param["metallicity"] = metallicity
        model_param["c_o_ratio"] = c_o_ratio

    if pt_profile == "molliere":
        temp, _, conv_press_median = retrieval_util.pt_ret_model(
            np.array([model_param["t1"], model_param["t2"], model_param["t3"]]),
            10.0 ** model_param["log_delta"],
            model_param["alpha"],
            model_param["tint"],
            pressure,
            model_param["metallicity"],
            model_param["c_o_ratio"],
        )

        if rad_conv_bound:
            press_min = np.mean(conv_press) - np.std(conv_press)
            press_max = np.mean(conv_press) + np.std(conv_press)

            ax.axhspan(
                press_min,
                press_max,
                zorder=0,
                color="lightsteelblue",
                linewidth=0.0,
                alpha=0.5,
            )

            ax.axhline(conv_press_median, zorder=0, color="cornflowerblue", alpha=0.5)

    elif pt_profile == "eddington":
        tau = pressure * 1e6 * 10.0 ** model_param["log_delta"]
        temp = (0.75 * model_param["tint"] ** 4.0 * (2.0 / 3.0 + tau)) ** 0.25

    elif pt_profile == "free":
        knot_temp = []
        for i in range(temp_nodes):
            knot_temp.append(model_param[f"t{i}"])

        knot_temp = np.asarray(knot_temp)

        ax.plot(knot_temp, knot_press, "o", ms=5.0, mew=0.0, color="tomato", zorder=3.0)

        if "pt_smooth" in parameters:
            pt_smooth = model_param["pt_smooth"]

        elif "pt_smooth_0" in parameters:
            pt_smooth = {}
            for i in range(temp_nodes - 1):
                pt_smooth[f"pt_smooth_{i}"] = item[param_index[f"pt_smooth_{i}"]]

        elif "pt_turn" in parameters:
            pt_smooth = {
                "pt_smooth_1": model_param["pt_smooth_1"],
                "pt_smooth_2": model_param["pt_smooth_2"],
                "pt_turn": model_param["pt_turn"],
                "pt_index": model_param["pt_index"],
            }

        else:
            pt_smooth = box.attributes["pt_smooth"]

        temp = retrieval_util.pt_spline_interp(
            knot_press, knot_temp, pressure, pt_smooth=pt_smooth
        )

    if envelope:
        temp_percent = np.percentile(temp_list, [0.3, 16.0, 84.0, 99.7], axis=0)

        ax.fill_betweenx(
            y=pressure,
            x1=temp_percent[0],
            x2=temp_percent[3],
            color="peachpuff",
            alpha=0.4,
            zorder=1,
            linewidth=0.0,
        )

        ax.fill_betweenx(
            y=pressure,
            x1=temp_percent[1],
            x2=temp_percent[2],
            color="peachpuff",
            alpha=1.0,
            zorder=1,
            linewidth=0.0,
        )

    ax.plot(temp, pressure, "-", lw=1, color="black", zorder=2)

    # data = np.loadtxt('res_struct.dat')
    # ax.plot(data[:, 1], data[:, 0], lw=1, color='tab:purple')

    # Add cloud condensation profiles

    if (
        extra_axis == "grains"
        and "metallicity" in model_param
        and "c_o_ratio" in model_param
    ):
        if box.attributes["quenching"] == "pressure":
            p_quench = 10.0 ** model_param["log_p_quench"]

        elif box.attributes["quenching"] == "diffusion":
            p_quench = retrieval_util.quench_pressure(
                radtrans.rt_object.press,
                radtrans.rt_object.temp,
                model_param["metallicity"],
                model_param["c_o_ratio"],
                model_param["logg"],
                model_param["log_kzz"],
            )

        else:
            p_quench = None

        # Import interpol_abundances here because it is slow

        if "poor_mans_nonequ_chem" in sys.modules:
            from poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances
        else:
            from petitRADTRANS.poor_mans_nonequ_chem.poor_mans_nonequ_chem import (
                interpol_abundances,
            )

        abund_in = interpol_abundances(
            np.full(pressure.shape[0], model_param["c_o_ratio"]),
            np.full(pressure.shape[0], model_param["metallicity"]),
            temp,
            pressure,
            Pquench_carbon=p_quench,
        )

        for item in cloud_species:
            if f"{item[:-3].lower()}_tau" in model_param:
                # Calculate the scaled mass fraction of the clouds
                model_param[
                    f"{item[:-3].lower()}_fraction"
                ] = retrieval_util.scale_cloud_abund(
                    model_param,
                    radtrans.rt_object,
                    pressure,
                    temp,
                    abund_in["MMW"],
                    "equilibrium",
                    abund_in,
                    item,
                    model_param[f"{item[:-3].lower()}_tau"],
                    pressure_grid=radtrans.pressure_grid,
                )

        for cloud_item in cloud_species:
            if cloud_item in radtrans.cloud_species:
                cond_temp = retrieval_util.get_condensation_curve(
                    composition=cloud_item[:-3],
                    press=pressure,
                    metallicity=model_param["metallicity"],
                    c_o_ratio=model_param["c_o_ratio"],
                    mmw=np.mean(abund_in["MMW"]),
                )

                ax.plot(
                    cond_temp,
                    pressure,
                    "--",
                    lw=0.8,
                    color=next(color_iter, "black"),
                    zorder=2,
                )

    if box.attributes["chemistry"] == "free":
        # Remove these parameters otherwise ReadRadtrans.get_model()
        # will assume equilibrium chemistry
        del model_param["metallicity"]
        del model_param["c_o_ratio"]

    if radtrans is not None:
        # Recalculate the best-fit model to update the attributes of radtrans.rt_object
        model_box = radtrans.get_model(model_param)

        contr_1d = np.mean(model_box.contribution, axis=1)
        contr_1d = ax.get_xlim()[0] + 0.5 * (contr_1d / np.amax(contr_1d)) * (
            ax.get_xlim()[1] - ax.get_xlim()[0]
        )

        ax.plot(
            contr_1d, 1e-6 * radtrans.rt_object.press, ls="--", lw=0.5, color="black"
        )

        if extra_axis == "photosphere":
            # Calculate the total optical depth
            # (line and continuum opacities)
            # radtrans.rt_object.calc_opt_depth(10.**model_param['logg'])

            wavelength = radtrans.rt_object.lambda_angstroem * 1e-4  # (um)

            # From Paul: The first axis of total_tau is the coordinate
            # of the cumulative opacity distribution function (ranging
            # from 0 to 1). A correct average is obtained by
            # multiplying the first axis with self.w_gauss, then
            # summing them. This is then the actual wavelength-mean.

            if radtrans.scattering:
                w_gauss = radtrans.rt_object.w_gauss[..., np.newaxis, np.newaxis]

                # From petitRADTRANS: Only use 0 index for species
                # because for lbl or test_ck_shuffle_comp = True
                # everything has been moved into the 0th index
                optical_depth = np.sum(
                    w_gauss * radtrans.rt_object.total_tau[:, :, 0, :], axis=0
                )

            else:
                # TODO Ask Paul if correct
                w_gauss = radtrans.rt_object.w_gauss[
                    ..., np.newaxis, np.newaxis, np.newaxis
                ]
                optical_depth = np.sum(
                    w_gauss * radtrans.rt_object.total_tau[:, :, :, :], axis=0
                )

                # Sum over all species
                optical_depth = np.sum(optical_depth, axis=1)

            ax2 = ax.twiny()

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
                top=True,
                bottom=False,
                left=True,
                right=True,
            )

            if ylim is None:
                ax2.set_ylim(max_press, 1e-6)
            else:
                ax2.set_ylim(ylim[0], ylim[1])

            ax2.set_yscale("log")

            ax2.set_xlabel(
                "Wavelength (\N{GREEK SMALL LETTER MU}m)", fontsize=13, va="bottom"
            )

            if offset is not None:
                ax2.get_xaxis().set_label_coords(0.5, 1.0 + abs(offset[0]))
            else:
                ax2.get_xaxis().set_label_coords(0.5, 1.06)

            photo_press = np.zeros(wavelength.shape[0])

            for i in range(photo_press.shape[0]):
                # Interpolate the optical depth to
                # the photosphere at tau = 2/3
                press_interp = interp1d(optical_depth[i, :], radtrans.rt_object.press)
                photo_press[i] = press_interp(2.0 / 3.0) * 1e-6  # cgs to (bar)

            ax2.plot(
                wavelength,
                photo_press,
                lw=0.5,
                color="tab:blue",
                label=r"Photosphere ($\tau$ = 2/3)",
            )

        elif extra_axis == "grains":
            if len(radtrans.cloud_species) > 0:
                ax2 = ax.twiny()

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
                    top=True,
                    bottom=False,
                    left=True,
                    right=True,
                )

                if ylim is None:
                    ax2.set_ylim(max_press, 1e-6)
                else:
                    ax2.set_ylim(ylim[0], ylim[1])

                ax2.set_xscale("log")
                ax2.set_yscale("log")

                ax2.set_xlabel("Average particle radius (µm)", fontsize=13, va="bottom")

                # Recalculate the best-fit model to update the r_g attribute of radtrans.rt_object
                radtrans.get_model(model_param)

                if offset is not None:
                    ax2.get_xaxis().set_label_coords(0.5, 1.0 + abs(offset[0]))
                else:
                    ax2.get_xaxis().set_label_coords(0.5, 1.06)

            else:
                raise ValueError(
                    "The Radtrans object does not contain any cloud "
                    "species. Please set the argument of 'extra_axis' "
                    "either to 'photosphere' or None."
                )

            color_iter = iter(cloud_colors)

            for cloud_item in cloud_species:
                if cloud_item in radtrans.cloud_species:
                    cloud_index = radtrans.rt_object.cloud_species.index(cloud_item)

                    label = ""
                    for char in cloud_item[:-3]:
                        if char.isnumeric():
                            label += f"$_{char}$"
                        else:
                            label += char

                    if label == "KCL":
                        label = "KCl"

                    ax2.plot(
                        # (cm) -> (um)
                        radtrans.rt_object.r_g[:, cloud_index] * 1e4,
                        # (Ba) -> (Bar)
                        radtrans.rt_object.press * 1e-6,
                        lw=0.8,
                        color=next(color_iter),
                        label=label,
                    )

        if extra_axis is not None:
            ax2.legend(loc="upper right", frameon=False, fontsize=12.0)

    else:
        if extra_axis is not None:
            warnings.warn(
                "The argument of extra_axis is ignored because radtrans does not "
                "contain a ReadRadtrans object."
            )

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig


@typechecked
def plot_opacities(
    tag: str,
    radtrans: read_radtrans.ReadRadtrans,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
) -> mpl.figure.Figure:
    """
    Function to plot the line and continuum opacity structure of the
    atmosphere by using the median parameters from posterior samples.

    Parameters
    ----------
    tag : str
        Database tag with the posterior samples.
    radtrans : read_radtrans.ReadRadtrans
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        The parameter is not used if the argument is set to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values are used
        if the argument is set to ``None``.
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
        print("Plotting opacities...", end="", flush=True)
    else:
        print(f"Plotting opacities: {output}...", end="", flush=True)

    species_db = database.Database()
    box = species_db.get_samples(tag)
    model_param = box.prob_sample

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig = plt.figure(figsize=(10.0, 6.0))
    gridsp = mpl.gridspec.GridSpec(2, 5, width_ratios=[4, 0.25, 1.5, 4, 0.25])
    gridsp.update(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[0, 0])
    ax2 = plt.subplot(gridsp[1, 0])
    ax3 = plt.subplot(gridsp[0, 1])
    ax4 = plt.subplot(gridsp[1, 1])

    ax5 = plt.subplot(gridsp[0, 3])
    ax6 = plt.subplot(gridsp[1, 3])
    ax7 = plt.subplot(gridsp[0, 4])
    ax8 = plt.subplot(gridsp[1, 4])

    radtrans.get_model(model_param)

    # Line opacities

    wavelength, opacity = radtrans.rt_object.get_opa(radtrans.rt_object.temp)

    wavelength *= 1e4  # (um)

    opacity_line = np.zeros(
        (radtrans.rt_object.freq.shape[0], radtrans.rt_object.press.shape[0])
    )

    for item in opacity.values():
        opacity_line += item

    # Continuum opacities

    opacity_cont_abs = radtrans.rt_object.continuum_opa
    opacity_cont_scat = radtrans.rt_object.continuum_opa_scat
    # opacity_cont_scat = radtrans.rt_object.continuum_opa_scat_emis
    opacity_total = opacity_line + opacity_cont_abs + opacity_cont_scat

    albedo = opacity_cont_scat / opacity_total

    # if radtrans.scattering:
    #     opacity_cont = radtrans.rt_object.continuum_opa_scat_emis
    # else:
    #     opacity_cont = radtrans.rt_object.continuum_opa_scat

    ax1.tick_params(
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
        top=True,
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
        top=True,
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
        top=True,
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
        bottom=True,
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
        bottom=True,
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
        bottom=True,
        left=True,
        right=True,
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
        bottom=True,
        left=True,
        right=True,
    )

    ax5.tick_params(
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
        labelbottom=False,
    )

    ax5.tick_params(
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
        labelbottom=False,
    )

    ax6.tick_params(
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

    ax6.tick_params(
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

    ax7.tick_params(
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

    ax7.tick_params(
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

    ax8.tick_params(
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

    ax8.tick_params(
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

    ax1.xaxis.set_major_locator(MultipleLocator(1.0))
    ax2.xaxis.set_major_locator(MultipleLocator(1.0))

    ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.2))

    ax5.xaxis.set_major_locator(MultipleLocator(1.0))
    ax6.xaxis.set_major_locator(MultipleLocator(1.0))

    ax5.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax6.xaxis.set_minor_locator(MultipleLocator(0.2))

    # ax1.yaxis.set_major_locator(LogLocator(base=10.))
    # ax2.yaxis.set_major_locator(LogLocator(base=10.))
    # ax3.yaxis.set_major_locator(LogLocator(base=10.))
    # ax4.yaxis.set_major_locator(LogLocator(base=10.))

    # ax1.yaxis.set_minor_locator(LogLocator(base=1.))
    # ax2.yaxis.set_minor_locator(LogLocator(base=1.))
    # ax3.yaxis.set_minor_locator(LogLocator(base=1.))
    # ax4.yaxis.set_minor_locator(LogLocator(base=1.))

    xx_grid, yy_grid = np.meshgrid(wavelength, 1e-6 * radtrans.rt_object.press)

    fig_1 = ax1.pcolormesh(
        xx_grid,
        yy_grid,
        np.transpose(opacity_line),
        cmap="viridis",
        shading="gouraud",
        norm=LogNorm(vmin=1e-6 * np.amax(opacity_line), vmax=np.amax(opacity_line)),
    )

    cb = Colorbar(ax=ax3, mappable=fig_1, orientation="vertical", ticklocation="right")
    cb.ax.set_ylabel("Line opacity (cm$^2$/g)", rotation=270, labelpad=20, fontsize=11)

    fig_2 = ax2.pcolormesh(
        xx_grid,
        yy_grid,
        np.transpose(albedo),
        cmap="viridis",
        shading="gouraud",
        norm=LogNorm(vmin=1e-4 * np.amax(albedo), vmax=np.amax(albedo)),
    )

    cb = Colorbar(ax=ax4, mappable=fig_2, orientation="vertical", ticklocation="right")
    cb.ax.set_ylabel("Single scattering albedo", rotation=270, labelpad=20, fontsize=11)

    fig_3 = ax5.pcolormesh(
        xx_grid,
        yy_grid,
        np.transpose(opacity_cont_abs),
        cmap="viridis",
        shading="gouraud",
        norm=LogNorm(
            vmin=1e-6 * np.amax(opacity_cont_abs), vmax=np.amax(opacity_cont_abs)
        ),
    )

    cb = Colorbar(ax=ax7, mappable=fig_3, orientation="vertical", ticklocation="right")
    cb.ax.set_ylabel(
        "Continuum absorption (cm$^2$/g)", rotation=270, labelpad=20, fontsize=11
    )

    fig_4 = ax6.pcolormesh(
        xx_grid,
        yy_grid,
        np.transpose(opacity_cont_scat),
        cmap="viridis",
        shading="gouraud",
        norm=LogNorm(
            vmin=1e-6 * np.amax(opacity_cont_scat), vmax=np.amax(opacity_cont_scat)
        ),
    )

    cb = Colorbar(ax=ax8, mappable=fig_4, orientation="vertical", ticklocation="right")
    cb.ax.set_ylabel(
        "Continuum scattering (cm$^2$/g)", rotation=270, labelpad=20, fontsize=11
    )

    ax1.set_ylabel("Pressure (bar)", fontsize=13)

    ax2.set_xlabel("Wavelength (\N{GREEK SMALL LETTER MU}m)", fontsize=13)
    ax2.set_ylabel("Pressure (bar)", fontsize=13)

    ax5.set_ylabel("Pressure (bar)", fontsize=13)

    ax6.set_xlabel("Wavelength (\N{GREEK SMALL LETTER MU}m)", fontsize=13)
    ax6.set_ylabel("Pressure (bar)", fontsize=13)

    ax1.set_xlim(wavelength[0], wavelength[-1])
    ax2.set_xlim(wavelength[0], wavelength[-1])

    ax5.set_xlim(wavelength[0], wavelength[-1])
    ax6.set_xlim(wavelength[0], wavelength[-1])

    ax1.set_ylim(
        radtrans.rt_object.press[-1] * 1e-6, radtrans.rt_object.press[0] * 1e-6
    )
    ax2.set_ylim(
        radtrans.rt_object.press[-1] * 1e-6, radtrans.rt_object.press[0] * 1e-6
    )

    ax5.set_ylim(
        radtrans.rt_object.press[-1] * 1e-6, radtrans.rt_object.press[0] * 1e-6
    )
    ax6.set_ylim(
        radtrans.rt_object.press[-1] * 1e-6, radtrans.rt_object.press[0] * 1e-6
    )

    if offset is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

        ax2.get_xaxis().set_label_coords(0.5, offset[0])
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)

        ax5.get_xaxis().set_label_coords(0.5, offset[0])
        ax5.get_yaxis().set_label_coords(offset[1], 0.5)

        ax6.get_xaxis().set_label_coords(0.5, offset[0])
        ax6.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.1)
        ax1.get_yaxis().set_label_coords(-0.14, 0.5)

        ax2.get_xaxis().set_label_coords(0.5, -0.1)
        ax2.get_yaxis().set_label_coords(-0.14, 0.5)

        ax5.get_xaxis().set_label_coords(0.5, -0.1)
        ax5.get_yaxis().set_label_coords(-0.14, 0.5)

        ax6.get_xaxis().set_label_coords(0.5, -0.1)
        ax6.get_yaxis().set_label_coords(-0.14, 0.5)

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")

    ax5.set_yscale("log")
    ax6.set_yscale("log")
    ax7.set_yscale("log")
    ax8.set_yscale("log")

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig


@typechecked
def plot_clouds(
    tag: str,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
    composition: str = "MgSiO3",
) -> mpl.figure.Figure:
    """
    Function to plot the size distributions for a given cloud
    composition as function as pressure. The size distributions are
    calculated for the median sample by using the ``radius_g`` (as
    function of pressure) and ``sigma_g``.

    Parameters
    ----------
    tag : str
        Database tag with the posterior samples.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values are
        used if the argument set to ``None``.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        The parameter is not used if the argument is set to ``None``.
    composition : str
        Cloud composition (e.g. 'MgSiO3', 'Fe', 'Al2O3', 'Na2S', 'KCl').

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for
        further customization of the plot.
    """

    species_db = database.Database()
    box = species_db.get_samples(tag)
    model_param = box.prob_sample

    if (
        f"{composition.lower()}_fraction" not in model_param
        and "log_tau_cloud" not in model_param
        and f"{composition}(c)" not in model_param
    ):
        raise ValueError(
            f"The mass fraction of the {composition} clouds is "
            "not found. The median sample contains the following "
            f"parameters: {list(model_param.keys())}"
        )

    if output is None:
        print(f"Plotting {composition} clouds...", end="", flush=True)
    else:
        print(f"Plotting {composition} clouds: {output}...", end="", flush=True)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig = plt.figure(figsize=(4.0, 3.0))
    gridsp = mpl.gridspec.GridSpec(1, 2, width_ratios=[4, 0.25])
    gridsp.update(wspace=0.1, hspace=0.0, left=0, right=1, bottom=0, top=1)

    ax1 = plt.subplot(gridsp[0, 0])
    ax2 = plt.subplot(gridsp[0, 1])

    radtrans.get_model(model_param)

    cloud_index = radtrans.rt_object.cloud_species.index(f"{composition}(c)")
    radius_g = radtrans.rt_object.r_g[:, cloud_index] * 1e4  # (cm) -> (um)
    sigma_g = model_param["sigma_lnorm"]

    r_bins = np.logspace(-3.0, 3.0, 1000)
    radii = (r_bins[1:] + r_bins[:-1]) / 2.0

    dn_dr = np.zeros((radius_g.shape[0], radii.shape[0]))

    for i, item in enumerate(radius_g):
        dn_dr[i,] = lognorm.pdf(radii, s=np.log(sigma_g), loc=0.0, scale=item)

    ax1.tick_params(
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
        labelbottom=True,
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
        top=True,
        bottom=True,
        left=True,
        right=True,
        labelbottom=True,
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
        top=True,
        bottom=True,
        left=True,
        right=True,
    )

    xx_grid, yy_grid = np.meshgrid(radii, 1e-6 * radtrans.rt_object.press)

    mesh_fig = ax1.pcolormesh(
        xx_grid,
        yy_grid,
        dn_dr,
        cmap="viridis",
        shading="auto",
        norm=LogNorm(vmin=1e-10 * np.amax(dn_dr), vmax=np.amax(dn_dr)),
    )

    cb = Colorbar(
        ax=ax2, mappable=mesh_fig, orientation="vertical", ticklocation="right"
    )
    cb.ax.set_ylabel("dn/dr", rotation=270, labelpad=20, fontsize=11)

    for item in radtrans.rt_object.press * 1e-6:  # (bar)
        ax1.axhline(item, ls="-", lw=0.1, color="white")

    for item in radtrans.rt_object.cloud_radii * 1e4:  # (um)
        ax1.axvline(item, ls="-", lw=0.1, color="white")

    ax1.text(
        0.07,
        0.07,
        rf"$\sigma_\mathrm{{g}}$ = {sigma_g:.2f}",
        ha="left",
        va="bottom",
        transform=ax1.transAxes,
        color="black",
        fontsize=13.0,
    )

    ax1.set_ylabel("Pressure (bar)", fontsize=13)
    ax1.set_xlabel("Grain radius (µm)", fontsize=13)

    ax1.set_xlim(radii[0], radii[-1])
    ax1.set_ylim(
        radtrans.rt_object.press[-1] * 1e-6, radtrans.rt_object.press[0] * 1e-6
    )

    if offset is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax1.get_xaxis().set_label_coords(0.5, -0.1)
        ax1.get_yaxis().set_label_coords(-0.15, 0.5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig


@typechecked
def plot_abundances(
    tag: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
    legend: Optional[dict] = None,
    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
) -> mpl.figure.Figure:
    """
    Function to plotting the retrieved abundance profiles.

    Parameters
    ----------
    tag : str
        Database tag with the posterior samples.
    xlim : tuple(float, float), None
        Limits of the temperature axis. Default values are used if
        set to ``None``.
    ylim : tuple(float, float), None
        Limits of the pressure axis. Default values are used if set
        to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values are
        used if the argument set to ``None``.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.
    legend : dict, None
        Dictionary with legend properties. Default values will
        be used if the argument is set to ``None``.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        The parameter is not used if the argument is set to ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for
        further customization of the plot.
    """

    species_db = database.Database()
    box = species_db.get_samples(tag)
    model_param = box.prob_sample

    if output is None:
        print("Plotting abundances...", end="", flush=True)
    else:
        print(f"Plotting abundances: {output}...", end="", flush=True)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig = plt.figure(figsize=(4.0, 3.0))
    gridsp = mpl.gridspec.GridSpec(1, 1)
    gridsp.update(wspace=0.1, hspace=0.0, left=0, right=1, bottom=0, top=1)

    ax = plt.subplot(gridsp[0, 0])

    radtrans.get_model(model_param)

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
        labelbottom=True,
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
        labelbottom=True,
    )

    sub_num = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    for line_idx, line_item in enumerate(radtrans.rt_object.line_species):
        line_label = line_item.split("_")[0]
        line_label = line_label.translate(sub_num)

        ax.plot(
            radtrans.rt_object.line_abundances[:, line_idx],
            radtrans.rt_object.press * 1e-6,
            lw=0.7,
            label=line_label,
        )

    ax.set_xlabel("Abundance", fontsize=13)
    ax.set_ylabel("Pressure (bar)", fontsize=13)

    if xlim is not None:
        ax.set_xlim(1e-10, 1.0)

    if ylim is None:
        ax.set_ylim(
            1e-6 * radtrans.rt_object.press[-1], 1e-6 * radtrans.rt_object.press[0]
        )
    else:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_xscale("log")
    ax.set_yscale("log")

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.1)
        ax.get_yaxis().set_label_coords(-0.15, 0.5)

    if legend is None:
        ax.legend(loc="upper left", fontsize=7.0)
    else:
        ax.legend(**legend)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig
