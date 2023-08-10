"""
Module for plotting MCMC results.
"""

import warnings

from typing import List, Optional, Tuple, Union

import h5py
import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import RegularGridInterpolator

from species.core import constants
from species.data import database
from species.util import plot_util, dust_util, read_util, retrieval_util


@typechecked
def plot_walkers(
    tag: str,
    nsteps: Optional[int] = None,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
) -> mpl.figure.Figure:
    """
    Function to plot the step history of the walkers.

    Parameters
    ----------
    tag : str
        Database tag with the samples.
    nsteps : int, None
        Number of steps that are plotted. All steps are
        plotted if the argument is set to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values
        are used if the arguments is set to ``None``.
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
        print("Plotting walkers...", end="", flush=True)
    else:
        print(f"Plotting walkers: {output}...", end="", flush=True)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    species_db = database.Database()
    box = species_db.get_samples(tag)

    samples = box.samples
    labels = plot_util.update_labels(box.parameters)

    if samples.ndim == 2:
        raise ValueError(
            f"The samples of '{tag}' have only 2 dimensions "
            f"whereas 3 are required for plotting the walkers. "
            f"The plot_walkers function can only be used after "
            "running the MCMC with run_mcmc and not after "
            f"running run_ultranest or run_multinest."
        )

    ndim = samples.shape[-1]

    fig = plt.figure(figsize=(6, ndim * 1.5))
    gridsp = mpl.gridspec.GridSpec(ndim, 1)
    gridsp.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

    for i in range(ndim):
        ax = plt.subplot(gridsp[i, 0])

        if i == ndim - 1:
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

        else:
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
                labelbottom=False,
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
                labelbottom=False,
            )

        if i == ndim - 1:
            ax.set_xlabel("Step number", fontsize=10)
        else:
            ax.set_xlabel("", fontsize=10)

        ax.set_ylabel(labels[i], fontsize=10)

        if offset is not None:
            ax.get_xaxis().set_label_coords(0.5, offset[0])
            ax.get_yaxis().set_label_coords(offset[1], 0.5)

        else:
            ax.get_xaxis().set_label_coords(0.5, -0.22)
            ax.get_yaxis().set_label_coords(-0.09, 0.5)

        if nsteps is not None:
            ax.set_xlim(0, nsteps)

        for j in range(samples.shape[0]):
            ax.plot(samples[j, :, i], ls="-", lw=0.5, color="black", alpha=0.5)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig


@typechecked
def plot_posterior(
    tag: str,
    burnin: Optional[int] = None,
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    title_fmt: Union[str, List[str]] = ".2f",
    limits: Optional[List[Tuple[float, float]]] = None,
    max_prob: bool = False,
    vmr: bool = False,
    inc_luminosity: bool = False,
    inc_mass: bool = False,
    inc_log_mass: bool = False,
    inc_pt_param: bool = False,
    inc_loglike: bool = False,
    inc_abund: bool = True,
    output: Optional[str] = None,
    object_type: str = "planet",
    param_inc: Optional[List[str]] = None,
) -> mpl.figure.Figure:
    """
    Function to plot the posterior distribution
    of the fitted parameters.

    Parameters
    ----------
    tag : str
        Database tag with the samples.
    burnin : int, None
        Number of burnin steps to exclude. All samples
        are used if the argument is set to ``None``.
    title : str, None
        Plot title. No title is shown if the arguments
        is set to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values
        are used if the arguments is set to ``None``.
    title_fmt : str, list(str)
        Format of the titles above the 1D distributions. Either a
        single string, which will be used for all parameters, or a
        list with the title format for each parameter separately
        (in the order as shown in the corner plot).
    limits : list(tuple(float, float), ), None
        Axis limits of all parameters. Automatically set if the
        argument is set to ``None``.
    max_prob : bool
        Plot the position of the sample with the
        maximum posterior probability.
    vmr : bool
        Plot the volume mixing ratios (i.e. number fractions)
        instead of the mass fractions of the retrieved species with
        :class:`~species.analysis.retrieval.AtmosphericRetrieval`.
    inc_luminosity : bool
        Include the log10 of the luminosity in the posterior plot
        as calculated from the effective temperature and radius.
    inc_mass : bool
        Include the mass in the posterior plot as calculated
        from the surface gravity and radius.
    inc_log_mass : bool
        Include the logarithm of the mass, :math:`\\log10{M}`, in
        the posterior plot, as calculated from the surface gravity
        and radius.
    inc_pt_param : bool
        Include the parameters of the pressure-temperature profile.
        Only used if the ``tag`` contains samples obtained with
        :class:`~species.analysis.retrieval.AtmosphericRetrieval`.
    inc_loglike : bool
        Include the log10 of the likelihood as additional
        parameter in the corner plot.
    inc_abund : bool
        Include the abundances when retrieving free abundances with
        :class:`~species.analysis.retrieval.AtmosphericRetrieval`.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.
    object_type : str
        Object type ('planet' or 'star'). With 'planet', the radius
        and mass are expressed in Jupiter units. With 'star', the
        radius and mass are expressed in solar units.
    param_inc : list(str), None
        List with subset of parameters that will be included in the
        posterior plot. This parameter can also be used to change the
        order of the parameters in the posterior plot. All parameters
        will be included if the argument is set to ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    """

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    if burnin is None:
        burnin = 0

    species_db = database.Database()

    box = species_db.get_samples(tag, burnin=burnin)
    samples = box.samples

    # index_sel = [0, 1, 8, 9, 14]
    # samples = samples[:, index_sel]
    #
    # for i in range(13, 9, -1):
    #     del box.parameters[i]
    #
    # del box.parameters[2]
    # del box.parameters[2]
    # del box.parameters[2]
    # del box.parameters[2]
    # del box.parameters[2]
    # del box.parameters[2]

    ndim = len(box.parameters)

    if not inc_pt_param and box.spectrum == "petitradtrans":
        pt_param = ["tint", "t1", "t2", "t3", "alpha", "log_delta"]

        index_del = []
        item_del = []

        for i in range(100):
            pt_item = f"t{i}"

            if pt_item in box.parameters:
                param_index = np.argwhere(np.array(box.parameters) == pt_item)[0]
                index_del.append(param_index)
                item_del.append(pt_item)

            else:
                break

        for item in pt_param:
            if item in box.parameters and item not in item_del:
                param_index = np.argwhere(np.array(box.parameters) == item)[0]
                index_del.append(param_index)
                item_del.append(item)

        samples = np.delete(samples, index_del, axis=1)
        ndim -= len(index_del)

        for item in item_del:
            box.parameters.remove(item)

    if box.spectrum == "petitradtrans":
        n_line_species = box.attributes["n_line_species"]

        line_species = []
        for i in range(n_line_species):
            line_species.append(box.attributes[f"line_species{i}"])

    if "abund_nodes" not in box.attributes:
        box.attributes["abund_nodes"] = "None"

    if box.spectrum == "petitradtrans" and box.attributes["chemistry"] == "free":
        if box.attributes["abund_nodes"] == "None":
            box.parameters.append("c_h_ratio")
            box.parameters.append("o_h_ratio")
            box.parameters.append("c_o_ratio")

            ndim += 3

            abund_index = {}

            for line_item in line_species:
                abund_index[line_item] = box.parameters.index(line_item)

            c_h_ratio = np.zeros(samples.shape[0])
            o_h_ratio = np.zeros(samples.shape[0])
            c_o_ratio = np.zeros(samples.shape[0])

            for sample_item in samples:
                abund_dict = {}
                for line_item in line_species:
                    abund_dict[line_item] = sample_item[abund_index[line_item]]

                (
                    c_h_ratio[i],
                    o_h_ratio[i],
                    c_o_ratio[i],
                ) = retrieval_util.calc_metal_ratio(
                    abund_dict,
                    line_species,
                )

    if (
        vmr
        and box.spectrum == "petitradtrans"
        and box.attributes["chemistry"] == "free"
    ):
        print("Changing mass fractions to number fractions...", end="", flush=True)

        # Get all available line species
        all_line_species = retrieval_util.get_line_species()

        # Get the atomic and molecular masses
        masses = retrieval_util.atomic_masses()

        # Create array for the updated samples
        updated_samples = np.zeros(samples.shape)

        for i, samples_item in enumerate(samples):
            # Initiate a dictionary for the log10 mass fraction of the metals
            log_x_abund = {}

            for param_item in box.parameters:
                if param_item in all_line_species:
                    # Get the index of the parameter
                    param_index = box.parameters.index(param_item)

                    # Store log10 mass fraction in the dictionary
                    log_x_abund[param_item] = samples_item[param_index]

            # Create a dictionary with all mass fractions, including H2 and He
            x_abund = retrieval_util.mass_fractions(log_x_abund, line_species)

            # Calculate the mean molecular weight from the input mass fractions
            mmw = retrieval_util.mean_molecular_weight(x_abund)

            for param_item in box.parameters:
                if param_item in all_line_species:
                    # Get the index of the parameter
                    param_index = box.parameters.index(param_item)

                    # Overwrite the sample with the log10 number fraction
                    samples_item[param_index] = np.log10(
                        10.0 ** samples_item[param_index] * mmw / masses[param_item]
                    )

            # Store the updated sample to the array
            updated_samples[i,] = samples_item

        # Overwrite the samples in the SamplesBox
        box.samples = updated_samples

    print("Median sample:")
    for key, value in box.median_sample.items():
        print(f"   - {key} = {value:.2e}")

    if "gauss_mean" in box.parameters:
        param_index = np.argwhere(np.array(box.parameters) == "gauss_mean")[0]
        samples[:, param_index] *= 1e3  # (um) -> (nm)

    if "gauss_sigma" in box.parameters:
        param_index = np.argwhere(np.array(box.parameters) == "gauss_sigma")[0]
        samples[:, param_index] *= 1e3  # (um) -> (nm)

    if box.prob_sample is not None:
        par_val = tuple(box.prob_sample.values())

        print("Maximum posterior sample:")
        for key, value in box.prob_sample.items():
            print(f"   - {key} = {value:.2e}")

    for item in box.parameters:
        if item[0:11] == "wavelength_":
            param_index = box.parameters.index(item)

            # (um) -> (nm)
            box.samples[:, param_index] *= 1e3

    if output is None:
        print("Plotting the posterior...", end="", flush=True)
    else:
        print(f"Plotting the posterior: {output}...", end="", flush=True)

    # Include a subset of parameters

    if param_inc is not None:
        param_new = np.zeros((samples.shape[0], len(param_inc)))
        for i, item in enumerate(param_inc):
            param_index = box.parameters.index(item)
            param_new[:, i] = samples[:, param_index]

        box.parameters = param_inc
        ndim = len(param_inc)
        samples = param_new

    # Add [C/H], [O/H], and C/O if free abundances were retrieved

    if box.attributes["abund_nodes"] == "None":
        for param_item in box.parameters:
            if param_item.split("_")[0] == "H2O":
                samples = np.column_stack((samples, c_h_ratio, o_h_ratio, c_o_ratio))
                break

    # Include the derived bolometric luminosity

    if inc_luminosity:
        if "teff" in box.parameters and "radius" in box.parameters:
            teff_index = np.argwhere(np.array(box.parameters) == "teff")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]

            lum_planet = (
                4.0
                * np.pi
                * (samples[..., radius_index] * constants.R_JUP) ** 2
                * constants.SIGMA_SB
                * samples[..., teff_index] ** 4.0
                / constants.L_SUN
            )

            if "disk_teff" in box.parameters and "disk_radius" in box.parameters:
                teff_index = np.argwhere(np.array(box.parameters) == "disk_teff")[0]
                radius_index = np.argwhere(np.array(box.parameters) == "disk_radius")[0]

                lum_disk = (
                    4.0
                    * np.pi
                    * (samples[..., radius_index] * constants.R_JUP) ** 2
                    * constants.SIGMA_SB
                    * samples[..., teff_index] ** 4.0
                    / constants.L_SUN
                )

                samples = np.append(samples, np.log10(lum_planet + lum_disk), axis=-1)
                box.parameters.append("luminosity")
                ndim += 1

                samples = np.append(samples, lum_disk / lum_planet, axis=-1)
                box.parameters.append("luminosity_disk_planet")
                ndim += 1

            else:
                samples = np.append(samples, np.log10(lum_planet), axis=-1)
                box.parameters.append("luminosity")
                ndim += 1

        elif "teff_0" in box.parameters and "radius_0" in box.parameters:
            luminosity = 0.0

            for i in range(100):
                teff_index = np.argwhere(np.array(box.parameters) == f"teff_{i}")
                radius_index = np.argwhere(np.array(box.parameters) == f"radius_{i}")

                if len(teff_index) > 0 and len(radius_index) > 0:
                    luminosity += (
                        4.0
                        * np.pi
                        * (samples[..., radius_index[0]] * constants.R_JUP) ** 2
                        * constants.SIGMA_SB
                        * samples[..., teff_index[0]] ** 4.0
                        / constants.L_SUN
                    )

                else:
                    break

            samples = np.append(samples, np.log10(luminosity), axis=-1)
            box.parameters.append("luminosity")
            ndim += 1

            # teff_index = np.argwhere(np.array(box.parameters) == 'teff_0')
            # radius_index = np.argwhere(np.array(box.parameters) == 'radius_0')
            #
            # luminosity_0 = 4. * np.pi * (samples[..., radius_index[0]]*constants.R_JUP)**2 \
            #     * constants.SIGMA_SB * samples[..., teff_index[0]]**4. / constants.L_SUN
            #
            # samples = np.append(samples, np.log10(luminosity_0), axis=-1)
            # box.parameters.append('luminosity_0')
            # ndim += 1
            #
            # teff_index = np.argwhere(np.array(box.parameters) == 'teff_1')
            # radius_index = np.argwhere(np.array(box.parameters) == 'radius_1')
            #
            # luminosity_1 = 4. * np.pi * (samples[..., radius_index[0]]*constants.R_JUP)**2 \
            #     * constants.SIGMA_SB * samples[..., teff_index[0]]**4. / constants.L_SUN
            #
            # samples = np.append(samples, np.log10(luminosity_1), axis=-1)
            # box.parameters.append('luminosity_1')
            # ndim += 1
            #
            # teff_index_0 = np.argwhere(np.array(box.parameters) == 'teff_0')
            # radius_index_0 = np.argwhere(np.array(box.parameters) == 'radius_0')
            #
            # teff_index_1 = np.argwhere(np.array(box.parameters) == 'teff_1')
            # radius_index_1 = np.argwhere(np.array(box.parameters) == 'radius_1')
            #
            # luminosity_0 = 4. * np.pi * (samples[..., radius_index_0[0]]*constants.R_JUP)**2 \
            #     * constants.SIGMA_SB * samples[..., teff_index_0[0]]**4. / constants.L_SUN
            #
            # luminosity_1 = 4. * np.pi * (samples[..., radius_index_1[0]]*constants.R_JUP)**2 \
            #     * constants.SIGMA_SB * samples[..., teff_index_1[0]]**4. / constants.L_SUN
            #
            # samples = np.append(samples, np.log10(luminosity_0/luminosity_1), axis=-1)
            # box.parameters.append('luminosity_ratio')
            # ndim += 1

            # r_tmp = samples[..., radius_index_0[0]]*constants.R_JUP
            # lum_diff = (luminosity_1*constants.L_SUN-luminosity_0*constants.L_SUN)
            #
            # m_mdot = (3600.*24.*365.25)*lum_diff*r_tmp/constants.GRAVITY/constants.M_JUP**2
            #
            # samples = np.append(samples, m_mdot, axis=-1)
            # box.parameters.append('m_mdot')
            # ndim += 1

    # Include the derived mass

    if inc_mass:
        if "logg" in box.parameters and "radius" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]

            mass_samples = read_util.get_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("mass")
            ndim += 1

        else:
            warnings.warn(
                "Samples with the log(g) and radius are required for 'inc_mass=True'."
            )

    if inc_log_mass:
        if "logg" in box.parameters and "radius" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]

            mass_samples = read_util.get_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            mass_samples = np.log10(mass_samples)
            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("log_mass")
            ndim += 1

        else:
            warnings.warn(
                "Samples with the log(g) and radius are required for 'inc_log_mass=True'."
            )

    # Change from Jupiter to solar units if star

    if "radius" in box.parameters:
        radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]
        if object_type == "star":
            samples[:, radius_index] *= constants.R_JUP / constants.R_SUN

    if "mass" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "mass")[0]
        if object_type == "star":
            samples[:, mass_index] *= constants.M_JUP / constants.M_SUN

    if "log_mass" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "log_mass")[0]
        if object_type == "star":
            samples[:, mass_index] = np.log10(
                10.0 ** samples[:, mass_index] * constants.M_JUP / constants.M_SUN
            )

    # Include the log-likelihood

    if inc_loglike:
        # Get ln(L) of the samples
        ln_prob = box.ln_prob[..., np.newaxis]

        # Normalized by the maximum ln(L)
        ln_prob -= np.amax(ln_prob)

        # Convert ln(L) to log10(L)
        log_prob = ln_prob * np.exp(1.0)

        # Convert log10(L) to L
        prob = 10.0**log_prob

        # Normalize to an integrated probability of 1
        prob /= np.sum(prob)

        samples = np.append(samples, np.log10(prob), axis=-1)
        box.parameters.append("log_prob")
        ndim += 1

    # Remove abundances

    if not inc_abund and box.attributes["chemistry"] == "free":
        index_del = []
        item_del = []

        if box.attributes["abund_nodes"] == "None":
            for line_item in line_species:
                param_index = np.argwhere(np.array(box.parameters) == line_item)[0]
                index_del.append(param_index)
                item_del.append(line_item)

        else:
            for line_item in line_species:
                for node_idx in range(box.attributes["abund_nodes"]):
                    param_index = np.argwhere(
                        np.array(box.parameters) == f"{line_item}_{node_idx}"
                    )[0]
                    index_del.append(param_index)
                    item_del.append(f"{line_item}_{node_idx}")

        samples = np.delete(samples, index_del, axis=1)
        ndim -= len(index_del)

        for item in item_del:
            box.parameters.remove(item)

    # Update axes labels

    labels = plot_util.update_labels(box.parameters, object_type=object_type)

    # Check if parameter values were fixed

    index_sel = []
    index_del = []

    for i in range(ndim):
        if np.amin(samples[:, i]) == np.amax(samples[:, i]):
            index_del.append(i)
        else:
            index_sel.append(i)

    samples = samples[:, index_sel]

    for i in range(len(index_del) - 1, -1, -1):
        del labels[index_del[i]]

    ndim -= len(index_del)

    samples = samples.reshape((-1, ndim))

    if isinstance(title_fmt, list) and len(title_fmt) != ndim:
        raise ValueError(
            f"The number of items in the list of 'title_fmt' ({len(title_fmt)}) is "
            f"not equal to the number of dimensions of the samples ({ndim})."
        )

    hist_titles = []

    for i, item in enumerate(labels):
        unit_start = item.find("(")

        if unit_start == -1:
            param_label = item
            unit_label = None

        else:
            param_label = item[:unit_start]
            # Remove parenthesis from the units
            unit_label = item[unit_start + 1 : -1]

        q_16, q_50, q_84 = corner.quantile(samples[:, i], [0.16, 0.5, 0.84])
        q_minus, q_plus = q_50 - q_16, q_84 - q_50

        if isinstance(title_fmt, str):
            fmt = "{{0:{0}}}".format(title_fmt).format

        elif isinstance(title_fmt, list):
            fmt = "{{0:{0}}}".format(title_fmt[i]).format

        best_fit = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        best_fit = best_fit.format(fmt(q_50), fmt(q_minus), fmt(q_plus))

        if unit_label is None:
            hist_title = f"{param_label} = {best_fit}"

        else:
            hist_title = f"{param_label} = {best_fit} {unit_label}"

        hist_titles.append(hist_title)

    fig = corner.corner(
        samples,
        quantiles=[0.16, 0.5, 0.84],
        labels=labels,
        label_kwargs={"fontsize": 13},
        titles=hist_titles,
        show_titles=True,
        title_fmt=None,
        title_kwargs={"fontsize": 12},
    )

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        for j in range(ndim):
            if i >= j:
                ax = axes[i, j]

                ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

                labelleft = j == 0 and i != 0
                labelbottom = i == ndim - 1

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
                    labelleft=labelleft,
                    labelbottom=labelbottom,
                    labelright=False,
                    labeltop=False,
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
                    labelleft=labelleft,
                    labelbottom=labelbottom,
                    labelright=False,
                    labeltop=False,
                )

                if limits is not None:
                    ax.set_xlim(limits[j])

                if max_prob:
                    ax.axvline(par_val[j], color="tomato")

                if i > j:
                    if max_prob:
                        ax.axhline(par_val[i], color="tomato")
                        ax.plot(par_val[j], par_val[i], "s", color="tomato")

                    if limits is not None:
                        ax.set_ylim(limits[i])

                if offset is not None:
                    ax.get_xaxis().set_label_coords(0.5, offset[0])
                    ax.get_yaxis().set_label_coords(offset[1], 0.5)

                else:
                    ax.get_xaxis().set_label_coords(0.5, -0.26)
                    ax.get_yaxis().set_label_coords(-0.27, 0.5)

    if title:
        fig.suptitle(title, y=1.02, fontsize=16)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig


@typechecked
def plot_mag_posterior(
    tag: str,
    filter_name: str,
    burnin: int = None,
    xlim: Tuple[float, float] = None,
    output: Optional[str] = None,
) -> Tuple[np.ndarray, mpl.figure.Figure]:
    """
    Function to plot the posterior distribution of the synthetic
    magnitudes. The posterior samples are also returned.

    Parameters
    ----------
    tag : str
        Database tag with the posterior samples.
    filter_name : str
        Filter name.
    burnin : int, None
        Number of burnin steps to exclude. All samples are
        used if the argument is set to ``None``.
    xlim : tuple(float, float), None
        Axis limits. Automatically set if the argument is
        set to ``None``.
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.

    Returns
    -------
    np.ndarray
        Array with the posterior samples of the magnitude.
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    """

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    species_db = database.Database()

    samples = species_db.get_mcmc_photometry(tag, filter_name, burnin)

    if output is None:
        print("Plotting photometry samples...", end="", flush=True)
    else:
        print(f"Plotting photometry samples: {output}...", end="", flush=True)

    fig = corner.corner(
        samples,
        labels=["Magnitude"],
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={"fontsize": 13.0},
        show_titles=True,
        title_kwargs={"fontsize": 12.0},
        title_fmt=".2f",
    )

    axes = np.array(fig.axes).reshape((1, 1))

    ax = axes[0, 0]

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

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.get_xaxis().set_label_coords(0.5, -0.26)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return samples, fig


@typechecked
def plot_size_distributions(
    tag: str,
    burnin: Optional[int] = None,
    random: Optional[int] = None,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
) -> mpl.figure.Figure:
    """
    Function to plot random samples of the log-normal
    or power-law size distributions.

    Parameters
    ----------
    tag : str
        Database tag with the samples.
    burnin : int, None
        Number of burnin steps to exclude. All samples are used if the
        argument is set to ``None``. Only required after running MCMC
        with :func:`~species.analysis.fit_model.FitModel.run_mcmc`.
    random : int, None
        Number of randomly selected samples. All samples are used
        if the argument set to ``None``.
    offset : tuple(float, float), None
        Offset of the x- and y-axis label. Default values are used
        if the argument set to ``None``.
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
        print("Plotting size distributions...", end="", flush=True)
    else:
        print(f"Plotting size distributions: {output}...", end="", flush=True)

    if burnin is None:
        burnin = 0

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    species_db = database.Database()
    box = species_db.get_samples(tag)

    if "lognorm_radius" not in box.parameters and "powerlaw_max" not in box.parameters:
        raise ValueError(
            "The SamplesBox does not contain extinction parameter for a log-normal "
            "or power-law size distribution."
        )

    samples = box.samples

    if samples.ndim == 2 and random is not None:
        ran_index = np.random.randint(samples.shape[0], size=random)
        samples = samples[ran_index,]

    elif samples.ndim == 3:
        if burnin > samples.shape[1]:
            raise ValueError(
                f"The 'burnin' value is larger than the number of steps "
                f"({samples.shape[1]}) that are made by the walkers."
            )

        samples = samples[:, burnin:, :]

        ran_walker = np.random.randint(samples.shape[0], size=random)
        ran_step = np.random.randint(samples.shape[1], size=random)
        samples = samples[ran_walker, ran_step, :]

    if "lognorm_radius" in box.parameters:
        log_r_index = box.parameters.index("lognorm_radius")
        sigma_index = box.parameters.index("lognorm_sigma")

        log_r_g = samples[:, log_r_index]
        sigma_g = samples[:, sigma_index]

    if "powerlaw_max" in box.parameters:
        r_max_index = box.parameters.index("powerlaw_max")
        exponent_index = box.parameters.index("powerlaw_exp")

        r_max = samples[:, r_max_index]
        exponent = samples[:, exponent_index]

    fig = plt.figure(figsize=(6, 3))
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

    ax.set_xlabel("Grain size (µm)", fontsize=12)
    ax.set_ylabel("dn/dr", fontsize=12)

    ax.set_xscale("log")

    if "powerlaw_max" in box.parameters:
        ax.set_yscale("log")

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.22)
        ax.get_yaxis().set_label_coords(-0.09, 0.5)

    for i in range(samples.shape[0]):
        if "lognorm_radius" in box.parameters:
            dn_grains, r_width, radii = dust_util.log_normal_distribution(
                10.0 ** log_r_g[i], sigma_g[i], 1000
            )

            # Exclude radii smaller than 1 nm
            indices = np.argwhere(radii >= 1e-3)

            dn_grains = dn_grains[indices]
            r_width = r_width[indices]
            radii = radii[indices]

        elif "powerlaw_max" in box.parameters:
            dn_grains, r_width, radii = dust_util.power_law_distribution(
                exponent[i], 1e-3, 10.0 ** r_max[i], 1000
            )

        ax.plot(radii, dn_grains / r_width, ls="-", lw=0.5, color="black", alpha=0.5)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig


@typechecked
def plot_extinction(
    tag: str,
    burnin: Optional[int] = None,
    random: Optional[int] = None,
    wavel_range: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
) -> mpl.figure.Figure:
    """
    Function to plot random samples of the extinction, either from
    fitting a size distribution of enstatite grains (``dust_radius``,
    ``dust_sigma``, and ``dust_ext``), or from fitting ISM extinction
    (``ism_ext`` and optionally ``ism_red``).

    Parameters
    ----------
    tag : str
        Database tag with the samples.
    burnin : int, None
        Number of burnin steps to exclude. All samples are used if the
        argument is set to ``None``. Only required after running MCMC
        with :func:`~species.analysis.fit_model.FitModel.run_mcmc`.
    random : int, None
        Number of randomly selected samples. All samples are used if
        the argument is set to ``None``.
    wavel_range : tuple(float, float), None
        Wavelength range (um) for the extinction. The default
        wavelength range (0.4, 10.) is used if the argument is
        set to ``None``.
    xlim : tuple(float, float), None
        Limits of the wavelength axis. The range is set automatically
        if the argument is set to ``None``.
    ylim : tuple(float, float)
        Limits of the extinction axis. The range is set automatically
        if the argument is set to ``None``.
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

    if burnin is None:
        burnin = 0

    if wavel_range is None:
        wavel_range = (0.4, 10.0)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    species_db = database.Database()
    box = species_db.get_samples(tag)

    samples = box.samples

    if samples.ndim == 2 and random is not None:
        ran_index = np.random.randint(samples.shape[0], size=random)
        samples = samples[ran_index,]

    elif samples.ndim == 3:
        if burnin > samples.shape[1]:
            raise ValueError(
                f"The 'burnin' value is larger than the number of steps "
                f"({samples.shape[1]}) that are made by the walkers."
            )

        samples = samples[:, burnin:, :]

        ran_walker = np.random.randint(samples.shape[0], size=random)
        ran_step = np.random.randint(samples.shape[1], size=random)
        samples = samples[ran_walker, ran_step, :]

    fig = plt.figure(figsize=(6, 3))
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

    ax.set_xlabel("Wavelength (\N{GREEK SMALL LETTER MU}m)", fontsize=12)
    ax.set_ylabel("Extinction (mag)", fontsize=12)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    else:
        ax.get_xaxis().set_label_coords(0.5, -0.22)
        ax.get_yaxis().set_label_coords(-0.09, 0.5)

    sample_wavel = np.linspace(wavel_range[0], wavel_range[1], 100)

    if (
        "lognorm_radius" in box.parameters
        and "lognorm_sigma" in box.parameters
        and "lognorm_ext" in box.parameters
    ):
        cross_optical, dust_radius, dust_sigma = dust_util.interp_lognorm([], [])

        log_r_index = box.parameters.index("lognorm_radius")
        sigma_index = box.parameters.index("lognorm_sigma")
        ext_index = box.parameters.index("lognorm_ext")

        log_r_g = samples[:, log_r_index]
        sigma_g = samples[:, sigma_index]
        dust_ext = samples[:, ext_index]

        database_path = dust_util.check_dust_database()

        with h5py.File(database_path, "r") as h5_file:
            cross_section = np.asarray(
                h5_file["dust/lognorm/mgsio3/crystalline/cross_section"]
            )
            wavelength = np.asarray(
                h5_file["dust/lognorm/mgsio3/crystalline/wavelength"]
            )

        cross_interp = RegularGridInterpolator(
            (wavelength, dust_radius, dust_sigma), cross_section
        )

        for i in range(samples.shape[0]):
            cross_tmp = cross_optical["Generic/Bessell.V"](
                (10.0 ** log_r_g[i], sigma_g[i])
            )

            n_grains = dust_ext[i] / cross_tmp / 2.5 / np.log10(np.exp(1.0))

            sample_cross = np.zeros(sample_wavel.shape)

            for j, item in enumerate(sample_wavel):
                sample_cross[j] = cross_interp((item, 10.0 ** log_r_g[i], sigma_g[i]))

            sample_ext = 2.5 * np.log10(np.exp(1.0)) * sample_cross * n_grains

            ax.plot(sample_wavel, sample_ext, ls="-", lw=0.5, color="black", alpha=0.5)

    elif (
        "powerlaw_max" in box.parameters
        and "powerlaw_exp" in box.parameters
        and "powerlaw_ext" in box.parameters
    ):
        cross_optical, dust_max, dust_exp = dust_util.interp_powerlaw([], [])

        r_max_index = box.parameters.index("powerlaw_max")
        exp_index = box.parameters.index("powerlaw_exp")
        ext_index = box.parameters.index("powerlaw_ext")

        r_max = samples[:, r_max_index]
        exponent = samples[:, exp_index]
        dust_ext = samples[:, ext_index]

        database_path = dust_util.check_dust_database()

        with h5py.File(database_path, "r") as h5_file:
            cross_section = np.asarray(
                h5_file["dust/powerlaw/mgsio3/crystalline/cross_section"]
            )
            wavelength = np.asarray(
                h5_file["dust/powerlaw/mgsio3/crystalline/wavelength"]
            )

        cross_interp = RegularGridInterpolator(
            (wavelength, dust_max, dust_exp), cross_section
        )

        for i in range(samples.shape[0]):
            cross_tmp = cross_optical["Generic/Bessell.V"](
                (10.0 ** r_max[i], exponent[i])
            )

            n_grains = dust_ext[i] / cross_tmp / 2.5 / np.log10(np.exp(1.0))

            sample_cross = np.zeros(sample_wavel.shape)

            for j, item in enumerate(sample_wavel):
                sample_cross[j] = cross_interp((item, 10.0 ** r_max[i], exponent[i]))

            sample_ext = 2.5 * np.log10(np.exp(1.0)) * sample_cross * n_grains

            ax.plot(sample_wavel, sample_ext, ls="-", lw=0.5, color="black", alpha=0.5)

    elif "ism_ext" in box.parameters:
        ext_index = box.parameters.index("ism_ext")
        ism_ext = samples[:, ext_index]

        if "ism_red" in box.parameters:
            red_index = box.parameters.index("ism_red")
            ism_red = samples[:, red_index]

        else:
            # Use default ISM redenning (R_V = 3.1) if ism_red was not fitted
            ism_red = np.full(samples.shape[0], 3.1)

        for i in range(samples.shape[0]):
            sample_ext = dust_util.ism_extinction(ism_ext[i], ism_red[i], sample_wavel)

            ax.plot(sample_wavel, sample_ext, ls="-", lw=0.5, color="black", alpha=0.5)

    else:
        raise ValueError("The SamplesBox does not contain extinction parameters.")

    if output is None:
        print("Plotting extinction...", end="", flush=True)
    else:
        print(f"Plotting extinction: {output}...", end="", flush=True)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")

    print(" [DONE]")

    return fig
