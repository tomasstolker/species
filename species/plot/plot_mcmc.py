"""
Module for plotting MCMC results.
"""

import warnings

from typing import List, Optional, Tuple, Union

import corner
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm

from species.core import constants
from species.read.read_isochrone import ReadIsochrone
from species.read.read_model import ReadModel
from species.util.convert_util import logg_to_mass
from species.util.core_util import print_section
from species.util.plot_util import update_labels
from species.util.dust_util import (
    interp_lognorm,
    interp_powerlaw,
    ism_extinction,
    log_normal_distribution,
    power_law_distribution,
)
from species.util.retrieval_util import (
    atomic_masses,
    calc_metal_ratio,
    get_line_species,
    mass_frac_dict,
    mean_molecular_weight,
)


@typechecked
def plot_posterior(
    tag: str,
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
    show_priors: bool = False,
    show_grid: bool = True,
    kwargs_corner: Optional[dict] = None,
) -> mpl.figure.Figure:
    """
    Function to plot the posterior distribution of the
    model parameters. For model grids, colored ticks show
    the grid points, which can be used to check the width
    of the posterior distributions relative to the spacing
    between the grid points.

    Parameters
    ----------
    tag : str
        Database tag with the samples.
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
        Plot the position of the sample with the maximum likelihood
        probability. The sample may need to match with the maximum
        posterior probability, depending on the use of any normal
        priors.
    vmr : bool
        Plot the volume mixing ratios (i.e. number fractions)
        instead of the mass fractions of the retrieved species with
        :class:`~species.fit.retrieval.AtmosphericRetrieval`.
    inc_luminosity : bool
        Include the log10 of the luminosity in the posterior plot
        as calculated from the effective temperature and radius.
    inc_mass : bool
        Include the mass in the posterior plot as calculated
        from the surface gravity and radius.
    inc_log_mass : bool
        Include the logarithm of the mass, :math:`\\log_{10}{M}`, in
        the posterior plot, as calculated from the surface gravity
        and radius.
    inc_pt_param : bool
        Include the parameters of the pressure-temperature profile.
        Only used if the ``tag`` contains samples obtained with
        :class:`~species.fit.retrieval.AtmosphericRetrieval`.
    inc_loglike : bool
        Include the log-likelihood, :math:`\\ln{L}`, as additional
        parameter in the corner plot.
    inc_abund : bool
        Include the abundances when retrieving free abundances with
        :class:`~species.fit.retrieval.AtmosphericRetrieval`.
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
    show_priors : bool
        Plot the normal priors in the diagonal panels together with the
        1D marginalized posterior distributions (default: False). This
        will only show the priors that had a normal distribution, so
        those that were set with the ``normal_prior`` parameter in
        :class:`~species.fit.fit_model.FitModel` and
        :class:`~species.fit.retrieval.AtmosphericRetrieval.setup_retrieval`.
    show_grid : bool
        Show with lines for the grid points of the atmospheric model
        on the 1D and 2D marginalized posteriors (default: True).
        This parameter has only an effect for results obtained with
        :class:`~species.fit.fit_model.FitModel`.
    kwargs_corner : dict, None
        Dictionary with keyword arguments that can be used to adjust the
        parameters of the `corner() function
        <https://corner.readthedocs.io/en/latest/api/>`_ of ``corner.py``.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    """

    from species.data.database import Database

    species_db = Database()

    box = species_db.get_samples(tag)
    samples = box.samples

    print_section("Plot posterior distributions")

    print(f"Database tag: {tag}")
    print(f"Object type: {object_type}")
    print(f"Manual parameters: {param_inc}\n")

    if "model_type" in box.attributes:
        print((f"Model type: {box.attributes['model_type']}"))
    elif "spec_type" in box.attributes:
        print((f"Model type: {box.attributes['spec_type']}"))

    if "model_name" in box.attributes:
        print((f"Model name: {box.attributes['model_name']}"))
    elif "spec_name" in box.attributes:
        print((f"Model type: {box.attributes['spec_name']}"))

    if "sampler" in box.attributes:
        print((f"Sampler: {box.attributes['sampler']}"))

    print(f"\nShow priors: {show_priors}")
    print(f"Show grid: {show_grid}")

    # Create empty dictionary if needed

    if kwargs_corner is None:
        kwargs_corner = {}

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

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

    if not inc_pt_param and box.model_name == "petitradtrans":
        pt_param = [
            "tint",
            "t1",
            "t2",
            "t3",
            "alpha",
            "log_delta",
            "T_bottom",
            "PTslope_1",
            "PTslope_2",
            "PTslope_3",
            "PTslope_4",
            "PTslope_5",
            "PTslope_6",
        ]

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

    if box.model_name == "petitradtrans":
        n_line_species = box.attributes["n_line_species"]

        line_species = []
        for i in range(n_line_species):
            line_species.append(box.attributes[f"line_species{i}"])

    if "abund_nodes" not in box.attributes:
        box.attributes["abund_nodes"] = "None"

    if box.model_name == "petitradtrans" and box.attributes["chemistry"] == "free":
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

            for i, sample_item in enumerate(samples):
                abund_dict = {}
                for line_item in line_species:
                    abund_dict[line_item] = sample_item[abund_index[line_item]]

                (
                    c_h_ratio[i],
                    o_h_ratio[i],
                    c_o_ratio[i],
                ) = calc_metal_ratio(
                    abund_dict,
                    line_species,
                )

    if (
        vmr
        and box.model_name == "petitradtrans"
        and box.attributes["chemistry"] == "free"
    ):
        print("Changing mass fractions to number fractions...", end="", flush=True)

        # Get all available line species
        all_line_species = get_line_species()

        # Get the atomic and molecular masses
        masses = atomic_masses()

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
            x_abund = mass_frac_dict(log_x_abund, line_species)

            # Calculate the mean molecular weight from the input mass fractions
            mmw = mean_molecular_weight(x_abund)

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

    print("\nMedian parameters:")
    for param_key, param_value in box.median_sample.items():
        if isinstance(param_value, float):
            if -0.1 < param_value < 0.1:
                print(f"   - {param_key} = {param_value:.2e}")
            else:
                print(f"   - {param_key} = {param_value:.2f}")

    if "gauss_mean" in box.parameters:
        param_index = np.argwhere(np.array(box.parameters) == "gauss_mean")[0]
        samples[:, param_index] *= 1e3  # (um) -> (nm)

    if "gauss_sigma" in box.parameters:
        param_index = np.argwhere(np.array(box.parameters) == "gauss_sigma")[0]
        samples[:, param_index] *= 1e3  # (um) -> (nm)

    if box.prob_sample is not None:
        print("\nSample with the maximum likelihood:")
        for param_key, param_value in box.prob_sample.items():
            if isinstance(param_value, float):
                if -0.1 < param_value < 0.1:
                    print(f"   - {param_key} = {param_value:.2e}")
                else:
                    print(f"   - {param_key} = {param_value:.2f}")

    for param_item in box.parameters:
        if param_item[0:11] == "wavelength_":
            param_index = box.parameters.index(param_item)

            # (um) -> (nm)
            box.samples[:, param_index] *= 1e3

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

            lum_atm = (
                4.0
                * np.pi
                * (samples[..., radius_index] * constants.R_JUP) ** 2
                * constants.SIGMA_SB
                * samples[..., teff_index] ** 4.0
                / constants.L_SUN
            )

            n_disk = 0

            if "disk_teff" in box.parameters and "disk_radius" in box.parameters:
                n_disk = 1

            else:
                for disk_idx in range(100):
                    if (
                        f"disk_teff_{disk_idx}" in box.parameters
                        and f"disk_radius_{disk_idx}" in box.parameters
                    ):
                        n_disk += 1
                    else:
                        break

            if n_disk == 1:
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

                samples = np.append(samples, np.log10(lum_atm), axis=-1)
                box.parameters.append("log_lum_atm")
                ndim += 1

                samples = np.append(samples, np.log10(lum_disk), axis=-1)
                box.parameters.append("log_lum_disk")
                ndim += 1

                radius_bb = np.sqrt(
                    lum_atm
                    * constants.L_SUN
                    / (
                        16.0
                        * np.pi
                        * constants.SIGMA_SB
                        * samples[..., teff_index] ** 4
                    )
                )

                samples = np.append(samples, radius_bb / constants.R_JUP, axis=-1)
                box.parameters.append("radius_bb")
                ndim += 1

            elif n_disk > 1:
                lum_disk = 0.0

                for disk_idx in range(n_disk):
                    teff_index = np.argwhere(
                        np.array(box.parameters) == f"disk_teff_{disk_idx}"
                    )[0]
                    radius_index = np.argwhere(
                        np.array(box.parameters) == f"disk_radius_{disk_idx}"
                    )[0]

                    lum_disk += (
                        4.0
                        * np.pi
                        * (samples[..., radius_index] * constants.R_JUP) ** 2
                        * constants.SIGMA_SB
                        * samples[..., teff_index] ** 4.0
                        / constants.L_SUN
                    )

                    radius_bb = np.sqrt(
                        lum_atm
                        * constants.L_SUN
                        / (
                            16.0
                            * np.pi
                            * constants.SIGMA_SB
                            * samples[..., teff_index] ** 4
                        )
                    )

                    samples = np.append(samples, radius_bb / constants.R_JUP, axis=-1)
                    box.parameters.append(f"radius_bb_{disk_idx}")
                    ndim += 1

                samples = np.append(samples, np.log10(lum_atm), axis=-1)
                box.parameters.append("log_lum_atm")
                ndim += 1

                samples = np.append(samples, np.log10(lum_disk), axis=-1)
                box.parameters.append("log_lum_disk")
                ndim += 1

            else:
                samples = np.append(samples, np.log10(lum_atm), axis=-1)
                box.parameters.append("log_lum_atm")
                ndim += 1

        for i in range(100):
            if f"teff_{i}" in box.parameters and f"radius_{i}" in box.parameters:
                teff_index = np.argwhere(np.array(box.parameters) == f"teff_{i}")
                radius_index = np.argwhere(np.array(box.parameters) == f"radius_{i}")

                luminosity = (
                    4.0
                    * np.pi
                    * (samples[..., radius_index[0]] * constants.R_JUP) ** 2
                    * constants.SIGMA_SB
                    * samples[..., teff_index[0]] ** 4.0
                    / constants.L_SUN
                )

                samples = np.append(samples, np.log10(luminosity), axis=-1)
                box.parameters.append(f"log_lum_{i}")
                ndim += 1

            else:
                break

        if "teff_0" in box.parameters and "radius_0" in box.parameters:
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
            box.parameters.append("log_lum")
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
        check_param = False

        if "logg" in box.parameters and "radius" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]

            mass_samples = logg_to_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("mass")
            ndim += 1

            check_param = True

        if "logg_0" in box.parameters and "radius_0" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg_0")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius_0")[0]

            mass_samples = logg_to_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("mass_0")
            ndim += 1

            check_param = True

        if "logg_1" in box.parameters and "radius_1" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg_1")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius_1")[0]

            mass_samples = logg_to_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("mass_1")
            ndim += 1

            check_param = True

        if not check_param:
            warnings.warn(
                "Samples with the log(g) and radius are required for 'inc_mass=True'."
            )

    if inc_log_mass:
        check_param = False

        if "logg" in box.parameters and "radius" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]

            mass_samples = logg_to_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            mass_samples = np.log10(mass_samples)
            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("log_mass")
            ndim += 1

            check_param = True

        if "logg_0" in box.parameters and "radius_0" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg_0")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius_0")[0]

            mass_samples = logg_to_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            mass_samples = np.log10(mass_samples)
            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("log_mass_0")
            ndim += 1

            check_param = True

        if "logg_1" in box.parameters and "radius_1" in box.parameters:
            logg_index = np.argwhere(np.array(box.parameters) == "logg_1")[0]
            radius_index = np.argwhere(np.array(box.parameters) == "radius_1")[0]

            mass_samples = logg_to_mass(
                samples[..., logg_index], samples[..., radius_index]
            )

            mass_samples = np.log10(mass_samples)
            samples = np.append(samples, mass_samples, axis=-1)

            box.parameters.append("log_mass_1")
            ndim += 1

            check_param = True

        if not check_param:
            warnings.warn(
                "Samples with the log(g) and radius are required for 'inc_log_mass=True'."
            )

    # Change from Jupiter to solar units if star

    if "radius" in box.parameters:
        radius_index = np.argwhere(np.array(box.parameters) == "radius")[0]
        if object_type == "star":
            samples[:, radius_index] *= constants.R_JUP / constants.R_SUN

    for radius_idx in range(100):
        if f"radius_{radius_idx}" in box.parameters:
            radius_index = np.argwhere(
                np.array(box.parameters) == f"radius_{radius_idx}"
            )[0]
            if object_type == "star":
                samples[:, radius_index] *= constants.R_JUP / constants.R_SUN
        else:
            break

    if "mass" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "mass")[0]
        if object_type == "star":
            samples[:, mass_index] *= constants.M_JUP / constants.M_SUN

    if "mass_0" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "mass_0")[0]
        if object_type == "star":
            samples[:, mass_index] *= constants.M_JUP / constants.M_SUN

    if "mass_1" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "mass_1")[0]
        if object_type == "star":
            samples[:, mass_index] *= constants.M_JUP / constants.M_SUN

    if "log_mass" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "log_mass")[0]
        if object_type == "star":
            samples[:, mass_index] = np.log10(
                10.0 ** samples[:, mass_index] * constants.M_JUP / constants.M_SUN
            )

    if "log_mass_0" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "log_mass_0")[0]
        if object_type == "star":
            samples[:, mass_index] = np.log10(
                10.0 ** samples[:, mass_index] * constants.M_JUP / constants.M_SUN
            )

    if "log_mass_1" in box.parameters:
        mass_index = np.argwhere(np.array(box.parameters) == "log_mass_1")[0]
        if object_type == "star":
            samples[:, mass_index] = np.log10(
                10.0 ** samples[:, mass_index] * constants.M_JUP / constants.M_SUN
            )

    if "disk_radius" in box.parameters:
        radius_index = np.argwhere(np.array(box.parameters) == "disk_radius")[0]
        if object_type == "star":
            samples[:, radius_index] *= constants.R_JUP / constants.AU

    for disk_idx in range(100):
        if f"disk_radius_{disk_idx}" in box.parameters:
            radius_index = np.argwhere(
                np.array(box.parameters) == f"disk_radius_{disk_idx}"
            )[0]
            if object_type == "star":
                samples[:, radius_index] *= constants.R_JUP / constants.AU
        else:
            break

    if "radius_bb" in box.parameters:
        radius_index = np.argwhere(np.array(box.parameters) == "radius_bb")[0]
        if object_type == "star":
            samples[:, radius_index] *= constants.R_JUP / constants.AU

    for disk_idx in range(100):
        if f"radius_bb_{disk_idx}" in box.parameters:
            radius_index = np.argwhere(
                np.array(box.parameters) == f"radius_bb_{disk_idx}"
            )[0]
            if object_type == "star":
                samples[:, radius_index] *= constants.R_JUP / constants.AU
        else:
            break

    # Include the log-likelihood, ln(L)

    if inc_loglike:
        # Get ln(L) of the samples
        ln_prob = box.ln_prob[..., np.newaxis]

        # Normalized by the maximum ln(L)
        # ln_prob -= np.amax(ln_prob)

        # Convert ln(L) to log10(L)
        # log_prob = ln_prob * np.exp(1.0)

        # Convert log10(L) to L
        # prob = 10.0**log_prob

        # Normalize to an integrated probability of 1
        # prob /= np.sum(prob)

        # samples = np.append(samples, np.log10(prob), axis=-1)
        # box.parameters.append("log_prob")
        # ndim += 1

        samples = np.append(samples, ln_prob, axis=-1)
        box.parameters.append("ln_prob")
        ndim += 1

    # Remove abundances

    if (
        not inc_abund
        and "chemistry" in box.attributes
        and box.attributes["chemistry"] == "free"
    ):
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

    # Include a subset of parameters

    if param_inc is not None:
        param_new = np.zeros((samples.shape[0], len(param_inc)))
        param_inc_new = []
        for param_idx, param_item in enumerate(param_inc):
            if param_item in box.parameters:
                param_index = box.parameters.index(param_item)
                param_new[:, param_idx] = samples[:, param_index]
                param_inc_new.append(param_item)

        box.parameters = param_inc_new
        ndim = len(param_inc_new)
        samples = param_new

    # Only for fitting evolutionary models
    # Remove index from parameter names when fitting 1 planet

    if "model_type" in box.attributes and box.attributes["model_type"] == "evolution":
        if box.attributes["n_planets"] == 1:
            param_copy = box.parameters.copy()
            box.parameters = []

            for param_item in param_copy:
                if param_item[-2:] == "_0":
                    box.parameters.append(param_item[:-2])
                else:
                    box.parameters.append(param_item)

    # Parameters to be included in the corner plot

    print("\nParameters included in corner plot:")
    for param_item in box.parameters:
        print(f"   - {param_item}")

    # Update axes labels

    box_param = box.parameters.copy()
    labels = update_labels(box.parameters, object_type=object_type)

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

    # Get parameter values of maximum likelihood

    if max_prob:
        max_idx = np.argmax(box.ln_prob)
        max_sample = samples[max_idx,]

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

    # Create corner plot

    fig = corner.corner(
        samples,
        quantiles=[0.16, 0.5, 0.84],
        labels=labels,
        label_kwargs={"fontsize": 13},
        titles=hist_titles,
        show_titles=True,
        title_fmt=None,
        title_kwargs={"fontsize": 12},
        **kwargs_corner,
    )

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]

            if show_priors and i == j and box_param[i] in box.normal_priors:
                norm_param = box.normal_priors[box_param[i]]

                x_norm = np.linspace(
                    norm_param[0] - 5.0 * norm_param[1],
                    norm_param[0] + 5.0 * norm_param[1],
                    200,
                )

                y_norm = norm.pdf(x_norm, norm_param[0], norm_param[1])

                ax.plot(
                    x_norm,
                    0.9 * ax.get_ylim()[1] * y_norm / np.amax(y_norm),
                    ls=":",
                    lw=2.0,
                    color="dodgerblue",
                )

            if i >= j:
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
                    ax.axvline(max_sample[j], color="tomato")

                if i > j:
                    if max_prob:
                        ax.axhline(max_sample[i], color="tomato")
                        ax.plot(max_sample[j], max_sample[i], "s", color="tomato")

                    if limits is not None:
                        ax.set_ylim(limits[i])

                if offset is not None:
                    ax.get_xaxis().set_label_coords(0.5, offset[0])
                    ax.get_yaxis().set_label_coords(offset[1], 0.5)

                else:
                    ax.get_xaxis().set_label_coords(0.5, -0.26)
                    ax.get_yaxis().set_label_coords(-0.27, 0.5)

    if show_grid:
        if (
            box.attributes["model_type"] == "atmosphere"
            and box.attributes["model_name"] != "petitradtrans"
        ):
            read_model = ReadModel(box.attributes["model_name"])
            grid_points = read_model.get_points()

        elif (
            box.attributes["model_type"] == "evolution"
            and box.attributes["regular_grid"] == True
        ):
            read_model = ReadIsochrone(box.attributes["model_name"])
            grid_points = read_model.get_points()

        else:
            grid_points = None

        if grid_points is not None:
            for i in range(ndim):
                for j in range(ndim):
                    ax = axes[i, j]

                    if (i == j or i > j) and box_param[j] in grid_points:
                        ax_ymin, ax_ymax = ax.get_ylim()
                        y_span = ax_ymax - ax_ymin

                        ax.vlines(
                            grid_points[box_param[j]],
                            ymin=ax_ymin,
                            ymax=ax_ymin + 0.08 * y_span,
                            colors="cadetblue",
                            linestyles="solid",
                            linewidth=0.8,
                            zorder=1,
                        )

                        ax.vlines(
                            grid_points[box_param[j]],
                            ymin=ax_ymax - 0.08 * y_span,
                            ymax=ax_ymax,
                            colors="cadetblue",
                            linestyles="solid",
                            linewidth=0.8,
                            zorder=1,
                        )

                    elif i > j and box_param[i] in grid_points:
                        ax_xmin, ax_xmax = ax.get_xlim()
                        x_span = ax_xmax - ax_xmin

                        ax.hlines(
                            grid_points[box_param[i]],
                            xmin=ax_xmin,
                            xmax=ax_xmin + 0.08 * x_span,
                            colors="cadetblue",
                            linestyles="solid",
                            linewidth=0.8,
                            zorder=1,
                        )

                        ax.hlines(
                            grid_points[box_param[i]],
                            xmin=ax_xmax - 0.08 * x_span,
                            xmax=ax_xmax,
                            colors="cadetblue",
                            linestyles="solid",
                            linewidth=0.8,
                            zorder=1,
                        )

    if title:
        fig.suptitle(title, y=1.02, fontsize=16)

    if output is None:
        plt.show()
    else:
        print(f"\nOutput: {output}")
        plt.savefig(output, bbox_inches="tight")

    return fig


@typechecked
def plot_mag_posterior(
    tag: str,
    filter_name: str,
    n_samples: Optional[int] = None,
    xlim: Optional[Tuple[float, float]] = None,
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
    n_samples : int, None
        Number of randomly drawn samples. All samples of the posterior
        are selected if the arguments is set to ``None``.
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

    if output is None:
        print("Plotting photometry samples...", end="", flush=True)
    else:
        print(f"Plotting photometry samples: {output}...", end="", flush=True)

    from species.data.database import Database

    species_db = Database()

    samples = species_db.get_mcmc_photometry(tag, filter_name, random=n_samples)

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

    from species.data.database import Database

    species_db = Database()
    box = species_db.get_samples(tag)

    print_section("Plot size distributions")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    if "lognorm_ext" not in box.parameters and "powerlaw_ext" not in box.parameters:
        raise ValueError(
            "The SamplesBox does not contain extinction parameter "
            "for a log-normal or power-law size distribution."
        )

    samples = box.samples

    if random is not None:
        ran_index = np.random.randint(samples.shape[0], size=random)
        samples = samples[ran_index,]

    print(f"Database tag: {tag}")
    print(f"Number of samples: {samples.shape[0]}")

    print(f"\nLabel offset: {offset}")

    if "lognorm_ext" in box.parameters:
        log_r_index = box.parameters.index("lognorm_radius")
        sigma_index = box.parameters.index("lognorm_sigma")

        log_r_g = samples[:, log_r_index]
        sigma_g = samples[:, sigma_index]

    if "powerlaw_ext" in box.parameters:
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

    if "powerlaw_ext" in box.parameters:
        ax.set_yscale("log")

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    for i in range(samples.shape[0]):
        if "lognorm_ext" in box.parameters:
            dn_grains, r_width, radii = log_normal_distribution(
                10.0 ** log_r_g[i], sigma_g[i], 1000
            )

            # Exclude radii smaller than 1 nm
            indices = np.argwhere(radii >= 1e-3)

            dn_grains = dn_grains[indices]
            r_width = r_width[indices]
            radii = radii[indices]

        elif "powerlaw_ext" in box.parameters:
            dn_grains, r_width, radii = power_law_distribution(
                exponent[i], 1e-3, 10.0 ** r_max[i], 1000
            )

        ax.plot(radii, dn_grains / r_width, ls="-", lw=0.5, color="tab:gray", alpha=0.5)

    if output is None:
        plt.show()
    else:
        print(f"Output: {output}")
        plt.savefig(output, bbox_inches="tight")

    return fig


@typechecked
def plot_extinction(
    tag: str,
    random: Optional[int] = None,
    wavel_range: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    offset: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
) -> mpl.figure.Figure:
    """
    Function to plot random samples of the extinction.

    Parameters
    ----------
    tag : str
        Database tag with the samples.
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

    from species.data.database import Database

    species_db = Database()
    samples_box = species_db.get_samples(tag)
    samples = samples_box.samples

    print_section("Plot extinction")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    if random is not None:
        ran_index = np.random.randint(samples.shape[0], size=random)
        samples = samples[ran_index,]

    if wavel_range is None:
        wavel_range = (0.4, 10.0)

    print(f"Database tag: {tag}")
    print(f"Number of samples: {samples.shape[0]}")

    print(f"\nWavelength range: {wavel_range}")
    print(f"Label offset: {offset}")

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

    ax.set_xlabel("Wavelength (µm)", fontsize=12)
    ax.set_ylabel("Extinction (mag)", fontsize=12)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if offset is not None:
        ax.get_xaxis().set_label_coords(0.5, offset[0])
        ax.get_yaxis().set_label_coords(offset[1], 0.5)

    sample_wavel = np.linspace(wavel_range[0], wavel_range[1], 100)

    if "lognorm_ext" in samples_box.parameters:
        dust_interp, _, _ = interp_lognorm(verbose=False)

        log_r_index = samples_box.parameters.index("lognorm_radius")
        sigma_index = samples_box.parameters.index("lognorm_sigma")
        ext_index = samples_box.parameters.index("lognorm_ext")

        for i in range(samples.shape[0]):
            cross_sections = dust_interp(
                (sample_wavel, 10.0 ** samples[i, log_r_index], samples[i, sigma_index])
            )
            sample_ext = -2.5 * np.log10(
                np.exp(-samples[i, ext_index] * cross_sections)
            )

            ax.plot(sample_wavel, sample_ext, ls="-", lw=0.5, color="black", alpha=0.5)

    elif "powerlaw_ext" in samples_box.parameters:
        dust_interp, _, _ = interp_powerlaw(verbose=False)

        r_max_index = samples_box.parameters.index("powerlaw_max")
        exp_index = samples_box.parameters.index("powerlaw_exp")
        ext_index = samples_box.parameters.index("powerlaw_ext")

        for i in range(samples.shape[0]):
            cross_sections = dust_interp(
                (sample_wavel, 10.0 ** samples[i, r_max_index], samples[i, exp_index])
            )
            sample_ext = -2.5 * np.log10(
                np.exp(-samples[i, ext_index] * cross_sections)
            )

            ax.plot(sample_wavel, sample_ext, ls="-", lw=0.5, color="black", alpha=0.5)

    elif "ext_av" in samples_box.parameters:
        ext_index = samples_box.parameters.index("ext_av")
        ext_av = samples[:, ext_index]

        if "ext_rv" in samples_box.parameters:
            rv_index = samples_box.parameters.index("ext_rv")
            ext_rv = samples[:, rv_index]

        else:
            # Use default ISM redenning (R_V = 3.1) if ext_rv was not fitted
            ext_rv = np.full(samples.shape[0], 3.1)

        for i in range(samples.shape[0]):
            sample_ext = ism_extinction(ext_av[i], ext_rv[i], sample_wavel)

            ax.plot(sample_wavel, sample_ext, ls="-", lw=0.5, color="black", alpha=0.5)

    elif "ism_ext" in samples_box.parameters:
        ext_index = samples_box.parameters.index("ism_ext")
        ism_ext = samples[:, ext_index]

        if "ism_red" in samples_box.parameters:
            red_index = samples_box.parameters.index("ism_red")
            ism_red = samples[:, red_index]

        else:
            # Use default ISM redenning (R_V = 3.1) if ism_red was not fitted
            ism_red = np.full(samples.shape[0], 3.1)

        for i in range(samples.shape[0]):
            sample_ext = ism_extinction(ism_ext[i], ism_red[i], sample_wavel)

            ax.plot(sample_wavel, sample_ext, ls="-", lw=0.5, color="black", alpha=0.5)

    else:
        raise ValueError("The SamplesBox does not contain extinction parameters.")

    if output is None:
        plt.show()
    else:
        print(f"Output: {output}")
        plt.savefig(output, bbox_inches="tight")

    return fig
