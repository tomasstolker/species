"""
Module with utility functions for plotting data.
"""

import warnings

from string import ascii_lowercase
from typing import Dict, List, Optional, Tuple

import numpy as np

from typeguard import typechecked

from species.core import constants
from species.util.model_util import convert_model_name


@typechecked
def sptype_to_index(
    field_range: Tuple[str, str], spec_types: np.ndarray, check_subclass: bool
) -> np.ndarray:
    """
    Function for mapping the spectral types of stellar and
    substellar objects to indices that corresponds with the
    discrete colorbar of a color-magnitude or color-color
    diagram.

    Parameters
    ----------
    field_range : tuple(str, str)
        Range of the discrete colorbar for the field objects. The tuple
        should contain the lower and upper value ('early M', 'late M',
        'early L', 'late L', 'early T', 'late T', 'early Y). Also
        stellar spectral types can be specified.
    spec_types : np.ndarray
        Array with the spectral types.
    check_subclass : bool
        Set to ``True`` if the discrete colorbar should distinguish
        early and late spectral types with different colors or set
        to ``False`` if subclasses should not be distinguished.

    Returns
    -------
    np.ndarray
        Array with spectral types mapped to indices. Spectral types
        that are not within the range specified with ``field_range``
        will be set to NaN.
    """

    spt_discrete = np.zeros(spec_types.size)

    if check_subclass:
        spt_check = [
            ("early O", "late O"),
            ("early B", "late B"),
            ("early A", "late A"),
            ("early F", "late F"),
            ("early G", "late G"),
            ("early K", "late K"),
            ("early M", "late M"),
            ("early L", "late L"),
            ("early T", "late T"),
            ("early Y", "late Y"),
        ]

        for i, item in enumerate(spec_types):
            if item[0:2] in ["O0", "O1", "O2", "O3", "O4"]:
                spt_discrete[i] = 0.5

            elif item[0:2] in ["O5", "O6", "O7", "O8", "O9"]:
                spt_discrete[i] = 1.5

            elif item[0:2] in ["B0", "B1", "B2", "B3", "B4"]:
                spt_discrete[i] = 2.5

            elif item[0:2] in ["B5", "B6", "B7", "B8", "B9"]:
                spt_discrete[i] = 3.5

            elif item[0:2] in ["A0", "A1", "A2", "A3", "A4"]:
                spt_discrete[i] = 4.5

            elif item[0:2] in ["A5", "A6", "A7", "A8", "A9"]:
                spt_discrete[i] = 5.5

            elif item[0:2] in ["F0", "F1", "F2", "F3", "F4"]:
                spt_discrete[i] = 6.5

            elif item[0:2] in ["F5", "F6", "F7", "F8", "F9"]:
                spt_discrete[i] = 7.5

            elif item[0:2] in ["G0", "G1", "G2", "G3", "G4"]:
                spt_discrete[i] = 8.5

            elif item[0:2] in ["G5", "G6", "G7", "G8", "G9"]:
                spt_discrete[i] = 9.5

            elif item[0:2] in ["K0", "K1", "K2", "K3", "K4"]:
                spt_discrete[i] = 10.5

            elif item[0:2] in ["K5", "K6", "K7", "K8", "K9"]:
                spt_discrete[i] = 11.5

            elif item[0:2] in ["M0", "M1", "M2", "M3", "M4"]:
                spt_discrete[i] = 12.5

            elif item[0:2] in ["M5", "M6", "M7", "M8", "M9"]:
                spt_discrete[i] = 13.5

            elif item[0:2] in ["L0", "L1", "L2", "L3", "L4"]:
                spt_discrete[i] = 14.5

            elif item[0:2] in ["L5", "L6", "L7", "L8", "L9"]:
                spt_discrete[i] = 15.5

            elif item[0:2] in ["T0", "T1", "T2", "T3", "T4"]:
                spt_discrete[i] = 16.5

            elif item[0:2] in ["T5", "T6", "T7", "T8", "T9"]:
                spt_discrete[i] = 17.5

            elif "Y" in item:
                spt_discrete[i] = 18.5

            else:
                spt_discrete[i] = np.nan

        count = 0
        for i, item in enumerate(spt_check):
            for j in range(2):
                if field_range[0] == item[j]:
                    spt_discrete -= float(count)
                    break
                count += 1

    else:
        spt_check = ["O", "B", "A", "F", "G", "K", "M", "L", "T", "Y"]

        for i, item in enumerate(spec_types):
            if item[0] == "O":
                spt_discrete[i] = 0.5

            elif item[0] == "B":
                spt_discrete[i] = 1.5

            elif item[0] == "A":
                spt_discrete[i] = 2.5

            elif item[0] == "F":
                spt_discrete[i] = 3.5

            elif item[0] == "G":
                spt_discrete[i] = 4.5

            elif item[0] == "K":
                spt_discrete[i] = 5.5

            elif item[0] == "M":
                spt_discrete[i] = 6.5

            elif item[0] == "L":
                spt_discrete[i] = 7.5

            elif item[0] == "T":
                spt_discrete[i] = 8.5

            elif item[0] == "Y":
                spt_discrete[i] = 9.5

            else:
                spt_discrete[i] = np.nan

        for i, item in enumerate(spt_check):
            if field_range[0] == item:
                spt_discrete -= float(i)
                break

    set_to_nan = spt_discrete < 0.0
    spt_discrete[set_to_nan] = np.nan

    return spt_discrete


@typechecked
def update_labels(param: List[str], object_type: str = "planet") -> List[str]:
    """
    Function for formatting the model parameters to use them
    as labels in the posterior plot.

    Parameters
    ----------
    param : list
        List with names of the model parameters.
    object_type : str
        Object type ('planet' or 'star'). With 'planet', the radius
        and mass are expressed in Jupiter units. With 'star', the
        radius and mass are expressed in solar units.

    Returns
    -------
    list
        List with parameter labels for plots.
    """

    cloud_species = ["fe", "mgsio3", "al2o3", "na2s", "kcl"]

    cloud_labels = ["Fe", r"MgSiO_{3}", r"Al_{2}O_{3}", r"Na_{2}S", "KCl"]

    abund_species = [
        "CO_all_iso",
        "CO_all_iso_HITEMP",
        "H2O",
        "H2O_HITEMP",
        "H2O_main_iso",
        "CH4",
        "CH4_main_iso",
        "NH3",
        "NH3_main_iso",
        "CO2",
        "CO2_main_iso",
        "H2S",
        "H2S_main_iso",
        "Na",
        "Na_allard",
        "Na_burrows",
        "Na_lor_cur",
        "K",
        "K_allard",
        "K_burrows",
        "K_lor_cur",
        "PH3",
        "PH3_main_iso",
        "VO",
        "VO_Plez",
        "TiO",
        "TiO_all_Exomol",
        "TiO_all_iso_Plez",
        "FeH",
        "FeH_main_iso",
        "MgSiO3(c)",
        "Fe(c)",
        "Al2O3(c)",
        "Na2S(c)",
        "KCL(c)",
    ]

    abund_labels = [
        "CO",
        "CO",
        "H_{2}O",
        "H_{2}O",
        "H_{2}O",
        "CH_{4}",
        "CH_{4}",
        "NH_{3}",
        "NH_{3}",
        "CO_{2}",
        "CO_{2}",
        "H_{2}S",
        "H_{2}S",
        "Na",
        "Na",
        "Na",
        "Na",
        "K",
        "K",
        "K",
        "K",
        "PH_{3}",
        "PH_{3}",
        "VO",
        "VO",
        "TiO",
        "TiO",
        "TiO",
        "FeH",
        "FeH",
        "MgSiO_{3}",
        "Fe",
        "Al_{2}O_{3}",
        "Na_{2}S",
        "KCl",
    ]

    if "teff" in param:
        index = param.index("teff")
        param[index] = r"$T_\mathrm{eff}$ (K)"

    if "teff_0" in param:
        index = param.index("teff_0")
        param[index] = r"$T_\mathrm{eff,1}$ (K)"

    if "teff_1" in param:
        index = param.index("teff_1")
        param[index] = r"$T_\mathrm{eff,2}$ (K)"

    if "logg" in param:
        index = param.index("logg")
        param[index] = r"$\log\,g$"

    if "logg_0" in param:
        index = param.index("logg_0")
        param[index] = r"$\log\,g_\mathrm{1}$"

    if "logg_1" in param:
        index = param.index("logg_1")
        param[index] = r"$\log\,g_\mathrm{2}$"

    if "metallicity" in param:
        index = param.index("metallicity")
        param[index] = "[Fe/H]"

    if "feh" in param:
        index = param.index("feh")
        param[index] = "[Fe/H]"

    if "feh_0" in param:
        index = param.index("feh_0")
        param[index] = r"[Fe/H]$_\mathrm{1}$"

    if "feh_1" in param:
        index = param.index("feh_1")
        param[index] = r"[Fe/H]$_\mathrm{2}$"

    if "fsed" in param:
        index = param.index("fsed")
        param[index] = r"$f_\mathrm{sed}$"

    if "fsed_1" in param:
        index = param.index("fsed_1")
        param[index] = r"$f_\mathrm{sed,1}$"

    if "fsed_2" in param:
        index = param.index("fsed_2")
        param[index] = r"$f_\mathrm{sed,2}$"

    if "f_clouds" in param:
        index = param.index("f_clouds")
        param[index] = r"$w_\mathrm{clouds}$"

    if "c_o_ratio" in param:
        index = param.index("c_o_ratio")
        param[index] = r"C/O"

    if "radius" in param:
        index = param.index("radius")
        if object_type == "planet":
            param[index] = r"$R$ ($R_\mathrm{J}$)"
        elif object_type == "star":
            param[index] = r"$R_\ast$ ($R_\mathrm{\odot}$)"

    if "distance" in param:
        index = param.index("distance")
        param[index] = "$d$ (pc)"

    if "parallax" in param:
        index = param.index("parallax")
        param[index] = r"$\varpi$ (mas)"

    if "vsini" in param:
        index = param.index("vsini")
        param[index] = r"$v\,\sin\,i$ (km s$^{-1}$)"

    if "mass" in param:
        index = param.index("mass")
        if object_type == "planet":
            param[index] = r"$M$ ($M_\mathrm{J}$)"
        elif object_type == "star":
            param[index] = r"$M_\ast$ ($M_\mathrm{\odot}$)"

    for i, item in enumerate(ascii_lowercase[1:]):
        if f"mass_{i}" in param:
            index = param.index(f"mass_{i}")
            param[index] = rf"$M_\mathrm{{{item}}}$ ($M_\mathrm{{J}}$)"
        else:
            break

    if "log_mass" in param:
        index = param.index("log_mass")
        if object_type == "planet":
            param[index] = r"$\log\,M/M_\mathrm{J}$"
        elif object_type == "star":
            param[index] = r"$\log\,M_\ast/M_\mathrm{\odot}$"

    if "age" in param:
        index = param.index("age")
        param[index] = "Age (Myr)"

    if "mass_1" in param:
        index = param.index("mass_1")
        param[index] = r"$M_\mathrm{b}$ ($M_\mathrm{J}$)"

    if "mass_2" in param:
        index = param.index("mass_2")
        param[index] = r"$M_\mathrm{c}$ ($M_\mathrm{J}$)"

    if "entropy" in param:
        index = param.index("entropy")
        param[index] = r"$S_\mathrm{i}$ ($k_\mathrm{B}/\mathrm{baryon}$)"

    if "entropy_1" in param:
        index = param.index("entropy_1")
        param[index] = r"$S_\mathrm{i,b}$ ($k_\mathrm{B}/\mathrm{baryon}$)"

    if "entropy_2" in param:
        index = param.index("entropy_2")
        param[index] = r"$S_\mathrm{i,c}$ ($k_\mathrm{B}/\mathrm{baryon}$)"

    if "dfrac_1" in param:
        index = param.index("dfrac_1")
        param[index] = r"$\log\,D_\mathrm{i,b}$"

    if "dfrac_2" in param:
        index = param.index("dfrac_2")
        param[index] = r"$\log\,D_\mathrm{i,c}$"

    if "y_frac" in param:
        index = param.index("y_frac")
        param[index] = r"$Y$"

    if "yfrac_1" in param:
        index = param.index("yfrac_1")
        param[index] = r"$Y_\mathrm{b}$"

    if "yfrac_2" in param:
        index = param.index("yfrac_2")
        param[index] = r"$Y_\mathrm{c}$"

    if "mcore_1" in param:
        index = param.index("mcore_1")
        param[index] = r"$M_\mathrm{core,b}$ ($M_\mathrm{E}$)"

    if "mcore_2" in param:
        index = param.index("mcore_2")
        param[index] = r"$M_\mathrm{core,c}$ ($M_\mathrm{E}$)"

    for i, item in enumerate(ascii_lowercase[1:]):
        if f"teff_evol_{i}" in param:
            index = param.index(f"teff_evol_{i}")
            param[index] = rf"$T_\mathrm{{eff, {item}}}$ (K)"
        else:
            break

    for i, item in enumerate(ascii_lowercase[1:]):
        if f"radius_evol_{i}" in param:
            index = param.index(f"radius_evol_{i}")
            param[index] = rf"$R_\mathrm{{{item}}}$ ($R_\mathrm{{J}}$)"
        else:
            break

    for i, item in enumerate(ascii_lowercase[1:]):
        if f"logg_evol_{i}" in param:
            index = param.index(f"logg_evol_{i}")
            param[index] = rf"$\log\,g_\mathrm{{{item}}}$"
        else:
            break

    for i, item in enumerate(ascii_lowercase[1:]):
        if f"inflate_lbol{i}" in param:
            index = param.index(f"inflate_lbol{i}")
            param[index] = rf"$\sigma_{{L,{{{item}}}}}$ (dex)"
        else:
            break

    for i, item in enumerate(ascii_lowercase[1:]):
        if f"inflate_mass{i}" in param:
            index = param.index(f"inflate_mass{i}")
            param[index] = rf"$\sigma_{{M,{{{item}}}}}$ ($M_\mathrm{{J}}$)"
        else:
            break

    if "luminosity" in param:
        index = param.index("luminosity")
        if object_type == "planet":
            param[index] = r"$\log\,L/L_\mathrm{\odot}$"
        elif object_type == "star":
            param[index] = r"$\log\,L_\ast/L_\mathrm{\odot}$"

    if "luminosity_ratio" in param:
        index = param.index("luminosity_ratio")
        param[index] = r"$\log\,L_\mathrm{1}/L_\mathrm{2}$"

    if "luminosity_disk_planet" in param:
        index = param.index("luminosity_disk_planet")
        param[index] = r"$L_\mathrm{disk}/L_\mathrm{atm}$"

    if "lognorm_radius" in param:
        index = param.index("lognorm_radius")
        param[index] = r"$\log\,r_\mathrm{g}$"

    if "lognorm_sigma" in param:
        index = param.index("lognorm_sigma")
        param[index] = r"$\sigma_\mathrm{g}$"

    if "lognorm_ext" in param:
        index = param.index("lognorm_ext")
        param[index] = r"$A_V$"

    if "powerlaw_min" in param:
        index = param.index("powerlaw_min")
        param[index] = r"$\log\,a_\mathrm{min}/\mathrm{µm}$"

    if "powerlaw_max" in param:
        index = param.index("powerlaw_max")
        param[index] = r"$\log\,a_\mathrm{max}/\mathrm{µm}$"

    if "powerlaw_exp" in param:
        index = param.index("powerlaw_exp")
        param[index] = r"$\beta$"

    if "powerlaw_ext" in param:
        index = param.index("powerlaw_ext")
        param[index] = r"$A_V$"

    if "ism_ext" in param:
        index = param.index("ism_ext")
        param[index] = r"$A_V$"

    if "ism_red" in param:
        index = param.index("ism_red")
        param[index] = r"$R_V$"

    for item in param:
        if item.startswith("phot_ext_"):
            index = param.index(item)
            filter_name = item[9:].split("/")[1]
            param[index] = rf"$A_\mathrm{{{filter_name}}}$"
            break

    if "tint" in param:
        index = param.index("tint")
        param[index] = r"$T_\mathrm{int}$ (K)"

    for i in range(15):
        if f"t{i}" in param:
            index = param.index(f"t{i}")
            param[index] = rf"$T_\mathrm{{{i}}}$ (K)"

    if "alpha" in param:
        index = param.index("alpha")
        param[index] = r"$\alpha$"

    if "log_sigma_alpha" in param:
        index = param.index("log_sigma_alpha")
        param[index] = r"$\log\,\sigma_\alpha$"

    if "log_delta" in param:
        index = param.index("log_delta")
        param[index] = r"$\log\,\delta$"

    if "T_bottom" in param:
        index = param.index("T_bottom")
        param[index] = r"$T_\mathrm{bottom}$ (K)"

    num_layer = 6  # could make a variable in the future
    for i in range(num_layer):
        if f"PTslope_{num_layer - i}" in param:
            index = param.index(f"PTslope_{num_layer - i}")
            param[index] = rf"$(dlnT/dlnP)_\mathrm{{P={i}bar}}$ (K)"

    if "log_p_quench" in param:
        index = param.index("log_p_quench")
        param[index] = r"$\log\,P_\mathrm{quench}$"

    if "sigma_lnorm" in param:
        index = param.index("sigma_lnorm")
        param[index] = r"$\sigma_\mathrm{g}$"

    if "log_kzz" in param:
        index = param.index("log_kzz")
        param[index] = r"$\log\,K_\mathrm{zz}$"

    if "kzz" in param:
        # Backward compatibility
        index = param.index("kzz")
        param[index] = r"$\log\,K_\mathrm{zz}$"

    for i, item in enumerate(cloud_species):
        if f"{item}_fraction" in param:
            index = param.index(f"{item}_fraction")
            param[
                index
            ] = rf"$\log\,\tilde{{\mathrm{{X}}}}_\mathrm{{{cloud_labels[i]}}}$"

        if f"{item}_tau" in param:
            index = param.index(f"{item}_tau")
            param[index] = rf"$\bar{{\tau}}_\mathrm{{{cloud_labels[i]}}}$"

        if f"log_p_base_{item}" in param:
            index = param.index(f"log_p_base_{item}")
            param[index] = rf"$\log\,P_\mathrm{{{cloud_labels[i]}}}$"

    for i, item_i in enumerate(cloud_species):
        for j, item_j in enumerate(cloud_species):
            if f"{item_i}_{item_j}_ratio" in param:
                index = param.index(f"{item_i}_{item_j}_ratio")
                param[index] = (
                    rf"$\log\,\tilde{{\mathrm{{X}}}}"
                    rf"_\mathrm{{{cloud_labels[i]}}}/"
                    rf"\mathrm{{\tilde{{X}}}}_\mathrm{{{cloud_labels[j]}}}$"
                )

    for i, item in enumerate(abund_species):
        if item in param:
            index = param.index(item)
            param[index] = rf"$\log\,\mathrm{{{abund_labels[i]}}}$"

    for i, item in enumerate(param):
        if item[0:8] == "scaling_":
            item_name = item[8:]
            if item_name.find("\\_") == -1 and item_name.find("_") > 0:
                item_name = item_name.replace("_", "\\_")
            param[i] = rf"$a_\mathrm{{{item_name}}}$"

        elif item[0:6] == "error_":
            item_name = item[6:]
            if item_name.find("\\_") == -1 and item_name.find("_") > 0:
                item_name = item_name.replace("_", "\\_")
            param[i] = rf"$b_\mathrm{{{item_name}}}$"

        elif item[0:7] == "radvel_":
            item_name = item[7:]
            if item_name.find("\\_") == -1 and item_name.find("_") > 0:
                item_name = item_name.replace("_", "\\_")
            param[i] = rf"RV$_\mathrm{{{item_name}}}$ (km s$^{{-1}}$)"

        elif item[0:11] == "wavelength_":
            item_name = item[11:]
            if item_name.find("\\_") == -1 and item_name.find("_") > 0:
                item_name = item_name.replace("_", "\\_")
            param[i] = rf"$c_\mathrm{{{item_name}}}$ (nm)"

        elif item[-6:] == "_error":
            item_name = item[:-6]
            if item_name.find("\\_") == -1 and item_name.find("_") > 0:
                item_name = item_name.replace("_", "\\_")
            param[i] = rf"$f_\mathrm{{{item_name}}}$"

        elif item[0:9] == "corr_len_":
            item_name = item[9:]
            if item_name.find("\\_") == -1 and item_name.find("_") > 0:
                item_name = item_name.replace("_", "\\_")
            param[i] = rf"$\log\,\ell_\mathrm{{{item_name}}}$"

        elif item[0:9] == "corr_amp_":
            item_name = item[9:]
            if item_name.find("\\_") == -1 and item_name.find("_") > 0:
                item_name = item_name.replace("_", "\\_")
            param[i] = rf"$f_\mathrm{{{item_name}}}$"

    if "c_h_ratio" in param:
        index = param.index("c_h_ratio")
        param[index] = r"[C/H]"

    if "o_h_ratio" in param:
        index = param.index("o_h_ratio")
        param[index] = r"[O/H]"

    for i in range(100):
        if f"teff_{i}" in param:
            index = param.index(f"teff_{i}")
            param[index] = rf"$T_\mathrm{{{i+1}}}$ (K)"

        else:
            break

    for i in range(100):
        if f"radius_{i}" in param:
            index = param.index(f"radius_{i}")
            param[index] = rf"$R_\mathrm{{{i+1}}}$ ($R_\mathrm{{J}}$)"

        else:
            break

    for i in range(100):
        if f"luminosity_{i}" in param:
            index = param.index(f"luminosity_{i}")
            param[index] = rf"$\log\,L_\mathregular{{{i+1}}}/L_\mathregular{{\odot}}$"

        else:
            break

    if "disk_teff" in param:
        index = param.index("disk_teff")
        param[index] = r"$T_\mathrm{disk}$ (K)"

    if "disk_radius" in param:
        index = param.index("disk_radius")
        param[index] = r"$R_\mathrm{disk}$ ($R_\mathrm{J}$)"

    if "log_powerlaw_a" in param:
        index = param.index("log_powerlaw_a")
        param[index] = r"$a_\mathrm{powerlaw}$"

    if "log_powerlaw_b" in param:
        index = param.index("log_powerlaw_b")
        param[index] = r"$b_\mathrm{powerlaw}$"

    if "log_powerlaw_c" in param:
        index = param.index("log_powerlaw_c")
        param[index] = r"$c_\mathrm{powerlaw}$"

    if "pt_smooth" in param:
        index = param.index("pt_smooth")
        param[index] = r"$\sigma_\mathrm{P-T}$"

    if "abund_smooth" in param:
        index = param.index("abund_smooth")
        param[index] = r"$\sigma_\mathrm{abund}$"

    if "log_prob" in param:
        index = param.index("log_prob")
        param[index] = r"$\log\,\mathcal{L}$"

    if "log_tau_cloud" in param:
        index = param.index("log_tau_cloud")
        param[index] = r"$\log\,\tau_\mathrm{cloud}$"

    if "veil_a" in param:
        index = param.index("veil_a")
        param[index] = r"$a_\mathrm{veil}$"

    if "veil_b" in param:
        index = param.index("veil_b")
        param[index] = r"$b_\mathrm{veil}$"

    if "veil_ref" in param:
        index = param.index("veil_ref")
        param[index] = r"$F_\mathrm{ref, veil}$"

    if "gauss_amplitude" in param:
        index = param.index("gauss_amplitude")
        param[index] = r"$a$ (W m$^{-2}$ µm$^{-1}$)"

    if "gauss_mean" in param:
        index = param.index("gauss_mean")
        param[index] = r"$\lambda$ (nm)"

    if "gauss_sigma" in param:
        index = param.index("gauss_sigma")
        param[index] = r"$\sigma$ (nm)"

    if "gauss_amplitude_2" in param:
        index = param.index("gauss_amplitude_2")
        param[index] = r"$a_2$ (W m$^{-2}$ µm$^{-1}$)"

    if "gauss_mean_2" in param:
        index = param.index("gauss_mean_2")
        param[index] = r"$\lambda_2$ (nm)"

    if "gauss_sigma_2" in param:
        index = param.index("gauss_sigma_2")
        param[index] = r"$\sigma_2$ (nm)"

    if "gauss_fwhm" in param:
        index = param.index("gauss_fwhm")
        param[index] = r"FWHM (km s$^{-1}$)"

    if "line_flux" in param:
        index = param.index("line_flux")
        param[index] = r"$F_\mathrm{line}$ (W m$^{-2}$)"

    if "line_luminosity" in param:
        index = param.index("line_luminosity")
        param[index] = r"$L_\mathrm{line}$ ($L_\mathrm{\odot}$)"

    if "log_line_lum" in param:
        index = param.index("log_line_lum")
        param[index] = r"$\log\,L_\mathrm{line}/L_\mathrm{\odot}$"

    if "log_acc_lum" in param:
        index = param.index("log_acc_lum")
        param[index] = r"$\log\,L_\mathrm{acc}/L_\mathrm{\odot}$"

    if "line_eq_width" in param:
        index = param.index("line_eq_width")
        param[index] = r"EW ($\AA$)"

    if "line_vrad" in param:
        index = param.index("line_vrad")
        param[index] = r"RV (km s$^{-1}$)"

    if "log_kappa_0" in param:
        index = param.index("log_kappa_0")
        param[index] = r"$\log\,\kappa_0$"

    if "log_kappa_abs" in param:
        index = param.index("log_kappa_abs")
        param[index] = r"$\log\,\kappa_\mathrm{abs}$"

    if "log_kappa_sca" in param:
        index = param.index("log_kappa_sca")
        param[index] = r"$\log\,\kappa_\mathrm{sca}$"

    if "opa_index" in param:
        index = param.index("opa_index")
        param[index] = r"$\xi$"

    if "opa_abs_index" in param:
        index = param.index("opa_abs_index")
        param[index] = r"$\xi_\mathrm{abs}$"

    if "opa_sca_index" in param:
        index = param.index("opa_sca_index")
        param[index] = r"$\xi_\mathrm{sca}$"

    if "log_p_base" in param:
        index = param.index("log_p_base")
        param[index] = r"$\log\,P_\mathrm{cloud}$"

    if "albedo" in param:
        index = param.index("albedo")
        param[index] = r"$\omega$"

    if "opa_knee" in param:
        index = param.index("opa_knee")
        param[index] = r"$\lambda_\mathrm{R}$ (µm)"

    if "lambda_ray" in param:
        index = param.index("lambda_ray")
        param[index] = r"$\lambda_\mathrm{R}$ (µm)"

    if "mix_length" in param:
        index = param.index("mix_length")
        param[index] = r"$\ell_\mathrm{m}$ ($H_\mathrm{p}$)"

    if "spec_weight" in param:
        index = param.index("spec_weight")
        param[index] = r"w$_\mathrm{spec}$"

    if "log_beta_r" in param:
        index = param.index("log_beta_r")
        param[index] = r"$\log\,\beta_\mathrm{r}$"

    if "log_gamma_r" in param:
        index = param.index("log_gamma_r")
        param[index] = r"$\log\,\gamma_\mathrm{r}$"

    if "gamma_r" in param:
        index = param.index("gamma_r")
        param[index] = r"$\gamma_\mathrm{r}$"

    if "log_kappa_gray" in param:
        index = param.index("log_kappa_gray")
        param[index] = r"$\log\,\kappa_\mathrm{gray}$"

    if "log_cloud_top" in param:
        index = param.index("log_cloud_top")
        param[index] = r"$\log\,P_\mathrm{top}$"

    if "flux_scaling" in param:
        index = param.index("flux_scaling")
        param[index] = r"$a_\mathrm{flux}$"

    if "log_flux_scaling" in param:
        index = param.index("log_flux_scaling")
        param[index] = r"$\log\,a_\mathrm{flux}$"

    if "flux_offset" in param:
        index = param.index("flux_offset")
        param[index] = r"$b_\mathrm{flux}$ (W m$^{-2}$ µm$^{-1}$)"

    return param


@typechecked
def quantity_unit(
    param: List[str], object_type: str
) -> Tuple[List[str], List[Optional[str]], List[str]]:
    """
    Function for creating lists with quantities, units, and labels
    for fitted parameter.

    Parameters
    ----------
    param : list
        List with parameter names.
    object_type : str
        Object type (``'planet'`` or ``'star'``).

    Returns
    -------
    list
        List with the quantities.
    list
        List with the units.
    list
        List with the parameter labels for plots.
    """

    quantity = []
    unit = []
    label = []

    for item in param:
        if item == "teff":
            quantity.append("teff")
            unit.append("K")
            label.append(r"$T_\mathrm{eff}$")

        if item == "logg":
            quantity.append("logg")
            unit.append(None)
            label.append(r"$\log g$")

        if item == "metallicity":
            quantity.append("metallicity")
            unit.append(None)
            label.append("[Fe/H]")

        if item == "feh":
            quantity.append("feh")
            unit.append(None)
            label.append("[Fe/H]")

        if item == "fsed":
            quantity.append("fsed")
            unit.append(None)
            label.append(r"$f_\mathrm{sed}$")

        if item == "c_o_ratio":
            quantity.append("c_o_ratio")
            unit.append(None)
            label.append("C/O")

        if item == "ad_index":
            quantity.append("ad_index")
            unit.append(None)
            label.append(r"$\gamma_\mathrm{ad}$")

        if item == "radius":
            quantity.append("radius")

            if object_type == "planet":
                unit.append(r"$R_\mathrm{J}$")
                label.append(r"$R$")

            elif object_type == "star":
                unit.append(r"$R_\mathrm{\odot}$")
                label.append(r"$R_\ast$")

        for i in range(100):
            if item == f"teff_{i}":
                quantity.append(f"teff_{i}")
                unit.append("K")
                label.append(rf"$T_\mathrm{{{i+1}}}$")

            else:
                break

        for i in range(100):
            if item == f"radius_{i}":
                quantity.append(f"radius_{i}")

                if object_type == "planet":
                    unit.append(r"$R_\mathrm{J}$")

                elif object_type == "star":
                    unit.append(r"$R_\mathrm{\odot}$")

                label.append(rf"$R_\mathrm{{{i+1}}}$")

            else:
                break

        if item == "distance":
            quantity.append("distance")
            unit.append("pc")
            label.append(r"$d$")

        if item == "parallax":
            quantity.append("parallax")
            unit.append("mas")
            label.append(r"$\varpi$")

        if item == "mass":
            quantity.append("mass")

            if object_type == "planet":
                unit.append(r"$M_\mathrm{J}$")
                label.append(r"$M$")

            elif object_type == "star":
                unit.append(r"$M_\mathrm{\odot}$")
                label.append(r"$M_\ast$")

        if item == "luminosity":
            quantity.append("luminosity")
            unit.append(None)
            label.append(r"$\log\,L/L_\mathrm{\odot}$")

        if item == "ism_ext":
            quantity.append("ism_ext")
            unit.append(None)
            label.append(r"$A_V$")

        if item == "lognorm_ext":
            quantity.append("lognorm_ext")
            unit.append(None)
            label.append(r"$A_V$")

        if item == "powerlaw_ext":
            quantity.append("powerlaw_ext")
            unit.append(None)
            label.append(r"$A_V$")

        if item.startswith("phot_ext_"):
            quantity.append(item)
            unit.append(None)
            filter_name = item[9:].split("/")[1]
            label.append(rf"$A_\mathrm{{{filter_name}}}$")

        if item == "pt_smooth":
            quantity.append("pt_smooth")
            unit.append(None)
            label.append(r"$\sigma_\mathrm{P-T}$")

        if item == "abund_smooth":
            quantity.append("abund_smooth")
            unit.append(None)
            label.append(r"$\sigma_\mathrm{abund}$")

        if item == "disk_teff":
            quantity.append("disk_teff")
            unit.append("K")
            label.append(r"$T_\mathrm{disk}$")

        if item == "disk_radius":
            quantity.append("disk_radius")
            unit.append(r"$R_\mathrm{J}$")
            label.append(r"$R_\mathrm{disk}$")

        if item == "flux_scaling":
            quantity.append("flux_scaling")
            unit.append(None)
            label.append(r"$a_\mathrm{flux}$")

        if item == "log_flux_scaling":
            quantity.append("log_flux_scaling")
            unit.append(None)
            label.append(r"$\log\,a_\mathrm{flux}$")

        if item == "flux_offset":
            quantity.append("flux_offset")
            unit.append(r"W m$^{-2}$ µm$^{-1}$")
            label.append(r"$b_\mathrm{flux}$")

    return quantity, unit, label


@typechecked
def field_bounds_ticks(
    field_range: Tuple[str, str],
    check_subclass: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Function for converting the specified field range into boundaries
    and labels for the discrete colorbar that is plotted with a
    color-magnitude or color-color diagram.

    Parameters
    ----------
    field_range : tuple(str, str)
        Range of the discrete colorbar for the field objects. The tuple
        should contain the lower and upper value ('early M', 'late M',
        'early L', 'late L', 'early T', 'late T', 'early Y). Also
        stellar spectral types can be specified.
    check_subclass : bool
        Set to ``True`` if the discrete colorbar should distinguish
        early and late spectral types with different colors or set
        to ``False`` if subclasses should not be distinguished.

    Returns
    -------
    np.ndarray
        Array with the boundaries for the discrete colorbar.
    np.ndarray
        Array with the midpoints for the discrete colorbar.
    list(str)
        List with the tick labels for the discrete colorbar.
    """

    if check_subclass:
        spectral_ranges = [
            "O0-O4",
            "O5-O9",
            "B0-B4",
            "B5-B9",
            "A0-A4",
            "A5-A9",
            "F0-F4",
            "F5-F9",
            "G0-G4",
            "G5-G9",
            "K0-K4",
            "K5-K9",
            "M0-M4",
            "M5-M9",
            "L0-L4",
            "L5-L9",
            "T0-T4",
            "T5-T9",
            "Y1-Y2",
        ]

        if field_range[0] == "early O":
            index_start = 0
        elif field_range[0] == "late O":
            index_start = 1
        elif field_range[0] == "early B":
            index_start = 2
        elif field_range[0] == "late B":
            index_start = 3
        elif field_range[0] == "early A":
            index_start = 4
        elif field_range[0] == "late A":
            index_start = 5
        elif field_range[0] == "early F":
            index_start = 6
        elif field_range[0] == "late F":
            index_start = 7
        elif field_range[0] == "early G":
            index_start = 8
        elif field_range[0] == "late G":
            index_start = 9
        elif field_range[0] == "early K":
            index_start = 10
        elif field_range[0] == "late K":
            index_start = 11
        elif field_range[0] == "early M":
            index_start = 12
        elif field_range[0] == "late M":
            index_start = 13
        elif field_range[0] == "early L":
            index_start = 14
        elif field_range[0] == "late L":
            index_start = 15
        elif field_range[0] == "early T":
            index_start = 16
        elif field_range[0] == "late T":
            index_start = 17
        elif field_range[0] == "early Y":
            index_start = 18

        if field_range[1] == "early O":
            index_end = 1
        elif field_range[1] == "late O":
            index_end = 2
        elif field_range[1] == "early B":
            index_end = 3
        elif field_range[1] == "late B":
            index_end = 4
        elif field_range[1] == "early A":
            index_end = 5
        elif field_range[1] == "late A":
            index_end = 6
        elif field_range[1] == "early F":
            index_end = 7
        elif field_range[1] == "late F":
            index_end = 8
        elif field_range[1] == "early G":
            index_end = 9
        elif field_range[1] == "late G":
            index_end = 10
        elif field_range[1] == "early K":
            index_end = 11
        elif field_range[1] == "late K":
            index_end = 12
        elif field_range[1] == "early M":
            index_end = 13
        elif field_range[1] == "late M":
            index_end = 14
        elif field_range[1] == "early L":
            index_end = 15
        elif field_range[1] == "late L":
            index_end = 16
        elif field_range[1] == "early T":
            index_end = 17
        elif field_range[1] == "late T":
            index_end = 18
        elif field_range[1] == "early Y":
            index_end = 19

    else:
        spectral_ranges = ["O", "B", "A", "F", "G", "K", "M", "L", "T", "Y"]

        if field_range[0] == "O":
            index_start = 0
        elif field_range[0] == "B":
            index_start = 1
        elif field_range[0] == "A":
            index_start = 2
        elif field_range[0] == "F":
            index_start = 3
        elif field_range[0] == "G":
            index_start = 4
        elif field_range[0] == "K":
            index_start = 5
        elif field_range[0] == "M":
            index_start = 6
        elif field_range[0] == "L":
            index_start = 7
        elif field_range[0] == "T":
            index_start = 8
        elif field_range[0] == "Y":
            index_start = 9

        if field_range[1] == "O":
            index_end = 1
        elif field_range[1] == "B":
            index_end = 2
        elif field_range[1] == "A":
            index_end = 3
        elif field_range[1] == "F":
            index_end = 4
        elif field_range[1] == "G":
            index_end = 5
        elif field_range[1] == "K":
            index_end = 6
        elif field_range[1] == "M":
            index_end = 7
        elif field_range[1] == "L":
            index_end = 8
        elif field_range[1] == "T":
            index_end = 9
        elif field_range[1] == "Y":
            index_end = 10

    index_range = index_end - index_start + 1

    bounds = np.linspace(index_start, index_end, index_range)
    ticks = np.linspace(index_start + 0.5, index_end - 0.5, index_range - 1)

    labels = spectral_ranges[index_start:index_end]

    ticks -= bounds[0]
    bounds -= bounds[0]

    return bounds, ticks, labels


@typechecked
def remove_color_duplicates(
    object_names: List[str], empirical_names: np.ndarray
) -> List[int]:
    """ "
    Function for deselecting young/low-gravity objects that will
    already be plotted individually as directly imaged objects.

    Parameters
    ----------
    object_names : list(str)
        List with names of directly imaged planets and brown dwarfs.
    empirical_names : np.ndarray
        Array with names of young/low-gravity objects.

    Returns
    -------
    list
        List with selected indices of the young/low-gravity objects.
    """

    indices = []

    for i, item in enumerate(empirical_names):
        if item == "beta_Pic_b" and "beta Pic b" in object_names:
            continue

        if item == "HR8799b" and "HR 8799 b" in object_names:
            continue

        if item == "HR8799c" and "HR 8799 c" in object_names:
            continue

        if item == "HR8799d" and "HR 8799 d" in object_names:
            continue

        if item == "HR8799e" and "HR 8799 e" in object_names:
            continue

        if item == "kappa_And_B" and "kappa And b" in object_names:
            continue

        if item == "HD1160B" and "HD 1160 B" in object_names:
            continue

        indices.append(i)

    return indices


@typechecked
def create_model_label(
    model_param: Dict[str, float],
    object_type: str,
    model_name: str,
    inc_model_name: bool,
    leg_param: List[str],
    param_fmt: Dict[str, str],
) -> str:
    """ "
    Function for creating a label that includes the parameters of a
    model spectrum. The label is used in the legend of a SED plot.

    Parameters
    ----------
    model_param : dict
        Dictionary with model parameters.
    model_name : str
        Name of the atmospheric model.
    inc_model_name : bool
        Include the model name in the label.
    object_type : str
        Object type ('planet' or 'star') that determines if
        Jupiter or solar units are used.
    leg_param : list(str)
        List with the parameters to include. Apart from atmospheric
        parameters (e.g. 'teff', 'logg', 'radius') also parameters
        such as 'mass' and 'luminosity' can be included. The default
        atmospheric parameters are included in the legend if the
        argument is an empty list.
    param_fmt : dict(str, str)
        Dictionary with formats that will be used for the model
        parameter. The parameters are included in the ``legend``
        when plotting the model spectra. Default formats are
        used if the argument of ``param_fmt`` is set to ``None``.

    Returns
    -------
    str
        Text label that includes the selected parameter names,
        the model values, and the parameter units.
    """

    # data_file = Path(__file__).parents[1].resolve() / "data/model_data/model_data.json"

    # with open(data_file, "r", encoding="utf-8") as json_file:
    #     model_data = json.load(json_file)
    #     model_list = list(model_data.keys())

    # Do not include these parameters by default in the legend

    not_default = ["distance", "parallax", "mass", "luminosity"]

    if len(leg_param) == 0:
        leg_param = list(model_param.keys())

        for param_item in not_default:
            if param_item in leg_param:
                leg_param.remove(param_item)

    # Remove parameters from the model_param dictionary
    # if they are not included in the leg_param list

    del_keys = []
    for param_item in model_param.keys():
        if param_item not in leg_param:
            del_keys.append(param_item)

    for param_item in del_keys:
        del model_param[param_item]

    # Get parameters keys, units, and labels

    par_key, par_unit, par_label = quantity_unit(
        param=list(model_param.keys()), object_type=object_type
    )

    # Initiate an empty label that will be returned

    label = ""
    # newline = False

    # Optionally include the model name into the label

    if model_name is not None and inc_model_name:
        label += convert_model_name(model_name)

        if len(par_key) > 0:
            label += ": "

    for i, item in enumerate(par_key):
        if item[:4] == "teff":
            value = f"{model_param[item]:{param_fmt['teff']}}"

        elif item[:6] == "radius":
            if object_type == "planet":
                value = f"{model_param[item]:{param_fmt['radius']}}"
            elif object_type == "star":
                value = f"{model_param[item]*constants.R_JUP/constants.R_SUN:{param_fmt['radius']}}"

        elif item == "mass" and item in leg_param:
            if object_type == "planet":
                value = f"{model_param[item]:{param_fmt['mass']}}"
            elif object_type == "star":
                value = f"{model_param[item]*constants.M_JUP/constants.M_SUN:{param_fmt['mass']}}"

        elif item == "luminosity" and item in leg_param:
            value = f"{np.log10(model_param[item]):{param_fmt['luminosity']}}"

        elif item in leg_param and item in param_fmt:
            value = f"{model_param[item]:{param_fmt[item]}}"

        elif item in leg_param and item not in param_fmt:
            warnings.warn(
                f"The '{item}' parameter is not found in "
                "the dictionary of 'param_fmt'."
            )

            value = f"{model_param[item]:{param_fmt[item]:.2f}}"

        else:
            continue

        # if len(label) > 80 and newline == False:
        #     label += '\n'
        #     newline = True

        if par_unit[i] is None:
            if len(label) > 0 and label[-2] != ":":
                label += ", "

            label += f"{par_label[i]} = {value}"

        else:
            if len(label) > 0 and label[-2] != ":":
                label += ", "

            label += f"{par_label[i]} = {value} {par_unit[i]}"

    return label


@typechecked
def convert_units_plot(flux_in: np.ndarray, units_out: Tuple[str, str]) -> np.ndarray:
    """
    Function for converting the wavelength units from
    :math:`\\mu\\text{m}^{-1}` and the flux units from
    :math:`\\text{W} \\text{m}^{-2} \\mu\\text{m}^{-1}`.

    Parameters
    ----------
    flux_in : np.ndarray
        Array with the input wavelengths and fluxes. The shape of the
        array should be (n_wavelengths, 3) with the columns being
        the wavelengths, flux densities, and uncertainties. For
        photometric fluxes, the array should also be 2D but with
        a single row/wavelength.
    units_out : tuple(str, str)
        Tuple with the units of the wavelength ("um", "angstrom", "A",
        "nm", "mm", "cm", "m", "Hz") and the units of the flux density
        ("W m-2 um-1", "W m-2 m-1", "W m-2 Hz-1", "erg s-1 cm-2 Hz-1",
        "mJy", "Jy", "MJy").

    Returns
    -------
    np.ndarray
        Array with the output in the same shape as ``flux_in``.
    """

    speed_light = constants.LIGHT * 1e6  # (um s-1)

    flux_out = np.zeros(flux_in.shape)

    # Convert wavelengths from micrometer (um)

    wavel_units = ["um", "angstrom", "A", "nm", "mm", "cm", "m", "Hz"]

    if units_out[0] == "um":
        flux_out[:, 0] = flux_in[:, 0].copy()

    elif units_out[0] in ["angstrom", "A"]:
        flux_out[:, 0] = flux_in[:, 0] * 1e4

    elif units_out[0] == "nm":
        flux_out[:, 0] = flux_in[:, 0] * 1e3

    elif units_out[0] == "mm":
        flux_out[:, 0] = flux_in[:, 0] * 1e-3

    elif units_out[0] == "cm":
        flux_out[:, 0] = flux_in[:, 0] * 1e-4

    elif units_out[0] == "m":
        flux_out[:, 0] = flux_in[:, 0] * 1e-6

    elif units_out[0] == "Hz":
        flux_out[:, 0] = speed_light / flux_in[:, 0]

    else:
        raise ValueError(
            f"The wavelength units '{units_out[0]}' are not supported. "
            f"Please choose from the following units: {wavel_units}"
        )

    # Convert flux density from W m-2 um-1

    flux_units = [
        "W m-2 um-1",
        "W m-2 m-1",
        "W m-2 Hz-1",
        "erg s-1 cm-2 Hz-1",
        "mJy",
        "Jy",
        "MJy",
    ]

    if units_out[1] == "W m-2 um-1":
        flux_out[:, 1] = flux_in[:, 1].copy()
        flux_out[:, 2] = flux_in[:, 2].copy()

    elif units_out[1] == "W m-2 m-1":
        flux_out[:, 1] = flux_in[:, 1] * 1e6
        flux_out[:, 2] = flux_in[:, 2] * 1e6

    elif units_out[1] == "W m-2 Hz-1":
        flux_out[:, 1] = flux_in[:, 1] * flux_out[:, 0] ** 2 / speed_light
        flux_out[:, 2] = flux_in[:, 2] * flux_out[:, 0] ** 2 / speed_light

    elif units_out[1] == "erg s-1 cm-2 Hz-1":
        flux_out[:, 1] = flux_in[:, 1] * 1e3 * flux_out[:, 0] ** 2 / speed_light
        flux_out[:, 2] = flux_in[:, 2] * 1e3 * flux_out[:, 0] ** 2 / speed_light

    elif units_out[1] == "mJy":
        flux_out[:, 1] = flux_in[:, 1] * 1e29 * flux_out[:, 0] ** 2 / speed_light
        flux_out[:, 2] = flux_in[:, 2] * 1e29 * flux_out[:, 0] ** 2 / speed_light

    elif units_out[1] == "Jy":
        flux_out[:, 1] = flux_in[:, 1] * 1e26 * flux_out[:, 0] ** 2 / speed_light
        flux_out[:, 2] = flux_in[:, 2] * 1e26 * flux_out[:, 0] ** 2 / speed_light

    elif units_out[1] == "MJy":
        flux_out[:, 1] = flux_in[:, 1] * 1e20 * flux_out[:, 0] ** 2 / speed_light
        flux_out[:, 2] = flux_in[:, 2] * 1e20 * flux_out[:, 0] ** 2 / speed_light

    else:
        raise ValueError(
            f"The flux units '{units_out[1]}' are not supported. "
            f"Please choose from the following units: {flux_units}"
        )

    return flux_out
