"""
Utility functions for generating ``petitRADTRANS`` spectra.
"""

from typing import Dict, List, Optional

import numpy as np

from typeguard import typechecked

from species.core.box import ModelBox
from species.read.read_radtrans import ReadRadtrans


@typechecked
def retrieval_spectrum(
    indices: Dict[str, np.int64],
    chemistry: str,
    pt_profile: str,
    line_species: List[str],
    cloud_species: List[str],
    quenching: Optional[str],
    spec_res: Optional[float],
    distance: Optional[float],
    pt_smooth: Optional[float],
    temp_nodes: Optional[np.integer],
    abund_nodes: Optional[np.integer],
    abund_smooth: Optional[float],
    read_rad: ReadRadtrans,
    sample: np.ndarray,
) -> ModelBox:
    """
    Function for calculating a petitRADTRANS spectrum
    from a posterior sample.

    Parameters
    ----------
    indices : dict
        Dictionary with the parameter indices for ``sample``.
    chemistry : str
        Chemistry type (``'equilibrium'`` or ``'free'``).
    pt_profile : str
        Pressure-temperature parametrization (``'molliere'``,
        ``'monotonic'``, or ``'free'``).
    line_species : list(str)
        List with the line species.
    cloud_species : list(str)
        List with the cloud species.
    quenching : str, None
        Quenching type for CO/CH4/H2O abundances. Either the quenching
        pressure (bar) is a free parameter (``quenching='pressure'``)
        or the quenching pressure is calculated from the mixing and
        chemical timescales (``quenching='diffusion'``). The quenching
        is not applied if the argument is set to ``None``.
    spec_res : float, None
        Spectral resolution. No smoothing is applied if the argument
        is set to ``None``.
    distance : float, None
        Distance (pc).
    pt_smooth : float, None
        Standard deviation of the Gaussian kernel that is used for
        smoothing the sampled temperature nodes of the P-T profile.
        Only required with `pt_profile='free'` or
        `pt_profile='monotonic'`. The argument should be given as
        log10(P/bar).
    temp_nodes : int, None
        Number of free temperature nodes that are used when
        ``pt_profile='monotonic'`` or ``pt_profile='free'``.
    abund_nodes : int, None
        Number of free abundance nodes that are used when
        ``chemistry='free'``.
    abund_smooth : float, None
        Standard deviation of the Gaussian kernel that is used for
        smoothing the abundance profiles, after the abundance nodes
        have been interpolated to a higher pressure resolution.
        Only required with ```chemistry='free'```. The argument
        should be given as :math:`\\log10{P/\\mathrm{bar}}`. No
        smoothing is applied if the argument if set to 0 or ``None``.
    read_rad : ReadRadtrans
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
    sample : np.ndarray
        Parameter values with their order given by the ``indices``.

    Returns
    -------
    species.core.box.ModelBox
        Box with the petitRADTRANS spectrum.
    """

    # Initiate parameter dictionary

    model_param = {}

    # Add log(g) and radius

    model_param["logg"] = sample[indices["logg"]]
    model_param["radius"] = sample[indices["radius"]]

    # Add distance

    if distance is not None:
        model_param["distance"] = distance

    # Add P-T profile parameters

    if pt_profile == "molliere":
        model_param["t1"] = sample[indices["t1"]]
        model_param["t2"] = sample[indices["t2"]]
        model_param["t3"] = sample[indices["t3"]]
        model_param["log_delta"] = sample[indices["log_delta"]]
        model_param["alpha"] = sample[indices["alpha"]]
        model_param["tint"] = sample[indices["tint"]]

    elif pt_profile == "gradient":
        num_layer = 6  # could make a variable in the future
        for index in range(num_layer):
            model_param[f"PTslope_{num_layer - index}"] = sample[
                indices[f"PTslope_{num_layer - index}"]
            ]
        model_param["T_bottom"] = sample[indices["T_bottom"]]

    elif pt_profile == "eddington":
        model_param["log_delta"] = sample[indices["log_delta"]]
        model_param["tint"] = sample[indices["tint"]]

    elif pt_profile in ["free", "monotonic"]:
        if temp_nodes is None:
            # For backward compatibility
            temp_nodes = 15

        for j in range(temp_nodes):
            model_param[f"t{j}"] = sample[indices[f"t{j}"]]

    if pt_smooth is not None:
        model_param["pt_smooth"] = pt_smooth

    # Add chemistry parameters

    if chemistry == "equilibrium":
        model_param["c_o_ratio"] = sample[indices["c_o_ratio"]]
        model_param["metallicity"] = sample[indices["metallicity"]]

    elif chemistry == "free":
        if abund_nodes is None:
            for line_item in line_species:
                model_param[line_item] = sample[indices[line_item]]

        else:
            for line_item in line_species:
                for node_idx in range(abund_nodes):
                    model_param[f"{line_item}_{node_idx}"] = sample[
                        indices[f"{line_item}_{node_idx}"]
                    ]

    if abund_smooth is not None:
        model_param["abund_smooth"] = abund_smooth

    if quenching == "pressure":
        model_param["log_p_quench"] = sample[indices["log_p_quench"]]

    # Add cloud parameters

    if "log_kappa_0" in indices:
        model_param["log_kappa_0"] = sample[indices["log_kappa_0"]]
        model_param["opa_index"] = sample[indices["opa_index"]]
        model_param["log_p_base"] = sample[indices["log_p_base"]]
        model_param["albedo"] = sample[indices["albedo"]]

        if "fsed" in indices:
            model_param["fsed"] = sample[indices["fsed"]]

        elif "fsed_1" in indices and "fsed_2" in indices:
            model_param["fsed_1"] = sample[indices["fsed_1"]]
            model_param["fsed_2"] = sample[indices["fsed_2"]]
            model_param["f_clouds"] = sample[indices["f_clouds"]]

        if "opa_knee" in indices:
            model_param["opa_knee"] = sample[indices["opa_knee"]]

    elif "log_kappa_abs" in indices:
        model_param["log_kappa_abs"] = sample[indices["log_kappa_abs"]]
        model_param["opa_abs_index"] = sample[indices["opa_abs_index"]]
        model_param["log_p_base"] = sample[indices["log_p_base"]]
        model_param["fsed"] = sample[indices["fsed"]]

        if "log_kappa_sca" in indices:
            model_param["log_kappa_sca"] = sample[indices["log_kappa_sca"]]
            model_param["opa_sca_index"] = sample[indices["opa_sca_index"]]
            model_param["lambda_ray"] = sample[indices["lambda_ray"]]

    elif "log_kappa_gray" in indices:
        model_param["log_kappa_gray"] = sample[indices["log_kappa_gray"]]
        model_param["log_cloud_top"] = sample[indices["log_cloud_top"]]

        if "albedo" in indices:
            model_param["albedo"] = sample[indices["albedo"]]

    elif len(cloud_species) > 0:
        if "fsed" in indices:
            model_param["fsed"] = sample[indices["fsed"]]
        else:
            for item in cloud_species:
                model_param[f"fsed_{item}"] = sample[indices[f"fsed_{item}"]]
        model_param["sigma_lnorm"] = sample[indices["sigma_lnorm"]]

        if "kzz" in indices:
            # Backward compatibility
            model_param["kzz"] = sample[indices["kzz"]]

        elif "log_kzz" in indices:
            model_param["log_kzz"] = sample[indices["log_kzz"]]

        for cloud_item in cloud_species:
            cloud_param = f"{cloud_item}_fraction"

            if cloud_param in indices:
                model_param[cloud_param] = sample[indices[cloud_param]]

            cloud_param = f"{cloud_item}_tau"

            if cloud_param in indices:
                model_param[cloud_param] = sample[indices[cloud_param]]

            if cloud_item in indices:
                model_param[cloud_item] = sample[indices[cloud_item]]

    if "log_tau_cloud" in indices:
        model_param["tau_cloud"] = 10.0 ** sample[indices["log_tau_cloud"]]

        if len(cloud_species) > 1:
            for cloud_item in cloud_species[1:]:
                cloud_ratio = f"{cloud_item}_{cloud_species[0]}_ratio"
                model_param[cloud_ratio] = sample[indices[cloud_ratio]]

    # Add extinction parameters

    if "ism_ext" in indices:
        model_param["ism_ext"] = sample[indices["ism_ext"]]

    if "ism_red" in indices:
        model_param["ism_red"] = sample[indices["ism_red"]]

    # Calculate spectrum

    model_box = read_rad.get_model(model_param, spec_res=spec_res)

    # Set content type of the ModelBox

    model_box.type = "mcmc"

    return model_box
