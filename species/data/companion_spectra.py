"""
Module for getting spectra of directly imaged planets and brown dwarfs.
"""

import os
import urllib.request

from typing import Dict, Optional, Tuple

from typeguard import typechecked


@typechecked
def get_spec_data() -> Dict[str, Dict[str, Tuple[str, Optional[str], float, str]]]:
    """
    Function for extracting a dictionary with the spectra of directly
    imaged planets. These data can be added to the database with
    :func:`~species.data.database.Database.add_companion`.

    Returns
    -------
    dict
        Dictionary with the spectrum, optional covariances, spectral
        resolution, and filename.
    """

    spec_data = {
        "beta Pic b": {
            "GPI_YJHK": (
                "betapicb_gpi_yjhk.dat",
                None,
                40.0,
                "Chilcote et al. 2017, AJ, 153, 182",
            ),
            "GRAVITY": (
                "BetaPictorisb_2018-09-22.fits",
                "BetaPictorisb_2018-09-22.fits",
                500.0,
                "Gravity Collaboration et al. 2020, A&A, 633, 110",
            ),
        },
        "51 Eri b": {
            "SPHERE_YJH": (
                "51erib_sphere_yjh.dat",
                None,
                25.0,
                "Samland et al. 2017, A&A, 603, 57",
            )
        },
        "HD 206893 B": {
            "SPHERE_YJH": (
                "hd206893b_sphere_yjh.dat",
                None,
                25.0,
                "Delorme et al. 2017, A&A, 608, 79",
            )
        },
        "HIP 65426 B": {
            "SPHERE_YJH": (
                "hip65426b_sphere_yjh.dat",
                None,
                25.0,
                "Cheetham et al. 2019, A&A, 622, 80",
            )
        },
        "HR 8799 e": {
            "SPHERE_YJH": (
                "hr8799e_sphere_yjh.dat",
                None,
                25.0,
                "Zurlo et al. 2016, A&A, 587, 57",
            )
        },
        "PDS 70 b": {
            "SPHERE_YJH": (
                "pds70b_sphere_yjh.dat",
                None,
                25.0,
                "Müller et al. 2018, A&A, 617, 2",
            )
        },
    }

    return spec_data


@typechecked
def companion_spectra(
    input_path: str, comp_name: str, verbose: bool = True
) -> Optional[Dict[str, Tuple[str, Optional[str], float]]]:
    """
    Function for getting available spectra of directly imaged planets
    and brown dwarfs.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    comp_name : str
        Companion name for which the spectra will be returned.
    verbose : bool
        Print details on the companion data that are added to the
        database.

    Returns
    -------
    dict, None
        Dictionary with the spectra of ``comp_name``. A ``None`` will
        be returned if there are not any spectra available.
    """

    spec_data = get_spec_data()

    if comp_name in spec_data:
        data_folder = os.path.join(input_path, "companion_data/")

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        spec_dict = {}

        for key, value in spec_data[comp_name].items():
            if verbose:
                print(f"Getting {key} spectrum of {comp_name}...", end="", flush=True)

            spec_url = (
                f"https://home.strw.leidenuniv.nl/~stolker/species/spectra/{value[0]}"
            )
            spec_file = os.path.join(data_folder, value[0])

            if value[1] is None:
                cov_file = None
            else:
                cov_file = os.path.join(data_folder, value[1])

            if not os.path.isfile(spec_file):
                urllib.request.urlretrieve(spec_url, spec_file)

            spec_dict[key] = (spec_file, cov_file, value[2])

            if verbose:
                print(" [DONE]")

                print(f"IMPORTANT: Please cite {value[3]}")
                print("           when making use of this spectrum in a publication")

    else:
        spec_dict = None

    return spec_dict
