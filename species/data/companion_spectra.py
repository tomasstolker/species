"""
Module for getting spectra of directly imaged planets and brown dwarfs.
"""

import json
import os
import pathlib
import urllib.request

from typing import Dict, Optional, Tuple

from typeguard import typechecked


@typechecked
def companion_spectra(
    input_path: str, comp_name: str, verbose: bool = True
) -> Optional[Dict[str, Tuple[str, Optional[str], float]]]:
    """
    Function for extracting a dictionary with the spectra of
    directly imaged planets and brown dwarfs. These data can
    be added to the database with the
    :func:`~species.data.database.Database.add_companion`
    method of :class:`~species.data.database.Database`.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    comp_name : str
        Companion name for which the spectra will be returned.
    verbose : bool
        Print details on the companion data that are added to
        the database.

    Returns
    -------
    dict, None
        Dictionary with the spectra of ``comp_name``. A ``None``
        will be returned if there are not any spectra available.
        The dictionary includes the spectrum, (optional)
        covariances, spectral resolution, and filename.
    """

    spec_file = pathlib.Path(__file__).parent.resolve() / "companion_spectra.json"

    with open(spec_file, "r", encoding="utf-8") as json_file:
        comp_spec = json.load(json_file)

    if comp_name in comp_spec:
        data_folder = os.path.join(input_path, "companion_data/")

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        spec_dict = {}

        for key, value in comp_spec[comp_name].items():
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

                print(f"Please cite {value[3]} when making "
                      "use of this spectrum in a publication")

    else:
        spec_dict = None

    return spec_dict
