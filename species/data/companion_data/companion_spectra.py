"""
Module for getting spectra of directly imaged planets and brown dwarfs.
"""

import json
import pooch

from pathlib import Path
from typing import Dict, Optional, Tuple

from typeguard import typechecked

from species.util.core_util import print_section


@typechecked
def companion_spectra(
    input_path: Path, comp_name: str, verbose: bool = True
) -> Optional[Dict[str, Tuple[str, Optional[str], float]]]:
    """
    Function for extracting a dictionary with the spectra of
    directly imaged planets and brown dwarfs. These data can
    be added to the database with the
    :func:`~species.data.database.Database.add_companion`
    method of :class:`~species.data.database.Database`.

    Parameters
    ----------
    input_path : Path
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

    data_folder = input_path / "companion_data"
    spec_file = Path(__file__).parent.resolve() / "companion_spectra.json"

    with open(spec_file, "r", encoding="utf-8") as json_file:
        comp_spec = json.load(json_file)

    if comp_name in comp_spec:
        if verbose:
            print_section("Get companion spectra")

        if not data_folder.exists():
            data_folder.mkdir()

        spec_dict = {}

        for spec_key, spec_value in comp_spec[comp_name].items():
            if verbose:
                print(f"Getting {spec_key} spectrum of {comp_name}...", end="", flush=True)

            spec_url = (
                f"https://home.strw.leidenuniv.nl/~stolker/species/spectra/{spec_value[0]}"
            )
            spec_file = data_folder / spec_value[0]

            if spec_value[1] is None:
                cov_file = None
            else:
                cov_file = data_folder / spec_value[1]

            if not spec_file.exists():
                pooch.retrieve(
                    url=spec_url,
                    known_hash=None,
                    fname=spec_value[0],
                    path=data_folder,
                    progressbar=True,
                )

            if cov_file is None:
                spec_dict[spec_key] = (str(spec_file), cov_file, spec_value[2])
            else:
                spec_dict[spec_key] = (str(spec_file), str(cov_file), spec_value[2])

            if verbose:
                print(" [DONE]")

                print(
                    f"Please cite {spec_value[3]} when making "
                    "use of this spectrum in a publication"
                )

    else:
        spec_dict = None

    return spec_dict
