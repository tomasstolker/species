"""
Module for getting spectra of directly imaged planets and brown dwarfs.
"""

import json

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pooch

from astropy.io import fits
from beartype import beartype
from beartype.typing import Dict, Optional, Tuple, Union

from species.util.core_util import print_section

GRAVITY_TARGETS = {
    "51 Eri b": "51Eri_b",
    "AF Lep b": "AFLep_b",
    "BD+70 260 b": "2MASSJ03504_b",
    "beta Pic b": "BetaPictoris_b",
    "beta Pic c": "BetaPictoris_c",
    "CD-35 2722 b": "CD-352722_b",
    "GQ Lup B": "GQLup_b",
    "HD 984 b": "HD984_b",
    "HD 1160 B": "HD1160_b",
    "HD 4113 c": "HD4113_c",
    "HD 4747 B": "HD4747_b",
    "HD 13724 B": "HD13724_b",
    "HD 17155 b": "HD17155_b",
    "HD 72946 B": "HD72946_b",
    "HD 95086 b": "HD95086_b",
    "HD 112863 b": "HD112863_b",
    "HD 135344 Ab": "SAO206463_b",
    "HD 167665 b": "HD167665_b",
    "HD 206505 b": "HD206505_b",
    "HD 206893 B": "HD206893_b",
    "HD 206893 c": "HD206893_c",
    "HIP 64892 B": "HIP64892_b",
    "HIP 65426 b": "HIP65426_b",
    "HIP 75056 b": "HIP75056_b",
    "HIP 79098 b": "HIP79098_b",
    "HIP 99770 b": "HIP99770_b",
    "HR 2562 b": "HR2562_b",
    "HR 3549 B": "HR3549_b",
    "HR 7672 b": "HR7672_b",
    "HR 8799 b": "HR8799_b",
    "HR 8799 c": "HR8799_c",
    "HR 8799 d": "HR8799_d",
    "HR 8799 e": "HR8799_e",
    "mu2 Sco b": "Mu2Sco_b",
    "PDS 70 b": "PDS70_b",
    "PDS 70 c": "PDS70_c",
    "PZ Tel B": "PZTel_b",
    "WT 766 b": "WT766_b",
    "YSES 1 b": "YSES1_b",
}


@beartype
def companion_spectra(
    input_path: Path,
    comp_name: str,
    verbose: bool = True,
) -> Optional[
    Dict[str, Tuple[Union[str, np.ndarray], Optional[Union[str, np.ndarray]], float]]
]:
    """
    Extract a dictionary with the spectra of directly imaged planets
    and brown dwarfs. These data can be added to the database with the
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
        Dictionary with the spectra of ``comp_name``. A
        ``None`` will be returned if there are not any
        spectra available. Each entry contains the spectrum,
        optional covariance matrix, and the spectral
        resolution. Data are returned as filenames or arrays.
    """

    data_folder = input_path / "companion_data"
    spec_file = Path(__file__).parent.resolve() / "companion_spectra.json"

    with open(spec_file, "r", encoding="utf-8") as json_file:
        comp_spec = json.load(json_file)

    gravity_target = GRAVITY_TARGETS.get(comp_name)

    if comp_name in comp_spec or gravity_target is not None:
        if verbose:
            print_section("Get companion spectra")

        if not data_folder.exists():
            data_folder.mkdir()

        spec_dict = {}

        for spec_key, spec_value in comp_spec.get(comp_name, {}).items():
            if spec_key == "GRAVITY" and gravity_target is not None:
                continue

            if verbose:
                print(
                    f"Getting {spec_key} spectrum of {comp_name}...", end="", flush=True
                )

            spec_url = f"https://home.strw.leidenuniv.nl/~stolker/species/spectra/{spec_value[0]}"
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

        if gravity_target is not None:
            if verbose:
                print(f"Getting combined GRAVITY spectrum of {comp_name}...")

            archive = data_folder / "ExoGRAVITY_Spectral_Library.zip"

            if not archive.exists():
                pooch.retrieve(
                    url="https://zenodo.org/records/17294469/files/ExoGRAVITY_Spectral_Library.zip",
                    known_hash="md5:6029fa5268eba33cd83da3a3944d0e5e",
                    fname="ExoGRAVITY_Spectral_Library.zip",
                    path=data_folder,
                    progressbar=verbose,
                )

            member = f"{gravity_target}/{gravity_target}_gravity_combined.fits"

            with ZipFile(archive) as zip_file:
                with fits.open(BytesIO(zip_file.read(member))) as hdul:
                    table = hdul["COMPANION_FLUX"].data

                    spectrum = np.column_stack(
                        (table["WAVELENGTH"], table["FLUX"], table["FLUX_STD"])
                    )

                    covariance = np.array(table["FLUX_COV"], dtype=float)

            spec_dict["GRAVITY"] = (spectrum, covariance, 500.0)

            if verbose:
                print(
                    "Please cite Kammerer et al. 2025, A&A, 704, A318 "
                    "when making use of this spectrum in a publication"
                )

    else:
        spec_dict = None

    return spec_dict
