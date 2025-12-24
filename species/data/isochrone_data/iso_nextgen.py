"""
Module with a function for adding the NextGen
evolutionary tracks to the database.
"""

from pathlib import Path

import h5py
import pooch

from typeguard import typechecked

from species.data.isochrone_data.iso_manual import add_manual


@typechecked
def add_nextgen(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the NextGen isochrone data to the database.

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    input_path : str
        Folder where the data is located.

    Returns
    -------
    NoneType
        None
    """

    url = (
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.NextGen.M-0.0.MKO.Vega"
    )

    iso_tag = "nextgen"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        pooch.retrieve(
            url=url,
            known_hash="a72aef342a1782553094114dabc3196c0862a0c17ee5c368fbd1b47b5c611363",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    add_manual(
        database=database,
        tag=iso_tag,
        file_name=str(data_file),
        model_name="nextgen",
    )
