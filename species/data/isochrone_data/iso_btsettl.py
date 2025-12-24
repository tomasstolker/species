"""
Module with a function for adding the BT-Setll
evolutionary tracks to the database.
"""

from pathlib import Path

import h5py
import pooch

from typeguard import typechecked

from species.data.isochrone_data.iso_manual import add_manual


@typechecked
def add_btsettl(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the BT-Settl isochrone data to the database.

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
        "model.BT-Settl.M-0.0.MKO.Vega"
    )

    iso_tag = "bt-settl"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="18e6a2b1e0b2452973c9af3e34138d5346cf04945239a085142f1fade5f7946e",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    add_manual(
        database=database,
        tag=iso_tag,
        file_name=str(data_file),
        model_name="bt-settl",
    )
