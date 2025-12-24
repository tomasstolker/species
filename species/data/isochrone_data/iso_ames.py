"""
Module with a function for adding the AMES-Cond and AMES-Dusty
evolutionary tracks to the database.
"""

from pathlib import Path

import h5py
import pooch

from typeguard import typechecked

from species.data.isochrone_data.iso_manual import add_manual


@typechecked
def add_ames(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the AMES-Cond and AMES-Dusty
    isochrone data to the database.

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

    url_list = [
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.AMES-Cond-2000.M-0.0.MKO.Vega",
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.AMES-dusty.M-0.0.MKO.Vega",
    ]

    file_hash = [
        "fc04e6f7c02982bb3187b55cdefc2464e3f1564fb8026a8958967cb889f0f581",
        "c7ba32ae10111c9ca692bf75154edac70b050c06cae211b421e1473725d6380c",
    ]

    iso_tags = ["ames-cond", "ames-dusty"]

    for url_idx, url_item in enumerate(url_list):
        input_file = url_item.split("/")[-1]
        data_file = Path(input_path) / input_file

        if not data_file.exists():
            print()

            pooch.retrieve(
                url=url_item,
                known_hash=file_hash[url_idx],
                fname=input_file,
                path=input_path,
                progressbar=True,
            )

        add_manual(
            database=database,
            tag=iso_tags[url_idx],
            file_name=str(data_file),
            model_name="ames",
        )
