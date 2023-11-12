import os
import urllib.request

from species.data.isochrone_data.iso_manual import add_manual


def add_ames(database, input_path):
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

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url_list = [
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.AMES-Cond-2000.M-0.0.MKO.Vega",
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.AMES-dusty.M-0.0.MKO.Vega",
    ]

    iso_tags = ["AMES-Cond", "AMES-Dusty"]
    iso_size = ["235 kB", "182 kB"]

    for i, url_item in enumerate(url_list):
        input_file = url_item.split("/")[-1]
        data_file = os.path.join(input_path, input_file)

        if not os.path.isfile(data_file):
            print(
                f"Downloading {iso_tags[i]} isochrones ({iso_size[i]})...",
                end="",
                flush=True,
            )
            urllib.request.urlretrieve(url_item, data_file)
            print(" [DONE]")

        add_manual(
            database=database,
            tag=iso_tags[i].lower(),
            file_name=data_file,
            model_name="ames",
        )
