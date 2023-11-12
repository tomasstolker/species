import os
import urllib.request

from species.data.isochrone_data.iso_manual import add_manual


def add_btsettl(database, input_path):
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

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url_iso = (
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.BT-Settl.M-0.0.MKO.Vega"
    )

    iso_tag = "BT-Settl"
    iso_size = "113 kB"

    input_file = url_iso.rsplit("/", maxsplit=1)[-1]
    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(f"Downloading {iso_tag} isochrones ({iso_size})...", end="", flush=True)
        urllib.request.urlretrieve(url_iso, data_file)
        print(" [DONE]")

    add_manual(
        database=database,
        tag=iso_tag.lower(),
        file_name=data_file,
        model_name="bt-settl",
    )
