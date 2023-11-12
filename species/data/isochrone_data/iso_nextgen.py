import os
import urllib.request

from species.data.isochrone_data.iso_manual import add_manual


def add_nextgen(database, input_path):
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

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url_iso = (
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.NextGen.M-0.0.MKO.Vega"
    )

    iso_tag = "NextGen"
    iso_size = "177 kB"

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
        model_name="nextgen",
    )
