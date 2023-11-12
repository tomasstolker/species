"""
Module for isochrone data from evolutionary models.
"""

from typing import Optional

import h5py

from species.data.isochrone_data.iso_ames import add_ames
from species.data.isochrone_data.iso_atmo import add_atmo
from species.data.isochrone_data.iso_baraffe2015 import add_baraffe2015
from species.data.isochrone_data.iso_btsettl import add_btsettl
from species.data.isochrone_data.iso_linder2019 import add_linder2019
from species.data.isochrone_data.iso_manual import add_manual
from species.data.isochrone_data.iso_marleau import add_marleau
from species.data.isochrone_data.iso_nextgen import add_nextgen
from species.data.isochrone_data.iso_saumon2008 import add_saumon2008
from species.data.isochrone_data.iso_sonora import add_sonora


def add_isochrone_grid(
    data_folder: str,
    hdf5_file: h5py._hl.files.File,
    model_name: str,
    filename: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    """
    Function for adding an isochrone grid to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    model_name : str
        Evolutionary model ('ames', 'atmo', 'baraffe2015',
        'bt-settl', 'linder2019', 'nextgen', 'saumon2008',
        'sonora', or 'manual'). Isochrones will be
        automatically downloaded. Alternatively,
        isochrone data can be downloaded from
        https://phoenix.ens-lyon.fr/Grids/ or
        https://perso.ens-lyon.fr/isabelle.baraffe/, and can
        be manually added by setting the ``filename`` and
        ``tag`` arguments, and setting ``model='manual'``.
    filename : str, None
        Filename with the isochrone data. Setting the argument
        is only required when ``model='manual'``. Otherwise,
        the argument can be set to ``None``.
    tag : str, None
        Database tag name where the isochrone that will be
        stored. Setting the argument is only required when
        ``model='manual'``. Otherwise, the argument can be
        set to ``None``.

    Returns
    -------
    None
        NoneType
    """

    if model_name == "ames":
        add_ames(hdf5_file, data_folder)

    elif model_name == "atmo":
        add_atmo(hdf5_file, data_folder)

    elif model_name == "baraffe2015":
        add_baraffe2015(hdf5_file, data_folder)

    elif model_name == "bt-settl":
        add_btsettl(hdf5_file, data_folder)

    elif model_name == "linder2019":
        add_linder2019(hdf5_file, data_folder)

    elif model_name == "manual":
        add_manual(hdf5_file, tag, filename)

    elif model_name == "marleau":
        add_marleau(hdf5_file, tag, filename)

    elif model_name == "nextgen":
        add_nextgen(hdf5_file, data_folder)

    elif model_name == "saumon2008":
        add_saumon2008(hdf5_file, data_folder)

    elif model_name == "sonora":
        add_sonora(hdf5_file, data_folder)

    else:
        raise ValueError(
            f"The evolutionary model_name '{model_name}' is "
            "not supported. Please choose another "
            "argument for 'model_name'. Have a look "
            "at the documentation of add_isochrones "
            "for details on the supported model_names."
        )
