"""
Module adding evolutionary tracks to the database.
"""

from typing import Optional

import h5py

from typeguard import typechecked

from species.data.isochrone_data.iso_ames import add_ames
from species.data.isochrone_data.iso_atmo import add_atmo
from species.data.isochrone_data.iso_baraffe2015 import add_baraffe2015
from species.data.isochrone_data.iso_btsettl import add_btsettl
from species.data.isochrone_data.iso_chabrier2023 import add_chabrier2023
from species.data.isochrone_data.iso_linder2019 import add_linder2019
from species.data.isochrone_data.iso_manual import add_manual
from species.data.isochrone_data.iso_marleau2014 import add_marleau2014
from species.data.isochrone_data.iso_nextgen import add_nextgen
from species.data.isochrone_data.iso_parsec import add_parsec
from species.data.isochrone_data.iso_saumon2008 import add_saumon2008
from species.data.isochrone_data.iso_sonora_bobcat import add_sonora_bobcat
from species.data.isochrone_data.iso_sonora_diamondback import add_sonora_diamondback
from species.data.isochrone_data.iso_spiegel2012 import add_spiegel2012


@typechecked
def add_isochrone_grid(
    data_folder: str,
    hdf5_file: h5py._hl.files.File,
    model_name: Optional[str] = None,
    filename: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    """
    Function for adding an isochrone grid to the database.

    Parameters
    ----------
    data_folder : str
        Folder where the data is located.
    hdf5_file : h5py._hl.files.File
        Database.
    model_name : str, None
        Evolutionary model ('ames', 'atmo', 'atmo-chabrier2023',
        'baraffe2015', 'bt-settl', 'linder2019', 'marleau2014',
        'nextgen', 'parsec', 'saumon2008', 'sonora-bobcat',
        'sonora-diamondback', 'spiegel2012'). Isochrones will
        be automatically downloaded. Alternatively, the
        ``filename`` parameter can be used in combination
        with ``tag``.
    filename : str, None
        Filename with the isochrone data. The argument of
        ``model`` will be ignored by setting the argument
        of ``filename``. When using ``filename``, also
        the argument of ``tag`` should be set. Only files
        with isochrone data from
        https://phoenix.ens-lyon.fr/Grids/ and
        https://perso.ens-lyon.fr/isabelle.baraffe/ are
        supported. The parameter is ignored by setting
        the argument to ``None``.
    tag : str, None
        Database tag name where the isochrone that will be
        stored. Setting the argument is only required in
        combination with the ``filename`` parameter.
        Otherwise, the argument can be set to ``None``.

    Returns
    -------
    None
        NoneType
    """

    if model_name == "ames":
        add_ames(hdf5_file, data_folder)

    elif model_name == "atmo":
        add_atmo(hdf5_file, data_folder)

    elif model_name == "atmo-chabrier2023":
        add_chabrier2023(hdf5_file, data_folder)

    elif model_name == "baraffe2015":
        add_baraffe2015(hdf5_file, data_folder)

    elif model_name == "bt-settl":
        add_btsettl(hdf5_file, data_folder)

    elif model_name == "linder2019":
        add_linder2019(hdf5_file, data_folder)

    elif model_name == "marleau2014":
        add_marleau2014(hdf5_file, data_folder)

    elif model_name == "nextgen":
        add_nextgen(hdf5_file, data_folder)

    elif model_name == "parsec":
        add_parsec(hdf5_file, data_folder)

    elif model_name == "saumon2008":
        add_saumon2008(hdf5_file, data_folder)

    elif model_name == "sonora-bobcat":
        add_sonora_bobcat(hdf5_file, data_folder)

    elif model_name == "sonora-diamondback":
        add_sonora_diamondback(hdf5_file, data_folder)

    elif model_name == "spiegel2012":
        add_spiegel2012(hdf5_file, data_folder)

    else:
        add_manual(hdf5_file, tag, filename, model_name=tag)
