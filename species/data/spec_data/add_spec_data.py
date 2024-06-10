"""
Module for adding spectral libraries to the database.
"""

from typing import List, Optional

import h5py

from typeguard import typechecked

from species.data.spec_data.spec_allers2013 import add_allers2013
from species.data.spec_data.spec_bonnefoy2014 import add_bonnefoy2014
from species.data.spec_data.spec_irtf import add_irtf
from species.data.spec_data.spec_kesseli2017 import add_kesseli2017
from species.data.spec_data.spec_spex import add_spex
from species.data.spec_data.spec_vega import add_vega


@typechecked
def add_spec_library(
    input_path: str,
    database: h5py._hl.files.File,
    spec_library: str,
    sptypes: Optional[List[str]] = None,
) -> None:
    """
    Function for adding spectral libraries to the database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    spec_library : str
        Name of the spectral library ('irtf', 'spex', 'kesseli+2017',
        'bonnefoy+2014', 'allers+2013').
    sptypes : list(str), None
        Spectral types ('F', 'G', 'K', 'M', 'L', 'T'). Currently
        only implemented for ``spec_library='irtf'``.

    Returns
    -------
    None
        NoneType
    """

    if spec_library[0:11] == "allers+2013":
        add_allers2013(input_path, database)

    elif spec_library[0:13] == "bonnefoy+2014":
        add_bonnefoy2014(input_path, database)

    elif spec_library[0:5] == "irtf":
        add_irtf(input_path, database, sptypes)

    elif spec_library[0:12] == "kesseli+2017":
        add_kesseli2017(input_path, database)

    elif spec_library[0:5] == "spex":
        add_spex(input_path, database)

    elif spec_library[0:5] == "vega":
        add_vega(input_path, database)

    else:
        raise ValueError(
            f"The spectral library '{spec_library}' is not supported. "
            "Please adjust the argument of 'spec_library'."
        )
