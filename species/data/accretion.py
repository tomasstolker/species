"""
Module for the accretion luminosity relation.
"""

import os

import h5py
import numpy as np
import pooch

from typeguard import typechecked

from species.core import constants


@typechecked
def add_accretion_relation(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the accretion relation from `Aoyama et al.
    (2021) <https://ui. adsabs.harvard.edu/abs/
    2021ApJ...917L..30A/abstract>`_ and extrapolation from
    `Marleau & Aoyama (2022) <https://ui.adsabs.harvard.edu/abs/
    2022RNAAS...6..262M/abstract>`_ to the database. It provides
    coefficients to convert line to accretion luminosities.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    None
        NoneType
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url = (
        "https://home.strw.leidenuniv.nl/~stolker/"
        "species/ab-Koeffienzenten_mehrStellen.dat"
    )

    data_file = os.path.join(input_path, "ab-Koeffienzenten_mehrStellen.dat")

    if not os.path.isfile(data_file):
        pooch.retrieve(
            url=url,
            known_hash="941b416e678128648c9ce485016af908f16bfe16ab72e0f8cb57a6bad963429a",
            fname="ab-Koeffienzenten_mehrStellen.dat",
            path=input_path,
            progressbar=True,
        )

    print("Adding coefficients for accretion relation (2.1 kB)...", end="", flush=True)

    data = np.genfromtxt(
        data_file,
        dtype=None,
        skip_header=1,
        encoding=None,
        names=True,
        usecols=[0, 1, 2, 3, 4],
    )

    line_names = data["name"]
    coefficients = np.column_stack([data["a"], data["b"]])

    n_init = data["ni"]
    n_final = data["nf"]

    delta_n_min = {"H": 9, "Pa": 8, "Br": 6}
    delta_n_max = {"H": 14, "Pa": 13, "Br": 12}

    for i, item in enumerate(["H", "Pa", "Br"]):
        for j in range(delta_n_min[item], delta_n_max[item] + 1):
            # n_f: Ly=1, H=Ba=2, Pa=3, Br=4
            n_f_tmp = i + 2

            idx_insert = np.argwhere(line_names == f"{item}{j+n_f_tmp-1}")[0][0] + 1
            line_names = np.insert(line_names, idx_insert, f"{item}{j+n_f_tmp}")
            n_final = np.insert(n_final, idx_insert, n_f_tmp)

            # delta_n = n_i - n_f
            n_i_tmp = j + n_f_tmp
            n_init = np.insert(n_init, idx_insert, n_i_tmp)

            # Relation for extrapolation of coefficients
            # See Marleau & Aoyama (2022)

            a_coeff = 0.811 - (1.0 / (9.90 * n_f_tmp - 9.5 * n_i_tmp))

            b_coeff = (
                1.0
                + 1.05 * np.log(n_i_tmp)
                + (1.0 / (n_i_tmp - n_f_tmp))
                - (1.0 / n_f_tmp)
            ) * (1.07 + 0.0694 * n_f_tmp) - 1.41

            coefficients = np.insert(
                coefficients, idx_insert, [[a_coeff, b_coeff]], axis=0
            )

    # Rest vacuum wavelength (um) from Rydberg formula,
    # which is valid usually to three or four decimal places.
    # Exact values, if needed, can be obtained from e.g.
    # Wiese & Fuhr (2009):
    # http://adsabs.harvard.edu/abs/2009JPCRD..38..565W
    # https://www.nist.gov/system/files/documents/srd/jpcrd382009565p.pdf
    wavelengths = 1e6 / (constants.RYDBERG * (1.0 / n_final**2 - 1.0 / n_init**2))

    # data = np.column_stack([line_names, wavelengths, n_init, n_final, coefficients[:, 0], coefficients[:, 1]])
    # np.savetxt('acc_lines.dat', data, delimiter=" ", fmt="%s")

    database.create_dataset("accretion/wavelengths", data=wavelengths)
    database.create_dataset("accretion/coefficients", data=coefficients)

    dtype = h5py.special_dtype(vlen=str)
    dset = database.create_dataset(
        "accretion/hydrogen_lines", (np.size(line_names),), dtype=dtype
    )
    dset[...] = line_names

    print(" [DONE]")

    print(
        "Please cite Aoyama et al. (2021) and Marleau & Aoyama "
        "(2022) when using the accretion relation in a publication"
    )
