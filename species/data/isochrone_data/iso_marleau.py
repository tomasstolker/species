import numpy as np

from species.core import constants


def add_marleau(database, tag, file_name):
    """
    Function for adding the Marleau et al. isochrone data
    to the database. The isochrone data can be requested
    from Gabriel Marleau.

    https://ui.adsabs.harvard.edu/abs/2019A%26A...624A..20M/abstract

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    file_name : str
        Filename with the isochrones data.

    Returns
    -------
    NoneType
        None
    """

    # M      age     S_0             L          S(t)            R        Teff
    # (M_J)  (Gyr)   (k_B/baryon)    (L_sol)    (k_B/baryon)    (R_J)    (K)
    mass, age, _, luminosity, _, radius, teff = np.loadtxt(file_name, unpack=True)

    age *= 1e3  # (Myr)
    luminosity = np.log10(luminosity)

    mass_cgs = 1e3 * mass * constants.M_JUP  # (g)
    radius_cgs = 1e2 * radius * constants.R_JUP  # (cm)

    logg = np.log10(1e3 * constants.GRAVITY * mass_cgs / radius_cgs**2)

    print(f"Adding isochrones: {tag}...", end="", flush=True)

    isochrones = np.vstack((age, mass, teff, luminosity, logg))
    isochrones = np.transpose(isochrones)

    index_sort = np.argsort(isochrones[:, 0])
    isochrones = isochrones[index_sort, :]

    dset = database.create_dataset(f"isochrones/{tag}/evolution", data=isochrones)

    dset.attrs["model"] = "marleau"

    print(" [DONE]")
