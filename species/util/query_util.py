"""
Text
"""

import os
import sys

# import time
import warnings

import h5py
import numpy as np

from numpy import ma
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from species.data import database


class NoStdStreams:
    """
    Text
    """

    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, "w")
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


# with NoStdStreams():
#     from astroquery.gaia import Gaia


def get_parallax():
    species_db = database.Database()
    species_db.add_photometry("vlm-plx")

    with h5py.File(species_db.database, "a") as hdf_file:
        name = np.asarray(hdf_file["photometry/vlm-plx/name"])
        ra_coord = np.asarray(hdf_file["photometry/vlm-plx/ra"])
        dec_coord = np.asarray(hdf_file["photometry/vlm-plx/dec"])
        distance = np.asarray(hdf_file["photometry/vlm-plx/distance"])
        distance_error = np.asarray(hdf_file["photometry/vlm-plx/distance_error"])

        simbad_id = []

        print("Querying SIMBAD...", end="", flush=True)

        for i, item in enumerate(name):
            target_coord = SkyCoord(
                ra_coord[i], dec_coord[i], unit=(u.deg, u.deg), frame="icrs"
            )

            result_table = Simbad.query_region(target_coord, radius="0d0m2s")

            if result_table is None:
                result_table = Simbad.query_region(target_coord, radius="0d0m5s")

            if result_table is None:
                result_table = Simbad.query_region(target_coord, radius="0d0m20s")

            if result_table is None:
                result_table = Simbad.query_region(target_coord, radius="0d1m0s")

            if item == "HIP38939B":
                sim_id = get_simbad("HIP38939")
            else:
                sim_id = result_table["MAIN_ID"][0]

            # For backward compatibility
            if not isinstance(sim_id, str):
                sim_id = sim_id.decode("utf-8")

            simbad_id.append(sim_id)

        print(" [DONE]")

        simbad_id = np.asarray(simbad_id)

        dtype = h5py.special_dtype(vlen=str)

        dset = hdf_file.create_dataset(
            "photometry/vlm-plx/simbad", (np.size(simbad_id),), dtype=dtype
        )

        dset[...] = simbad_id

        np.savetxt(
            "parallax.dat",
            np.column_stack([name, simbad_id, distance, distance_error]),
            header="VLM-PLX name - SIMBAD name - Distance (pc) - Error (pc)",
            fmt="%35s, %35s, %8.2f, %8.2f",
        )


def get_simbad(name):
    """
    Function for getting the SIMBAD identifier of an object.

    Parameters
    ----------
    name : np.ndarray

    Returns
    -------
    np.ndarray
        SIMBAD name.
    """

    simbad = Simbad.query_object(name)

    if simbad is None:
        simbad_id = None

    else:
        simbad_id = simbad["MAIN_ID"][0]

    return simbad_id


def get_distance(target):
    """
    Parameters
    ----------
    target : str
        Target name.

    Returns
    -------
    str
        SIMBAD name.
    tuple(float, float)
        Distance and uncertainty (pc).
    """

    # Liu et al. (2016)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/833/96

    # Dupuy & Liu (2012)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJS/201/19

    # Faherty et al. (2012)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/752/56

    # Weinberger et al. (2016)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/AJ/152/24

    # Theissen (2018)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/862/173

    # Kirkpatrick et al. (2012)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/753/156

    # Monet et al. (1992)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/AJ/103/638

    # Faherty et al. (2009)
    # https://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/AJ/137/1

    catalogs = [
        "J/ApJ/833/96/sample",
        "J/ApJS/201/19/plx-phot",
        "J/ApJ/752/56/targets",
        "J/AJ/152/24",
        "J/ApJ/862/173/table",
        "J/AJ/103/638/table1",
        "J/AJ/103/638",
        "J/AJ/137/1/table4",
        "J/AJ/137/1",
    ]

    # if target[-2:] == 'AB':
    #     target = target[:-2]
    #
    # elif target[-3:] == 'ABC':
    #     target = target[:-3]
    #
    # if target[0:7] == 'DENIS-P':
    #     target = target[:5]+target[7:]
    #
    #     if target[-2] == '.':
    #         target = target[:-2]
    #
    # if target[0:5] == 'DENIS' and target[6:7] != 'J':
    #     target = target[:5]+' J'+target[6:]

    # time.sleep(0.15)

    simbad = Simbad()
    simbad.add_votable_fields("parallax")

    simbad_result = simbad.query_object(target)

    # query SIMBAD
    if simbad_result is not None:
        simbad_id = simbad_result["MAIN_ID"][0]

        # For backward compatibility
        if not isinstance(simbad_id, str):
            simbad_id = simbad_id.decode("utf-8")

        parallax = simbad_result["PLX_VALUE"][0]  # (mas)
        parallax_error = simbad_result["PLX_ERROR"][0]  # (mas)

        if ma.is_masked(parallax):
            parallax = None

        if ma.is_masked(parallax_error):
            parallax_error = None

    else:
        simbad_id = None
        parallax = None
        parallax_error = None

    distance = None
    distance_error = None

    # query VizieR catalogs
    if parallax is None:

        for item in catalogs:
            result = Vizier.query_object(target, catalog=item)

            if result.keys():

                if "plx" in result[0].keys():
                    parallax = result[0]["plx"][0]  # (mas)
                    distance = None
                    if ma.is_masked(parallax):
                        parallax = None

                elif "Plx" in result[0].keys():
                    parallax = result[0]["Plx"][0]  # (mas)
                    distance = None
                    if ma.is_masked(parallax):
                        parallax = None

                elif "Dist" in result[0].keys():
                    distance = result[0]["Dist"][0]  # (pc)
                    parallax = None
                    if ma.is_masked(distance):
                        distance = None

                else:
                    parallax = None
                    distance = None

                if "e_plx" in result[0].keys():
                    parallax_error = result[0]["e_plx"][0]  # (mas)
                    distance_error = None
                    if ma.is_masked(parallax_error):
                        parallax_error = None

                elif "e_Plx" in result[0].keys():
                    parallax_error = result[0]["e_Plx"][0]  # (mas)
                    distance_error = None
                    if ma.is_masked(parallax_error):
                        parallax_error = None

                elif "e_Dist" in result[0].keys():
                    distance_error = result[0]["e_Dist"][0]  # (pc)
                    parallax_error = None
                    if ma.is_masked(distance_error):
                        distance_error = None

                else:
                    parallax_error = None
                    distance_error = None

            if parallax is not None or distance is not None:
                break

    # query Gaia catalog
    # if ma.is_masked(parallax):
    #
    #     if simbad_result is not None:
    #         coord_ra = simbad_result["RA"][0]
    #         coord_dec = simbad_result["DEC"][0]
    #
    #         coord = SkyCoord(
    #             ra=coord_ra, dec=coord_dec, unit=(u.hourangle, u.deg), frame="icrs"
    #         )
    #
    #         result = Gaia.query_object(
    #             coordinate=coord, width=1.0 * u.arcsec, height=1.0 * u.arcsec
    #         )
    #
    #         if result:
    #             parallax = result["parallax"][0]  # (mas)

    if parallax is not None:
        distance = 1.0 / (parallax * 1e-3)  # (pc)

    if parallax is not None and parallax_error is not None:
        distance_minus = distance - 1.0 / ((parallax + parallax_error) * 1e-3)  # (pc)
        distance_plus = 1.0 / ((parallax - parallax_error) * 1e-3) - distance  # (pc)
        distance_error = (distance_plus + distance_minus) / 2.0  # (pc)

    if parallax is None:
        parallax = np.nan

    if parallax_error is None:
        parallax_error = np.nan

    if distance is None:
        distance = np.nan

    if distance_error is None:
        distance_error = np.nan

    if np.isnan(parallax) and np.isnan(distance):
        warnings.warn(f"No parallax was found for {target} so storing a NaN value.")

    return simbad_id, (distance, distance_error)
