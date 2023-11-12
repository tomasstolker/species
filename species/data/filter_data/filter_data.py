"""
Module for downloading filter data from the website
of the SVO Filter Profile Service.
"""

import os
import urllib.request
import warnings

from typing import Optional, Tuple

import h5py
import numpy as np

from astropy.io.votable import parse_single_table
from typeguard import typechecked


@typechecked
def download_filter(
    filter_id: str,
    input_path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Function for downloading filter profile data
    from the SVO Filter Profile Service.

    Parameters
    ----------
    filter_id : str
        Filter name as listed on the website of the `SVO
        Filter Profile Service
        <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_.
    input_path : str
        Folder where the data is located.

    Returns
    -------
    np.ndarray
        Wavelength (um).
    np.ndarray
        Fractional transmission.
    str
        Detector type ('energy' or 'photon').
    """

    if filter_id == "Magellan/VisAO.rp":
        url = "https://xwcl.science/magao/visao/VisAO_rp_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_rp_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Magellan/VisAO.ip":
        url = "https://xwcl.science/magao/visao/VisAO_ip_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_ip_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Magellan/VisAO.zp":
        url = "https://xwcl.science/magao/visao/VisAO_zp_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_zp_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Keck/NIRC2.NB_4.05":
        # The filter profile of Br_alpha has been digitized from
        # https://www2.keck.hawaii.edu/inst/nirc2/filters.html

        url = "https://home.strw.leidenuniv.nl/~stolker/species/Keck_NIRC2.NB_4.05.dat"
        filter_path = os.path.join(input_path, "Keck_NIRC2.NB_4.05.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Keck/NIRC.Y":
        # The filter profile of the Y band has been
        # adopted from Hillenbrand et al. (2002)

        url = "https://home.strw.leidenuniv.nl/~stolker/species/Keck_NIRC.Y.dat"
        filter_path = os.path.join(input_path, "Keck_NIRC.Y.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id in ["LCO/VisAO.Ys", "Magellan/VisAO.Ys"]:
        url = "https://xwcl.science/magao/visao/VisAO_Ys_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_Ys_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        # Remove wavelengths with zero transmission
        wavelength = wavelength[:-7]
        transmission = transmission[:-7]

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ALMA/band6":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/alma_band6.dat"
        filter_path = os.path.join(input_path, "alma_band6.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ALMA/band7":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/alma_band7.dat"
        filter_path = os.path.join(input_path, "alma_band7.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    else:
        url = "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=" + filter_id
        filter_path = os.path.join(input_path, "filter.xml")
        urllib.request.urlretrieve(url, filter_path)

        try:
            table = parse_single_table(filter_path)

            wavelength = table.array["Wavelength"]
            transmission = table.array["Transmission"]

        except IndexError:
            wavelength = None
            transmission = None
            det_type = None

            warnings.warn(
                f"Filter '{filter_id}' is not available "
                "on the SVO Filter Profile Service."
            )

        except:
            os.remove(filter_path)

            raise ValueError(
                f"The filter data of '{filter_id}' could not "
                "be downloaded. Perhaps the website of the "
                "SVO Filter Profile Service (http://svo2.cab."
                "inta-csic.es/svo/theory/fps/) is not available?"
            )

        if transmission is not None:
            det_type = table.get_field_by_id("DetectorType").value

            # For backward compatibility
            if not isinstance(det_type, str):
                det_type = det_type.decode("utf-8")

            if int(det_type) == 0:
                det_type = "energy"

            elif int(det_type) == 1:
                det_type = "photon"

            else:
                det_type = "photon"

                warnings.warn(
                    f"Detector type, '{det_type}', not "
                    "recognized. Setting detector "
                    "type to photon-counting detector."
                )

            wavelength *= 1e-4  # (um)

        os.remove(filter_path)

    if wavelength is not None:
        indices = []

        for i in range(transmission.size):
            if i == 0 and transmission[i] == 0.0 and transmission[i + 1] == 0.0:
                indices.append(i)

            elif (
                i == transmission.size - 1
                and transmission[i - 1] == 0.0
                and transmission[i] == 0.0
            ):
                indices.append(i)

            elif (
                transmission[i - 1] == 0.0
                and transmission[i] == 0.0
                and transmission[i + 1] == 0.0
            ):
                indices.append(i)

        wavelength = np.delete(wavelength, indices)
        transmission = np.delete(transmission, indices)

        if np.amin(transmission) < 0.0:
            warnings.warn(
                f"The minimum transmission value of {filter_id} is "
                f"smaller than zero ({np.amin(transmission):.2e}). "
                f"Wavelengths with negative transmission "
                f"values will be removed."
            )

            indices = []

            for i, item in enumerate(transmission):
                if item > 0.0:
                    indices.append(i)

            wavelength = wavelength[indices]
            transmission = transmission[indices]

    return wavelength, transmission, det_type


@typechecked
def add_filter_profile(
    input_path: str, database: h5py._hl.files.File, filter_name: str
) -> None:
    """
    Function for downloading and adding a filter profile
    to the HDF5 database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    filter_name : str
        Filter name from the SVO Filter Profile Service (e.g.,
        'Paranal/NACO.Lp') or a user-defined name if a ``filename``
        is specified.

    Returns
    -------
    None
        NoneType
    """

    wavelength, transmission, detector_type = download_filter(filter_name, input_path)

    if wavelength is not None and transmission is not None:
        wavel_new = [wavelength[0]]
        transm_new = [transmission[0]]

        for i in range(wavelength.size - 1):
            if wavelength[i + 1] > wavel_new[-1]:
                # Required for the issue with the Keck/NIRC2.J filter on SVO
                wavel_new.append(wavelength[i + 1])
                transm_new.append(transmission[i + 1])

        dset = database.create_dataset(
            f"filters/{filter_name}", data=np.column_stack((wavel_new, transm_new))
        )

        dset.attrs["det_type"] = str(detector_type)
