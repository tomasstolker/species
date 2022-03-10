"""
Module with classes for storing data in `Box` objects.
"""

from typing import List, Union

import numpy as np
import spectres

import species

from species.analysis import photometry
from species.read import read_filter


def create_box(boxtype, **kwargs):
    """
    Function for creating a :class:`~species.core.box.Box`.

    Returns
    -------
    species.core.box
        Box with the data and parameters.
    """

    if boxtype == "colormag":
        box = ColorMagBox()
        box.library = kwargs["library"]
        box.object_type = kwargs["object_type"]
        box.filters_color = kwargs["filters_color"]
        box.filter_mag = kwargs["filter_mag"]
        box.color = kwargs["color"]
        box.magnitude = kwargs["magnitude"]
        box.names = kwargs["names"]
        if "sptype" in kwargs:
            box.sptype = kwargs["sptype"]
        if "mass" in kwargs:
            box.mass = kwargs["mass"]
        if "radius" in kwargs:
            box.radius = kwargs["radius"]

    if boxtype == "colorcolor":
        box = ColorColorBox()
        box.library = kwargs["library"]
        box.object_type = kwargs["object_type"]
        box.filters = kwargs["filters"]
        box.color1 = kwargs["color1"]
        box.color2 = kwargs["color2"]
        box.names = kwargs["names"]
        if "sptype" in kwargs:
            box.sptype = kwargs["sptype"]
        if "mass" in kwargs:
            box.mass = kwargs["mass"]
        if "radius" in kwargs:
            box.radius = kwargs["radius"]

    elif boxtype == "isochrone":
        box = IsochroneBox()
        box.model = kwargs["model"]
        box.filters_color = kwargs["filters_color"]
        box.filter_mag = kwargs["filter_mag"]
        box.color = kwargs["color"]
        box.magnitude = kwargs["magnitude"]
        box.teff = kwargs["teff"]
        box.logg = kwargs["logg"]
        box.masses = kwargs["masses"]

    elif boxtype == "model":
        box = ModelBox()
        box.model = kwargs["model"]
        box.wavelength = kwargs["wavelength"]
        box.flux = kwargs["flux"]
        box.parameters = kwargs["parameters"]
        box.quantity = kwargs["quantity"]
        if "contribution" in kwargs:
            box.contribution = kwargs["contribution"]
        if "bol_flux" in kwargs:
            box.bol_flux = kwargs["bol_flux"]

    elif boxtype == "object":
        box = ObjectBox()
        box.name = kwargs["name"]
        box.filters = kwargs["filters"]
        box.mean_wavel = kwargs["mean_wavel"]
        box.magnitude = kwargs["magnitude"]
        box.flux = kwargs["flux"]
        box.distance = kwargs["distance"]
        box.spectrum = kwargs["spectrum"]

    elif boxtype == "photometry":
        box = PhotometryBox()
        if "name" in kwargs:
            box.name = kwargs["name"]
        if "sptype" in kwargs:
            box.sptype = kwargs["sptype"]
        if "wavelength" in kwargs:
            box.wavelength = kwargs["wavelength"]
        if "flux" in kwargs:
            box.flux = kwargs["flux"]
        if "app_mag" in kwargs:
            box.app_mag = kwargs["app_mag"]
        if "abs_mag" in kwargs:
            box.abs_mag = kwargs["abs_mag"]
        if "filter_name" in kwargs:
            box.filter_name = kwargs["filter_name"]

    elif boxtype == "residuals":
        box = ResidualsBox()
        box.name = kwargs["name"]
        box.photometry = kwargs["photometry"]
        box.spectrum = kwargs["spectrum"]
        if "chi2_red" in kwargs:
            box.chi2_red = kwargs["chi2_red"]

    elif boxtype == "samples":
        box = SamplesBox()
        box.spectrum = kwargs["spectrum"]
        box.parameters = kwargs["parameters"]
        box.samples = kwargs["samples"]
        box.ln_prob = kwargs["ln_prob"]
        box.ln_evidence = kwargs["ln_evidence"]
        box.prob_sample = kwargs["prob_sample"]
        box.median_sample = kwargs["median_sample"]
        box.attributes = kwargs["attributes"]

    elif boxtype == "spectrum":
        box = SpectrumBox()
        box.spectrum = kwargs["spectrum"]
        box.wavelength = kwargs["wavelength"]
        box.flux = kwargs["flux"]
        box.error = kwargs["error"]
        box.name = kwargs["name"]
        if "simbad" in kwargs:
            box.simbad = kwargs["simbad"]
        if "sptype" in kwargs:
            box.sptype = kwargs["sptype"]
        if "distance" in kwargs:
            box.distance = kwargs["distance"]

    elif boxtype == "synphot":
        box = SynphotBox()
        box.name = kwargs["name"]
        box.flux = kwargs["flux"]

    return box


class Box:
    """
    Class for generic methods that can be applied on all `Box` object.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

    def open_box(self):
        """
        Method for inspecting the content of a `Box`.

        Returns
        -------
        NoneType
            None
        """

        print(f"Opening {type(self).__name__}...")

        for key, value in self.__dict__.items():
            print(f"{key} = {value}")


class ColorMagBox(Box):
    """
    Class for storing color-magnitude data in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.library = None
        self.object_type = None
        self.filters_color = None
        self.filter_mag = None
        self.color = None
        self.magnitude = None
        self.names = None
        self.sptype = None
        self.mass = None
        self.radius = None


class ColorColorBox(Box):
    """
    Class for storing color-color data in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.library = None
        self.object_type = None
        self.filters = None
        self.color1 = None
        self.color2 = None
        self.names = None
        self.sptype = None
        self.mass = None
        self.radius = None


class IsochroneBox(Box):
    """
    Class for storing isochrone data in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.model = None
        self.filters_color = None
        self.filter_mag = None
        self.color = None
        self.magnitude = None
        self.teff = None
        self.logg = None
        self.masses = None


class PhotometryBox(Box):
    """
    Class for storing photometric data in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.name = None
        self.sptype = None
        self.wavelength = None
        self.flux = None
        self.app_mag = None
        self.abs_mag = None
        self.filter_name = None


class ModelBox(Box):
    """
    Class for storing a model spectrum in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.model = None
        self.type = None
        self.wavelength = None
        self.flux = None
        self.parameters = None
        self.quantity = None
        self.contribution = None
        self.bol_flux = None

    def smooth_spectrum(self, spec_res: float) -> None:
        """
        Method for smoothing the spectrum with a Gaussian kernel to the
        instrument resolution. The method is best applied on an input
        spectrum with a logarithmic wavelength sampling (i.e. constant
        spectral resolution). Alternatively, the wavelength sampling
        may be linear, but the smoothing is slower in that case.

        Parameters
        ----------
        spec_res : float
            Spectral resolution that is used for the smoothing kernel.

        Returns
        -------
        NoneType
            None
        """

        self.flux = species.util.read_util.smooth_spectrum(
            self.wavelength, self.flux, spec_res
        )

    def resample_spectrum(self, wavel_resample: np.ndarray) -> None:
        """
        Method for resampling the spectrum with ``spectres`` to a new
        wavelength grid.

        Parameters
        ----------
        wavel_resample : np.ndarray
            Wavelength points (um) to which the spectrum will be
            resampled.

        Returns
        -------
        NoneType
            None
        """

        self.flux = spectres.spectres(
            wavel_resample,
            self.wavelength,
            self.flux,
            spec_errs=None,
            fill=np.nan,
            verbose=True,
        )

        self.wavelength = wavel_resample

    def synthetic_photometry(self, filter_name: Union[str, List[str]]) -> PhotometryBox:
        """
        Method for calculating synthetic photometry from the model
        spectrum that is stored in the ``ModelBox``.

        Parameters
        ----------
        filter_name : str, list(str)
            Single filter name or a list of filter names for which
            synthetic photometry will be calculated.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry.
        """

        if isinstance(filter_name, str):
            filter_name = [filter_name]

        list_wavel = []
        list_flux = []
        list_app_mag = []
        list_abs_mag = []

        for item in filter_name:
            syn_phot = photometry.SyntheticPhotometry(filter_name=item)

            syn_flux = syn_phot.spectrum_to_flux(
                wavelength=self.wavelength, flux=self.flux
            )

            syn_mag = syn_phot.spectrum_to_magnitude(
                wavelength=self.wavelength, flux=self.flux
            )

            list_flux.append(syn_flux)
            list_app_mag.append(syn_mag[0])
            list_abs_mag.append(syn_mag[1])

            filter_profile = read_filter.ReadFilter(filter_name=item)
            list_wavel.append(filter_profile.mean_wavelength())

        phot_box = create_box(
            boxtype="photometry",
            name=None,
            sptype=None,
            wavelength=list_wavel,
            flux=list_flux,
            app_mag=list_app_mag,
            abs_mag=list_abs_mag,
            filter_name=filter_name,
        )

        return phot_box


class ObjectBox(Box):
    """
    Class for storing object data in a :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.name = None
        self.filters = None
        self.mean_wavel = None
        self.magnitude = None
        self.flux = None
        self.distance = None
        self.spectrum = None


class ResidualsBox(Box):
    """
    Class for storing best-fit residuals in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.name = None
        self.photometry = None
        self.spectrum = None
        self.chi2_red = None


class SamplesBox(Box):
    """
    Class for storing posterior samples in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.spectrum = None
        self.parameters = None
        self.samples = None
        self.ln_prob = None
        self.ln_evidence = None
        self.prob_sample = None
        self.median_sample = None
        self.attributes = None


class SpectrumBox(Box):
    """
    Class for storing spectral data in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.spectrum = None
        self.wavelength = None
        self.flux = None
        self.error = None
        self.name = None
        self.simbad = None
        self.sptype = None
        self.distance = None


class SynphotBox(Box):
    """
    Class for storing synthetic photometry in a
    :class:`~species.core.box.Box`.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        self.name = None
        self.flux = None
