"""
Box module.
"""

import numpy as np


def open_box(box):
    """
    return:
    """

    for item in box.__dict__.keys():
        print(item)


def create_box(boxtype, **kwargs):
    """
    :return:
    """

    if boxtype == "spectrum":
        box = SpectrumBox()
        box.name = kwargs['name']
        box.wavelength = np.asarray([kwargs['wavelength']])
        box.flux = np.asarray([kwargs['flux']])

    elif boxtype == "photometry":
        box = PhotometryBox()
        box.name = kwargs['name']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']

    elif boxtype == "samples":
        box = SamplesBox()
        box.model = kwargs['model']
        box.parameters = kwargs['parameters']
        box.samples = kwargs['samples']

    return box


class SpectrumBox:
    """
    Text
    """

    def __init__(self):
        """
        :return:
        """

        self.spectrum = None
        self.wavelength = None
        self.flux = None
        self.name = None
        self.simbad = None
        self.sptype = None
        self.distance = None


class ModelBox:
    """
    Text
    """

    def __init__(self):
        """
        :return:
        """

        self.model = None
        self.wavelength = None
        self.flux = None
        self.teff = None
        self.logg = None
        self.feh = None


class PhotometryBox:
    """
    Text
    """

    def __init__(self):
        """
        :return:
        """

        self.name = None
        self.wavelength = None
        self.flux = None


class SamplesBox:
    """
    Text
    """

    def __init__(self):
        """
        :return:
        """

        self.model = None
        self.parameters = None
        self.samples = None
