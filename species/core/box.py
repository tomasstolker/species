"""
Box module.
"""

import numpy as np


def open_box(box):
    """
    return:
    """

    for item in box.__dict__.keys():
        print(item, '=', box.__dict__[item], '\n')


def create_box(boxtype, **kwargs):
    """
    :return:
    """

    if boxtype == 'model':
        box = ModelBox()
        box.model = kwargs['model']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']
        box.parameters = kwargs['parameters']

    elif boxtype == 'photometry':
        box = PhotometryBox()
        box.name = kwargs['name']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']

    elif boxtype == 'object':
        box = ObjectBox()
        box.name = kwargs['name']
        box.filter = kwargs['filter']
        box.magnitude = kwargs['magnitude']
        box.flux = kwargs['flux']
        box.distance = kwargs['distance']

    elif boxtype == 'samples':
        box = SamplesBox()
        box.model = kwargs['model']
        box.parameters = kwargs['parameters']
        box.samples = kwargs['samples']
        box.chisquare = kwargs['chisquare']

    elif boxtype == 'spectrum':
        box = SpectrumBox()
        box.name = kwargs['name']
        box.wavelength = np.asarray([kwargs['wavelength']])
        box.flux = np.asarray([kwargs['flux']])

    elif boxtype == 'synphot':
        box = SynphotBox()
        box.name = kwargs['name']
        box.flux = kwargs['flux']

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
        self.type = None
        self.wavelength = None
        self.flux = None
        self.parameters = None


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
        self.chisquare = None

class ObjectBox:
    """
    Text
    """

    def __init__(self):
        """
        :return:
        """

        self.name = None
        self.filter = None
        self.magnitude = None
        self.flux = None
        self.distance = None

class SynphotBox:
    """
    Text
    """

    def __init__(self):
        """
        :return:
        """

        self.name = None
        self.flux = None
