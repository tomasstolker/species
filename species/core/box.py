"""
Box module.
"""

import sys


def create_box(boxtype,
               **kwargs):
    """
    Returns
    -------
    species.core.box
    """

    if boxtype == 'colormag':
        box = ColorMagBox()
        box.library = kwargs['library']
        box.object_type = kwargs['object_type']
        box.filters_color = kwargs['filters_color']
        box.filter_mag = kwargs['filter_mag']
        box.color = kwargs['color']
        box.magnitude = kwargs['magnitude']
        box.sptype = kwargs['sptype']

    if boxtype == 'colorcolor':
        box = ColorColorBox()
        box.library = kwargs['library']
        box.object_type = kwargs['object_type']
        box.filters = kwargs['filters']
        box.color1 = kwargs['color1']
        box.color2 = kwargs['color2']
        box.sptype = kwargs['sptype']

    elif boxtype == 'isochrone':
        box = IsochroneBox()
        box.model = kwargs['model']
        box.filters_color = kwargs['filters_color']
        box.filter_mag = kwargs['filter_mag']
        box.color = kwargs['color']
        box.magnitude = kwargs['magnitude']
        box.teff = kwargs['teff']
        box.logg = kwargs['logg']
        box.mass = kwargs['mass']

    elif boxtype == 'model':
        box = ModelBox()
        box.model = kwargs['model']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']
        box.parameters = kwargs['parameters']

    elif boxtype == 'object':
        box = ObjectBox()
        box.name = kwargs['name']
        box.filter = kwargs['filter']
        box.magnitude = kwargs['magnitude']
        box.flux = kwargs['flux']
        box.distance = kwargs['distance']
        box.spectrum = kwargs['spectrum']

    elif boxtype == 'photometry':
        box = PhotometryBox()
        box.name = kwargs['name']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']

    elif boxtype == 'residuals':
        box = ResidualsBox()
        box.name = kwargs['name']
        box.photometry = kwargs['photometry']
        box.spectrum = kwargs['spectrum']

    elif boxtype == 'samples':
        box = SamplesBox()
        box.spectrum = kwargs['spectrum']
        box.parameters = kwargs['parameters']
        box.samples = kwargs['samples']
        box.best_sample = kwargs['best_sample']

    elif boxtype == 'spectrum':
        box = SpectrumBox()
        box.spectrum = kwargs['spectrum']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']
        box.error = kwargs['error']
        box.name = kwargs['name']
        box.simbad = kwargs['simbad']
        box.sptype = kwargs['sptype']
        box.distance = kwargs['distance']

    elif boxtype == 'synphot':
        box = SynphotBox()
        box.name = kwargs['name']
        box.flux = kwargs['flux']

    return box


class Box:
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

    def open_box(self):
        """
        Returns
        -------
        None
        """

        sys.stdout.write(f'Opening {type(self).__name__}...\n')
        sys.stdout.flush()

        for item in self.__dict__.keys():
            sys.stdout.write(str(item)+' = '+str(self.__dict__[item])+'\n')
            sys.stdout.flush()


class ColorMagBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.library = None
        self.object_type = None
        self.filters_color = None
        self.filter_mag = None
        self.color = None
        self.magnitude = None
        self.sptype = None


class ColorColorBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.library = None
        self.object_type = None
        self.filters = None
        self.color1 = None
        self.color2 = None
        self.sptype = None


class IsochroneBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.model = None
        self.filters_color = None
        self.filter_mag = None
        self.color = None
        self.magnitude = None
        self.teff = None
        self.logg = None
        self.mass = None


class ModelBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.model = None
        self.type = None
        self.wavelength = None
        self.flux = None
        self.parameters = None


class ObjectBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.name = None
        self.filter = None
        self.magnitude = None
        self.flux = None
        self.distance = None
        self.spectrum = None


class PhotometryBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.name = None
        self.wavelength = None
        self.flux = None


class ResidualsBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.name = None
        self.photometry = None
        self.spectrum = None


class SamplesBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.spectrum = None
        self.parameters = None
        self.samples = None
        self.best_sample = None


class SpectrumBox(Box):
    """
    Text
    """

    def __init__(self):
        """
        Returns
        -------
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
    Text
    """

    def __init__(self):
        """
        Returns
        -------
        None
        """

        self.name = None
        self.flux = None
