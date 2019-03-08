'''
Box module.
'''


def open_box(box):
    '''
    return:
    '''

    for item in box.__dict__.keys():
        sys.stdout.write(str(item)+' = '+str(box.__dict__[item])+'\n')
        sys.stdout.flush()

def create_box(boxtype, **kwargs):
    '''
    :return:
    '''

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

    elif boxtype == 'photometry':
        box = PhotometryBox()
        box.name = kwargs['name']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']

    elif boxtype == 'samples':
        box = SamplesBox()
        box.spectrum = kwargs['spectrum']
        box.parameters = kwargs['parameters']
        box.samples = kwargs['samples']
        box.chisquare = kwargs['chisquare']

    elif boxtype == 'spectrum':
        box = SpectrumBox()
        box.spectrum = kwargs['spectrum']
        box.wavelength = kwargs['wavelength']
        box.flux = kwargs['flux']
        box.name = kwargs['name']
        box.simbad = kwargs['simbad']
        box.sptype = kwargs['sptype']
        box.distance = kwargs['distance']

    elif boxtype == 'synphot':
        box = SynphotBox()
        box.name = kwargs['name']
        box.flux = kwargs['flux']

    return box


class ColorMagBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.library = None
        self.object_type = None
        self.filters_color = None
        self.filter_mag = None
        self.color = None
        self.magnitude = None
        self.sptype = None


class ColorColorBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.library = None
        self.object_type = None
        self.filters = None
        self.color1 = None
        self.color2 = None
        self.sptype = None


class ModelBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.model = None
        self.type = None
        self.wavelength = None
        self.flux = None
        self.parameters = None


class ObjectBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.name = None
        self.filter = None
        self.magnitude = None
        self.flux = None
        self.distance = None


class PhotometryBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.name = None
        self.wavelength = None
        self.flux = None


class SamplesBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.spectrum = None
        self.parameters = None
        self.samples = None
        self.chisquare = None


class SpectrumBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.spectrum = None
        self.wavelength = None
        self.flux = None
        self.name = None
        self.simbad = None
        self.sptype = None
        self.distance = None


class SynphotBox:
    '''
    Text
    '''

    def __init__(self):
        '''
        :return:
        '''

        self.name = None
        self.flux = None
