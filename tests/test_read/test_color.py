import os
import shutil

import pytest
import numpy as np

import species
from species.util import test_util


class TestColor:

    def setup_class(self):
        self.limit = 1e-8

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_color_magnitude(self):
        database = species.Database()
        database.add_photometry('vlm-plx')
        database.add_photometry('leggett')

        read_colormag = species.ReadColorMagnitude('vlm-plx',
                                                   ('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                                   'MKO/NSFCam.J')

        assert read_colormag.filters_color == ('MKO/NSFCam.J', 'MKO/NSFCam.H')
        assert read_colormag.filter_mag == 'MKO/NSFCam.J'

    def test_get_color_magnitude(self):
        read_colormag = species.ReadColorMagnitude('leggett',
                                                   ('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                                   'MKO/NSFCam.J')

        colormag_box = read_colormag.get_color_magnitude(object_type=None)
        colormag_box.open_box()

        assert np.nansum(colormag_box.color) == pytest.approx(14.05)
        assert np.nansum(colormag_box.magnitude) == pytest.approx(1519.373)

    def test_read_color_color(self):
        read_colorcolor = species.ReadColorColor('vlm-plx',
                                                 (('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                                  ('MKO/NSFCam.H', 'MKO/NSFCam.K')))

        assert read_colorcolor.filters_colors == (('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                                  ('MKO/NSFCam.H', 'MKO/NSFCam.K'))

    def test_get_color_color(self):
        read_colorcolor = species.ReadColorColor('vlm-plx',
                                                 (('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                                  ('MKO/NSFCam.H', 'MKO/NSFCam.K')))

        colorcolor_box = read_colorcolor.get_color_color(object_type=None)

        assert np.nansum(colorcolor_box.color1) == pytest.approx(166.99384)
        assert np.nansum(colorcolor_box.color2) == pytest.approx(138.5127)
