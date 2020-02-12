import os
import shutil

import pytest
import numpy as np

import species
from species.util import test_util


class TestSpectrum:

    def setup_class(self):
        self.limit = 1e-10
        self.test_path = os.path.dirname(__file__) + '/'

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        os.remove('spectrum.pdf')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_spectrum(self):
        database = species.Database()
        database.add_spectrum('irtf', sptypes=['L', ])

        read_spectrum = species.ReadSpectrum('irtf', filter_name='MKO/NSFCam.H')
        assert read_spectrum.wavel_range == pytest.approx((1.382, 1.8656), rel=self.limit, abs=0.)

    def test_get_spectrum(self):
        read_spectrum = species.ReadSpectrum('irtf', filter_name='MKO/NSFCam.H')
        spec_box = read_spectrum.get_spectrum(sptypes=['L', ], exclude_nan=True)

        assert spec_box.wavelength[0].shape == (1000, )
        assert spec_box.flux[0].shape == (1000, )

        assert np.sum(spec_box.wavelength[0]) == pytest.approx(1611.2432, rel=1e-7, abs=0.)
        assert np.sum(spec_box.flux[0]) == pytest.approx(2.3606193e-11, rel=1e-7, abs=0.)

        species.plot_spectrum(boxes=[spec_box, ],
                              filters=['MKO/NSFCam.H', ],
                              output='spectrum.pdf',
                              xlim=(1., 2.5),
                              offset=(-0.08, -0.06))

        assert os.path.exists('spectrum.pdf')
