import os
import shutil

import pytest
import numpy as np

import species
from species.util import test_util


class TestPlanck:

    def setup_class(self):
        self.limit = 1e-10
        self.test_path = os.path.dirname(__file__) + '/'

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_planck(self):
        read_planck = species.ReadPlanck(filter_name='MKO/NSFCam.J')
        assert read_planck.wavel_range == pytest.approx((1.1308, 1.3812), rel=self.limit, abs=0.)

        read_planck = species.ReadPlanck(wavel_range=(1., 5.))
        assert read_planck.wavel_range == (1., 5.)

    def test_get_spectrum(self):
        read_planck = species.ReadPlanck(filter_name='MKO/NSFCam.J')
        modelbox = read_planck.get_spectrum({'teff': 2000., 'radius': 1., 'distance': 10.}, 100.)

        assert modelbox.model == 'planck'
        assert modelbox.wavelength.shape == (42, )
        assert modelbox.flux.shape == (42, )

        assert np.sum(modelbox.wavelength) == pytest.approx(52.7026751397061, rel=self.limit, abs=0.)
        assert np.sum(modelbox.flux) == pytest.approx(8.332973825446955e-13, rel=self.limit, abs=0.)

    def test_get_flux(self):
        read_planck = species.ReadPlanck(filter_name='MKO/NSFCam.J')

        flux = read_planck.get_flux({'teff': 2000., 'radius': 1., 'distance': 10.})
        assert flux[0] == pytest.approx(1.9888949873704357e-14, rel=self.limit, abs=0.)

        synphot = species.SyntheticPhotometry(filter_name='MKO/NSFCam.J')
        flux = read_planck.get_flux({'teff': 2000., 'radius': 1., 'distance': 10.}, synphot=synphot)
        assert flux[0] == pytest.approx(1.9888949873704357e-14, rel=self.limit, abs=0.)
