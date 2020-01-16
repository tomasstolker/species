import os
import shutil
import pytest

import numpy as np

import species
from species.util import test_util


class TestFilter:

    def setup_class(self):
        self.limit = 1e-10
        self.test_path = os.path.dirname(__file__) + '/'

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit('./')

    def test_read_filter(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')

        assert read_filter.filter_name == 'MKO/NSFCam.H'

    def test_get_filter(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        filter_profile = read_filter.get_filter()

        assert filter_profile.shape == (2, 970)
        assert np.allclose(np.sum(filter_profile), 2089.2432, rtol=1e-7, atol=0.)

    def test_interpolate_filter(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        interp_filter = read_filter.interpolate_filter()

        assert interp_filter.x.shape == (970, )
        assert interp_filter.y.shape == (970, )

        assert np.allclose(np.sum(interp_filter.x), 1575.1079, rtol=1e-7, atol=0.)
        assert np.allclose(np.sum(interp_filter.y), 514.1352, rtol=1e-7, atol=0.)

    def test_wavelength_range(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        min_wavel, max_wavel = read_filter.wavelength_range()

        assert min_wavel == pytest.approx(1.382)
        assert max_wavel == pytest.approx(1.8656)

    def test_mean_wavelength(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        mean_wavel = read_filter.mean_wavelength()

        assert mean_wavel == pytest.approx(1.6298258)

    def test_filter_fwhm(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        filter_fwhm = read_filter.filter_fwhm()

        assert filter_fwhm == 0.2956945805173905
