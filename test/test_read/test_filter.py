import os
import shutil

import pytest
import numpy as np

import species
from species.util import test_util


class TestFilter:

    def setup_class(self):
        self.limit = 1e-8
        self.test_path = os.path.dirname(__file__) + '/'

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_filter(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')

        assert read_filter.filter_name == 'MKO/NSFCam.H'

    def test_get_filter(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        filter_profile = read_filter.get_filter()

        assert filter_profile.shape == (970, 2)
        assert np.sum(filter_profile) == pytest.approx(2089.2432, rel=1e-6, abs=0.)

    def test_detector_type(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        det_type = read_filter.detector_type()

        assert det_type == 'energy'

    def test_get_filter_photon_counter(self):
        read_filter = species.ReadFilter('Keck/NIRC2.J')
        filter_profile = read_filter.get_filter()

        assert filter_profile.shape == (1054, 2)
        assert np.sum(filter_profile) == pytest.approx(2038.0486, rel=1e-6, abs=0.)

    def test_detector_type_photon(self):
        read_filter = species.ReadFilter('Keck/NIRC2.J')
        det_type = read_filter.detector_type()

        assert det_type == 'photon'

    def test_interpolate_filter(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        interp_filter = read_filter.interpolate_filter()

        assert interp_filter.x.shape == (970, )
        assert interp_filter.y.shape == (970, )

        assert np.sum(interp_filter.x) == pytest.approx(1575.1079, rel=1e-8, abs=0.)
        assert np.sum(interp_filter.y) == pytest.approx(514.135186, rel=1e-6, abs=0.)

    def test_wavelength_range(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        min_wavel, max_wavel = read_filter.wavelength_range()

        assert min_wavel == pytest.approx(1.382, rel=1e-6, abs=0.)
        assert max_wavel == pytest.approx(1.8656, rel=1e-8, abs=0.)

    def test_mean_wavelength(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        mean_wavel = read_filter.mean_wavelength()

        assert mean_wavel == pytest.approx(1.6298258, rel=1e-6, abs=0.)

    def test_filter_fwhm(self):
        read_filter = species.ReadFilter('MKO/NSFCam.H')
        filter_fwhm = read_filter.filter_fwhm()

        assert filter_fwhm == pytest.approx(0.2956945805173905, rel=self.limit, abs=0.)
