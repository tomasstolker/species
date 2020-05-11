import os
import shutil

import pytest

import species
from species.util import test_util


class TestPhotometry:

    def setup_class(self):
        self.limit = 1e-10

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_synthetic_photometry(self):
        species.SyntheticPhotometry('MKO/NSFCam.J')

    def test_magnitude_to_flux(self):
        synphot = species.SyntheticPhotometry('MKO/NSFCam.J')
        flux, error = synphot.magnitude_to_flux(20., error=0.5)

        assert flux == pytest.approx(3.104545900342411e-17, rel=self.limit, abs=0.)
        assert error == pytest.approx(1.4807688130194138e-17, rel=self.limit, abs=0.)

    def test_flux_to_magnitude(self):
        synphot = species.SyntheticPhotometry('MKO/NSFCam.J')
        app_mag, abs_mag = synphot.flux_to_magnitude(1e-10, error=None, distance=(50., None))

        assert app_mag[0] == pytest.approx(3.729995213054507, rel=self.limit, abs=0.)
        assert abs_mag[0] == pytest.approx(0.23514519137441336, rel=self.limit, abs=0.)
