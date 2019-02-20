import os
import shutil

import numpy as np

import species
from . import tools


class TestPhotometry:

    def setup_class(self):
        self.limit = 1e-10

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree("data/")

    def test_species_init(self):
        tools.create_config()
        species.SpeciesInit("./")

    def test_synthetic_photometry(self):
        species.SyntheticPhotometry("MKO/NSFCam.J")

    def test_magnitude_to_flux(self):
        synphot = species.SyntheticPhotometry("MKO/NSFCam.J")
        flux, error = synphot.magnitude_to_flux(20., 0.5)

        assert np.allclose(flux, 3.104539116258e-17, rtol=self.limit, atol=0.)
        assert np.allclose(error, 1.4807655772287564e-17, rtol=self.limit, atol=0.)

    def test_flux_to_magnitude(self):
        synphot = species.SyntheticPhotometry("MKO/NSFCam.J")
        app_mag, abs_mag = synphot.flux_to_magnitude(1e-10, 50.)

        assert np.allclose(app_mag, 3.72999284049042, rtol=self.limit, atol=0.)
        assert np.allclose(abs_mag, 0.23514281881032595, rtol=self.limit, atol=0.)
