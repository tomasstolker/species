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

        assert np.allclose(flux, 3.1045460194170406e-17, rtol=self.limit, atol=0.)
        assert np.allclose(error, 1.4807688698141927e-17, rtol=self.limit, atol=0.)

    def test_flux_to_magnitude(self):
        synphot = species.SyntheticPhotometry("MKO/NSFCam.J")
        app_mag, abs_mag = synphot.flux_to_magnitude(1e-10, 50.)

        assert np.allclose(app_mag, 3.729995254697838, rtol=self.limit, atol=0.)
        assert np.allclose(abs_mag, 0.23514523301774481, rtol=self.limit, atol=0.)
