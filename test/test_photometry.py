import os
import shutil

import numpy as np

from species import SpeciesInit, SyntheticPhotometry


class TestPhotometry(object):

    def setup_class(self):
        self.limit = 1e-10

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree("data/")

    def test_species_init(self):
        SpeciesInit("./", "./data")

    def test_synthetic_photometry(self):
        SyntheticPhotometry("MKO/NSFCam.J")

    def test_magnitude_to_flux(self):
        synphot = SyntheticPhotometry("MKO/NSFCam.J")
        flux, error = synphot.magnitude_to_flux(20., 0.5)

        assert np.allclose(flux, 3.104539116258e-17, rtol=self.limit, atol=0.)
        assert np.allclose(error, (1.1457073596289663e-17, 1.815823794828546e-17), rtol=self.limit, atol=0.)

    def test_flux_to_magnitude(self):
        synphot = SyntheticPhotometry("MKO/NSFCam.J")
        app_mag, abs_mag = synphot.flux_to_magnitude(1e-10, 50.)

        assert np.allclose(app_mag, 20.0, rtol=self.limit, atol=0.)
        assert np.allclose(abs_mag, 16.505149978319906, rtol=self.limit, atol=0.)
