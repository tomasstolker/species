import os
import shutil
import urllib.request

import pytest
import numpy as np

import species
from species.util import test_util


class TestPhotometry:

    def setup_class(self):
        self.limit = 1e-8

        url = 'http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/Data/plnt_Jupiter.txt'
        urllib.request.urlretrieve(url, 'plnt_Jupiter.txt')

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        os.remove('plnt_Jupiter.txt')
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

    def test_spectrum_to_flux(self):
        jup_wavel, jup_flux, jup_err = np.loadtxt('plnt_Jupiter.txt', unpack=True)

        synphot = species.SyntheticPhotometry('MKO/NSFCam.J')

        phot_flux, phot_error = synphot.spectrum_to_flux(jup_wavel,
                                                         jup_flux,
                                                         error=jup_err,
                                                         threshold=None)

        assert phot_flux == pytest.approx(1.802998152236653e-09, rel=self.limit, abs=0.)

        # The error is estimated with Monte Carlo sampling
        assert phot_error == pytest.approx(8.8e-14, rel=0., abs=2e-14)

    def test_spectrum_to_flux_no_error(self):
        jup_wavel, jup_flux, _ = np.loadtxt('plnt_Jupiter.txt', unpack=True)

        synphot = species.SyntheticPhotometry('MKO/NSFCam.J')

        phot_flux, phot_error = synphot.spectrum_to_flux(jup_wavel,
                                                         jup_flux,
                                                         error=None,
                                                         threshold=None)

        assert phot_flux == pytest.approx(1.802998152236653e-09, rel=self.limit, abs=0.)
        assert phot_error is None

    def test_spectrum_to_flux_threshold(self):
        jup_wavel, jup_flux, _ = np.loadtxt('plnt_Jupiter.txt', unpack=True)

        synphot = species.SyntheticPhotometry('MKO/NSFCam.J')

        phot_flux, phot_error = synphot.spectrum_to_flux(jup_wavel,
                                                         jup_flux,
                                                         error=None,
                                                         threshold=0.05)

        assert phot_flux == pytest.approx(1.802998152236653e-09, rel=self.limit, abs=0.)
        assert phot_error is None

    def test_spectrum_to_flux_photon_detector(self):
        jup_wavel, jup_flux, jup_err = np.loadtxt('plnt_Jupiter.txt', unpack=True)

        synphot = species.SyntheticPhotometry('Keck/NIRC2.J')

        phot_flux, phot_error = synphot.spectrum_to_flux(jup_wavel,
                                                         jup_flux,
                                                         error=jup_err,
                                                         threshold=None)

        assert phot_flux == pytest.approx(1.8139884828774647e-09, rel=self.limit, abs=0.)

        # The error is estimated with Monte Carlo sampling
        assert phot_error == pytest.approx(8.4e-14, rel=0., abs=2e-14)

    def test_spectrum_to_magnitude(self):
        jup_wavel, jup_flux, jup_err = np.loadtxt('plnt_Jupiter.txt', unpack=True)

        synphot = species.SyntheticPhotometry('MKO/NSFCam.J')

        app_mag, abs_mag = synphot.spectrum_to_magnitude(jup_wavel,
                                                         jup_flux,
                                                         error=jup_err,
                                                         distance=(1., 0.01),
                                                         threshold=None)

        assert app_mag[0] == pytest.approx(0.5900070089410099, rel=self.limit, abs=0.)
        assert abs_mag[0] == pytest.approx(5.59000700894101, rel=self.limit, abs=0.)

        # The error is estimated with Monte Carlo sampling
        assert app_mag[1] == pytest.approx(5.368048545366946e-05, rel=0., abs=1e-5)
        assert abs_mag[1] == pytest.approx(0.021714790446227043, rel=0., abs=1e-2)
