import os
import shutil
import urllib.request

import pytest
import numpy as np

from species import SpeciesInit
from species.phot.syn_phot import SyntheticPhotometry
from species.util import test_util


class TestPhotometry:
    def setup_class(self):
        self.limit = 1e-8

        url = "http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/Data/plnt_Jupiter.txt"
        urllib.request.urlretrieve(url, "plnt_Jupiter.txt")

    def teardown_class(self):
        os.remove("species_database.hdf5")
        os.remove("species_config.ini")
        os.remove("plnt_Jupiter.txt")
        shutil.rmtree("data/")

    def test_species_init(self):
        test_util.create_config("./")
        SpeciesInit()

    def test_synthetic_photometry(self):
        SyntheticPhotometry("MKO/NSFCam.J")

    def test_magnitude_to_flux(self):
        synphot = SyntheticPhotometry("MKO/NSFCam.J")
        flux, error = synphot.magnitude_to_flux(20.0, error=0.5)

        assert flux == pytest.approx(3.066477264611711e-17, rel=self.limit, abs=0.0)
        assert error == pytest.approx(1.462611294865793e-17, rel=self.limit, abs=0.0)

    def test_flux_to_magnitude(self):
        synphot = SyntheticPhotometry("MKO/NSFCam.J")
        app_mag, abs_mag = synphot.flux_to_magnitude(
            1e-10, error=None, distance=(50.0, None)
        )

        assert app_mag[0] == pytest.approx(3.7165993727542115, rel=self.limit, abs=0.0)
        assert abs_mag[0] == pytest.approx(0.2217493510741182, rel=self.limit, abs=0.0)

    def test_spectrum_to_flux(self):
        jup_wavel, jup_flux, jup_err = np.loadtxt("plnt_Jupiter.txt", unpack=True)

        synphot = SyntheticPhotometry("MKO/NSFCam.J")

        phot_flux, phot_error = synphot.spectrum_to_flux(
            jup_wavel, jup_flux, error=jup_err, threshold=None
        )

        assert phot_flux == pytest.approx(
            1.802998152236653e-09, rel=self.limit, abs=0.0
        )

        # The error is estimated with Monte Carlo sampling
        assert phot_error == pytest.approx(8.8e-14, rel=0.0, abs=2e-14)

    def test_spectrum_to_flux_no_error(self):
        jup_wavel, jup_flux, _ = np.loadtxt("plnt_Jupiter.txt", unpack=True)

        synphot = SyntheticPhotometry("MKO/NSFCam.J")

        phot_flux, phot_error = synphot.spectrum_to_flux(
            jup_wavel, jup_flux, error=None, threshold=None
        )

        assert phot_flux == pytest.approx(
            1.802998152236653e-09, rel=self.limit, abs=0.0
        )
        assert phot_error is None

    def test_spectrum_to_flux_threshold(self):
        jup_wavel, jup_flux, _ = np.loadtxt("plnt_Jupiter.txt", unpack=True)

        synphot = SyntheticPhotometry("MKO/NSFCam.J")

        phot_flux, phot_error = synphot.spectrum_to_flux(
            jup_wavel, jup_flux, error=None, threshold=0.05
        )

        assert phot_flux == pytest.approx(
            1.802998152236653e-09, rel=self.limit, abs=0.0
        )
        assert phot_error is None

    def test_spectrum_to_flux_photon_detector(self):
        jup_wavel, jup_flux, jup_err = np.loadtxt("plnt_Jupiter.txt", unpack=True)

        synphot = SyntheticPhotometry("Keck/NIRC2.J")

        phot_flux, phot_error = synphot.spectrum_to_flux(
            jup_wavel, jup_flux, error=jup_err, threshold=None
        )

        assert phot_flux == pytest.approx(
            1.8139883721554032e-09, rel=self.limit, abs=0.0
        )

        # The error is estimated with Monte Carlo sampling
        assert phot_error == pytest.approx(8.4e-14, rel=0.0, abs=2e-14)

    def test_spectrum_to_magnitude(self):
        jup_wavel, jup_flux, jup_err = np.loadtxt("plnt_Jupiter.txt", unpack=True)

        synphot = SyntheticPhotometry("MKO/NSFCam.J")

        app_mag, abs_mag = synphot.spectrum_to_magnitude(
            jup_wavel, jup_flux, error=jup_err, distance=(1.0, 0.01), threshold=None
        )

        assert app_mag[0] == pytest.approx(0.576611165817664, rel=self.limit, abs=0.0)
        assert abs_mag[0] == pytest.approx(5.576611165817664, rel=self.limit, abs=0.0)

        # The error is estimated with Monte Carlo sampling
        assert app_mag[1] == pytest.approx(5.368048545366946e-05, rel=0.0, abs=2e-5)
        assert abs_mag[1] == pytest.approx(0.021714790446227043, rel=0.0, abs=1e-2)

    def test_zero_point(self):
        with pytest.warns(UserWarning) as warning:
            synphot = SyntheticPhotometry("MKO/NSFCam.J", zero_point=1e-2)

        flux, error = synphot.magnitude_to_flux(20.0, error=0.5)

        assert flux == pytest.approx(1.0280162981264745e-10, rel=self.limit, abs=0.0)
        assert error == pytest.approx(4.903307995457426e-11, rel=self.limit, abs=0.0)
