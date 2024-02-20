import os
import shutil

import pytest
import numpy as np

from species import SpeciesInit
from species.data.database import Database
from species.read.read_planck import ReadPlanck
from species.phot.syn_phot import SyntheticPhotometry
from species.util import test_util


class TestPlanck:
    def setup_class(self):
        self.limit = 1e-8
        self.test_path = os.path.dirname(__file__) + "/"

    def teardown_class(self):
        os.remove("species_database.hdf5")
        os.remove("species_config.ini")
        shutil.rmtree("data/")

    def test_species_init(self):
        test_util.create_config("./")
        SpeciesInit()

    def test_read_planck(self):
        read_planck = ReadPlanck(filter_name="MKO/NSFCam.J")
        assert read_planck.wavel_range == pytest.approx(
            (1.1308, 1.3812), rel=1e-6, abs=0.0
        )

        read_planck = ReadPlanck(wavel_range=(1.0, 5.0))
        assert read_planck.wavel_range == (1.0, 5.0)

    def test_get_spectrum(self):
        read_planck = ReadPlanck(filter_name="MKO/NSFCam.J")
        modelbox = read_planck.get_spectrum(
            {"teff": 2000.0, "radius": 1.0, "parallax": 100.0}, 100.0
        )

        assert modelbox.model == "planck"
        assert modelbox.wavelength.shape == (204,)
        assert modelbox.flux.shape == (204,)

        assert np.sum(modelbox.wavelength) == pytest.approx(
            255.37728257033913, rel=self.limit, abs=0.0
        )
        assert np.sum(modelbox.flux) == pytest.approx(
            4.228358073212532e-12, rel=self.limit, abs=0.0
        )

    def test_get_flux(self):
        read_planck = ReadPlanck(filter_name="MKO/NSFCam.J")

        # low relative precision because of filter profile precision
        flux = read_planck.get_flux({"teff": 2000.0, "radius": 1.0, "distance": 10.0})
        assert flux[0] == pytest.approx(2.079882900702339e-14, rel=1e-4, abs=0.0)

        # low relative precision because of filter profile precision
        synphot = SyntheticPhotometry(filter_name="MKO/NSFCam.J")
        flux = read_planck.get_flux(
            {"teff": 2000.0, "radius": 1.0, "distance": 10.0}, synphot=synphot
        )
        assert flux[0] == pytest.approx(2.079882900702339e-14, rel=1e-4, abs=0.0)
