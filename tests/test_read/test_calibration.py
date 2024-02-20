import os
import shutil

import pytest
import numpy as np

from species import SpeciesInit
from species.data.database import Database
from species.read.read_calibration import ReadCalibration
from species.util import test_util


class TestCalibration:
    def setup_class(self):
        self.limit = 1e-8
        self.test_path = os.path.dirname(__file__) + "/"
        self.model_param = {"scaling": 1.0}

    def teardown_class(self):
        os.remove("species_database.hdf5")
        os.remove("species_config.ini")
        shutil.rmtree("data/")

    def test_species_init(self):
        test_util.create_config("./")
        SpeciesInit()

    def test_read_calibration(self):
        database = Database()
        database.add_spectra("vega")

        read_calib = ReadCalibration("vega", filter_name="Paranal/NACO.H")
        assert read_calib.wavel_range == pytest.approx((1.44, 1.88), rel=1e-7, abs=0.0)

    def test_resample_spectrum(self):
        read_calib = ReadCalibration("vega")
        spec_box = read_calib.resample_spectrum(
            np.linspace(1.0, 2.0, 10), apply_mask=True
        )

        assert np.sum(spec_box.wavelength) == 15.0
        assert np.sum(spec_box.flux) == pytest.approx(
            2.2628022608148692e-08, rel=self.limit, abs=0.0
        )

    def test_get_spectrum(self):
        read_calib = ReadCalibration("vega", filter_name="Paranal/NACO.Lp")
        spec_box = read_calib.get_spectrum(
            self.model_param, apply_mask=True, wavel_sampling=200.0
        )

        assert np.sum(spec_box.wavelength) == pytest.approx(
            183.35527924487636, rel=self.limit, abs=0.0
        )
        assert np.sum(spec_box.flux) == pytest.approx(
            2.3131999524734138e-09, rel=self.limit, abs=0.0
        )

    def test_get_flux(self):
        read_calib = ReadCalibration("vega", filter_name="Paranal/NACO.H")
        flux = read_calib.get_flux(model_param=self.model_param)

        assert flux[0] == pytest.approx(1.1149293297882683e-09, rel=self.limit, abs=0.0)

    def test_get_magnitude(self):
        read_calib = ReadCalibration("vega", filter_name="Paranal/NACO.H")
        app_mag, abs_mag = read_calib.get_magnitude(model_param=self.model_param)

        assert app_mag[0] == 0.03
        assert abs_mag[0] is None
