import os
import shutil

import pytest
import numpy as np

from species import SpeciesInit
from species.data.database import Database
from species.read.read_model import ReadModel
from species.util import test_util


class TestModel:
    def setup_class(self):
        self.limit = 1e-8
        self.test_path = os.path.dirname(__file__) + "/"
        self.model_param = {
            "teff": 2200.0,
            "logg": 4.5,
            "radius": 1.0,
            "parallax": 100.0,
        }

    def teardown_class(self):
        os.remove("species_database.hdf5")
        os.remove("species_config.ini")
        shutil.rmtree("data/")

    def test_species_init(self):
        test_util.create_config("./")
        SpeciesInit()

    def test_read_model(self):
        database = Database()

        database.add_model(
            "ames-cond",
            wavel_range=(1.0, 5.0),
            wavel_sampling=200.0,
            teff_range=(2000.0, 2500.0),
        )

        read_model = ReadModel("ames-cond")
        assert read_model.model == "ames-cond"

    def test_get_model(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")

        model_box = read_model.get_model(
            self.model_param, spec_res=100.0, magnitude=False,
        )

        assert np.sum(model_box.wavelength) == pytest.approx(
            92.26773310928259, rel=self.limit, abs=0.0
        )
        assert np.sum(model_box.flux) == pytest.approx(
            1.709479501143029e-12, rel=self.limit, abs=0.0
        )

        model_box = read_model.get_model(
            self.model_param, spec_res=100.0, magnitude=True,
        )

        assert np.sum(model_box.wavelength) == pytest.approx(
            92.26773310928259, rel=self.limit, abs=0.0
        )
        assert np.sum(model_box.flux) == pytest.approx(
            646.3776539114224, rel=self.limit, abs=0.0
        )

    def test_get_data(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")
        model_box = read_model.get_data(self.model_param)

        assert np.sum(model_box.wavelength) == pytest.approx(
            92.26773310928259, rel=self.limit, abs=0.0
        )
        assert np.sum(model_box.flux) == pytest.approx(
            1.709461299237834e-12, rel=self.limit, abs=0.0
        )

    def test_get_flux(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")
        flux = read_model.get_flux(self.model_param)

        assert flux[0] == pytest.approx(3.489491094733077e-14, rel=self.limit, abs=0.0)

    def test_get_magnitude(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")
        magnitude = read_model.get_magnitude(self.model_param)

        assert magnitude[0] == pytest.approx(
            11.291213115013914, rel=self.limit, abs=0.0
        )
        assert magnitude[1] == pytest.approx(
            11.291213115013914, rel=self.limit, abs=0.0
        )

    def test_get_bounds(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")
        bounds = read_model.get_bounds()

        assert bounds["teff"] == (2000.0, 2500.0)
        assert bounds["logg"] == (2.5, 5.5)

    def test_get_wavelengths(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")
        wavelengths = read_model.get_wavelengths()

        assert np.sum(wavelengths) == pytest.approx(
            813.2224003071026, rel=1e-7, abs=0.0
        )

    def test_get_points(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")
        points = read_model.get_points()

        assert np.sum(points["teff"]) == 13500.0
        assert np.sum(points["logg"]) == 28.0

    def test_get_parameters(self):
        read_model = ReadModel("ames-cond", filter_name="Paranal/NACO.H")
        parameters = read_model.get_parameters()

        assert parameters == ["teff", "logg"]
