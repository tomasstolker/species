import os
import shutil

import pytest
import numpy as np

from species import SpeciesInit
from species.data.database import Database
from species.read.read_object import ReadObject
from species.util import test_util


class TestObject:
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

    def test_read_object(self):
        database = Database()

        with pytest.warns(UserWarning):
            database.add_companion(name="beta Pic b")

        read_object = ReadObject("beta Pic b")
        assert read_object.object_name == "beta Pic b"

        with pytest.raises(ValueError) as error:
            ReadObject("wrong name")

        assert (
            str(error.value)
            == "The object 'wrong name' is not present in the database."
        )

    def test_get_photometry(self):
        read_object = ReadObject("beta Pic b")
        photometry = read_object.get_photometry("Paranal/NACO.Lp")

        assert isinstance(photometry, np.ndarray)

        assert photometry[0] == pytest.approx(11.3, rel=self.limit, abs=0.0)
        assert photometry[1] == pytest.approx(0.06, rel=self.limit, abs=0.0)
        assert photometry[2] == pytest.approx(
            1.5566122192562612e-15, rel=self.limit, abs=0.0
        )
        assert photometry[3] == pytest.approx(
            8.606536033479756e-17, rel=self.limit, abs=0.0
        )

    def test_get_parallax(self):
        read_object = ReadObject("beta Pic b")
        parallax = read_object.get_parallax()

        assert parallax == (50.9307, 0.1482)

    def test_get_distance(self):
        read_object = ReadObject("beta Pic b")
        distance = read_object.get_distance()

        assert distance == (19.63452298908124, 0.05713373162362245)

    def test_get_absmag(self):
        read_object = ReadObject("beta Pic b")
        abs_mag = read_object.get_absmag("Paranal/NACO.Lp")

        assert abs_mag[0] == pytest.approx(9.834898226163453, rel=self.limit, abs=0.0)
        assert abs_mag[1] == pytest.approx(0.06033179718686261, rel=self.limit, abs=0.0)
