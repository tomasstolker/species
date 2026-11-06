import os
import shutil
import urllib.request

from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import pytest
import numpy as np

from species import SpeciesInit
from species.core.box import IsochroneBox
from species.data.database import Database
from species.plot.plot_evolution import plot_isochrones
from species.read.read_isochrone import ReadIsochrone
from species.util import test_util


class TestIsochrone:
    def setup_class(self):
        self.limit = 1e-6
        self.test_path = os.path.dirname(__file__) + "/"
        self.initialized = False

    def teardown_class(self):
        if self.initialized:
            os.remove("species_database.hdf5")
            os.remove("species_config.ini")
            shutil.rmtree("data/")

    def test_species_init(self):
        test_util.create_config("./")
        type(self).initialized = True
        SpeciesInit()

    def test_read_isochrone(self):
        database = Database()
        database.add_isochrones("ames")

        database.add_model("ames-cond", teff_range=(2000.0, 2500.0))

        read_isochrone = ReadIsochrone("ames-cond")
        assert read_isochrone.tag == "ames-cond"

    def test_get_isochrone(self):
        read_isochrone = ReadIsochrone("ames-cond")

        isochrone_box = read_isochrone.get_isochrone(
            age=100.0,
            masses=np.linspace(10.0, 100.0, 10),
            filter_mag="J",
            filters_color=("J", "H"),
        )

        assert np.sum(isochrone_box.mass) == pytest.approx(
            550.0, rel=self.limit, abs=0.0
        )

        assert np.sum(isochrone_box.teff) == pytest.approx(
            23007.53864754502, rel=self.limit, abs=0.0
        )

        assert np.sum(isochrone_box.logg) == pytest.approx(
            47.475577469602705, rel=self.limit, abs=0.0
        )

        assert np.sum(isochrone_box.radius) == pytest.approx(
            14.29977420304961, rel=self.limit, abs=0.0
        )

        assert isochrone_box.color.shape == (10,)
        assert isochrone_box.magnitude.shape == (10,)

        assert np.sum(isochrone_box.color) == pytest.approx(
            2.625186321644007, rel=self.limit, abs=0.0
        )

        assert np.sum(isochrone_box.magnitude) == pytest.approx(
            108.78841310475491, rel=self.limit, abs=0.0
        )

    def test_get_color_magnitude(self):
        read_isochrone = ReadIsochrone("ames-cond")

        colormag_box = read_isochrone.get_color_magnitude(
            age=100.0,
            masses=np.linspace(35.0, 45.0, 10),
            filters_color=("MKO/NSFCam.J", "MKO/NSFCam.H"),
            filter_mag="MKO/NSFCam.J",
        )

        assert colormag_box.object_type == "model"
        assert colormag_box.color.shape == (10,)
        assert colormag_box.magnitude.shape == (10,)

        assert np.sum(colormag_box.color) == pytest.approx(
            2.5274585721232263, rel=self.limit, abs=0.0
        )

        assert np.sum(colormag_box.magnitude) == pytest.approx(
            109.43164467710773, rel=self.limit, abs=0.0
        )

        assert np.sum(colormag_box.mass) == pytest.approx(
            400.0, rel=self.limit, abs=0.0
        )

    def test_get_color_color(self):
        read_isochrone = ReadIsochrone("ames-cond")

        colorcolor_box = read_isochrone.get_color_color(
            age=100.0,
            masses=np.linspace(35.0, 45.0, 10),
            filters_colors=(
                ("MKO/NSFCam.J", "MKO/NSFCam.H"),
                ("MKO/NSFCam.H", "MKO/NSFCam.Ks"),
            ),
        )

        assert colorcolor_box.object_type == "model"
        assert colorcolor_box.color1.shape == (10,)
        assert colorcolor_box.color2.shape == (10,)

        assert np.sum(colorcolor_box.color1) == pytest.approx(
            2.5274585721232263, rel=self.limit, abs=0.0
        )

        assert np.sum(colorcolor_box.color2) == pytest.approx(
            3.3721764168667043, rel=self.limit, abs=0.0
        )

        assert np.sum(colorcolor_box.mass) == pytest.approx(
            400.0, rel=self.limit, abs=0.0
        )

    def test_get_entropy_track(self):
        read_isochrone = ReadIsochrone("ames-cond")

        with pytest.raises(ValueError, match="does not have initial entropy"):
            read_isochrone.get_entropy_track(age=100.0, mass=50.0)

    @pytest.fixture
    def entropy_database(self, tmp_path, monkeypatch):
        database_path = tmp_path / "entropy.hdf5"
        config_path = tmp_path / "species_config.ini"
        config_path.write_text(f"[species]\ndatabase = {database_path}\n")
        monkeypatch.setenv("SPECIES_CONFIG", str(config_path))

        ages = np.array([10.0, 20.0])
        masses = np.array([1.0, 2.0, 3.0])
        entropies = np.array([8.0, 9.0, 10.0, 11.0])
        age_grid, mass_grid, entropy_grid = np.meshgrid(
            ages, masses, entropies, indexing="ij"
        )
        luminosity = 0.01 * age_grid + 0.1 * mass_grid + 0.2 * entropy_grid

        with h5py.File(database_path, "w") as database:
            for tag, regular in (("regular", True), ("irregular", False)):
                group = database.create_group(f"isochrones/{tag}")
                group.attrs["regular_grid"] = regular
                group["age"] = ages if regular else age_grid.ravel()
                group["mass"] = masses if regular else mass_grid.ravel()
                group["s_init"] = entropies if regular else entropy_grid.ravel()
                for name, values in (
                    ("log_lum", luminosity),
                    ("teff", 1000.0 + luminosity),
                    ("log_g", 4.0 + luminosity),
                    ("radius", 1.0 + luminosity),
                ):
                    group[name] = values if regular else values.ravel()
                group.create_group("filters").create_dataset("H", data=0)
                group["filters"].create_dataset("J", data=0)
                magnitudes = np.stack((luminosity, luminosity + 1.0), axis=-1)
                group["magnitudes"] = (
                    magnitudes if regular else magnitudes.reshape(-1, 2)
                )

        return database_path

    @pytest.mark.parametrize("tag", ["regular", "irregular"])
    def test_entropy_track_interpolation(self, entropy_database, tag):
        reader = ReadIsochrone(tag, verbose=False)
        track = reader.get_entropy_track(age=15.0, mass=2.0, param_interp=["log_lum"])

        assert isinstance(track, IsochroneBox)
        assert track.age == 15.0
        assert track.mass == 2.0
        np.testing.assert_array_equal(track.s_init, [8.0, 9.0, 10.0, 11.0])
        np.testing.assert_allclose(track.log_lum, 0.35 + 0.2 * track.s_init)
        assert track.teff is None
        assert track.logg is None
        assert track.radius is None

        track = reader.get_entropy_track(
            age=15.0, mass=2.0, s_inits=np.array([8.5, 10.5]), param_interp=["teff"]
        )
        np.testing.assert_allclose(track.teff, 1000.35 + 0.2 * track.s_init)
        assert track.log_lum is None

        track = reader.get_entropy_track(
            age=15.0,
            mass=2.0,
            s_inits=np.array([8.5, 10.5, 12.0]),
            filter_mag="J",
            filters_color=("J", "H"),
        )

        expected = 0.35 + 0.2 * track.s_init[:2]
        np.testing.assert_allclose(track.log_lum[:2], expected)
        np.testing.assert_allclose(track.teff[:2], 1000.0 + expected)
        np.testing.assert_allclose(track.logg[:2], 4.0 + expected)
        np.testing.assert_allclose(track.radius[:2], 1.0 + expected)
        np.testing.assert_allclose(track.magnitude[:2], expected + 1.0)
        np.testing.assert_allclose(track.color[:2], 1.0)

        for values in (
            track.log_lum,
            track.teff,
            track.logg,
            track.radius,
            track.magnitude,
            track.color,
        ):
            assert np.isnan(values[-1])

    @pytest.mark.parametrize(
        ("track_type", "expected_grid", "expected_center"),
        [("entropy", [8.0, 9.0, 10.0, 11.0], 9.0), ("mass", [1.0, 2.0, 3.0], 2.0)],
    )
    def test_plot_entropy_track(
        self,
        entropy_database,
        monkeypatch,
        tmp_path,
        track_type,
        expected_grid,
        expected_center,
    ):
        samples = np.array([[15.0, 2.0, 8.5], [15.0, 2.0, 9.0], [15.0, 2.0, 10.0]])

        attributes = {
            "n_param": 3,
            "n_planets": 1,
            "model_name": "regular",
            "log_lum": np.array([[2.0, 0.1]]),
            "age_prior": np.array([15.0, -1.0, 1.0]),
            "mass_prior": np.array([np.nan]),
            "parameter0": "age",
            "parameter1": "mass_0",
            "parameter2": "s_init_0",
        }

        box = SimpleNamespace(
            samples=samples,
            attributes=attributes,
            parameters=["age", "mass_0", "s_init_0"],
        )

        monkeypatch.setattr(Database, "__init__", lambda self: None)
        monkeypatch.setattr(Database, "get_samples", lambda self, tag: box)

        fig, tracks, indices = plot_isochrones(
            "test",
            n_samples=1,
            track_type=track_type,
            output=str(tmp_path / "plot.png"),
        )
        try:
            np.testing.assert_array_equal(tracks[0][0][0], expected_grid)
            assert len(indices) == 1
            errorbar = fig.axes[0].containers[0]
            assert errorbar.lines[0].get_xdata()[0] == pytest.approx(expected_center)

        finally:
            plt.close(fig)
