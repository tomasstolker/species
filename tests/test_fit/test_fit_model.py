import pytest

import species.fit.fit_model as fit_model_module
from species import SpeciesInit
from species.data.database import Database
from species.fit.fit_model import FitModel
from species.util import model_util, test_util


@pytest.fixture
def fit_model_database(tmp_path, monkeypatch):
    # Create an isolated database with one synthetic photometric object.
    monkeypatch.chdir(tmp_path)
    test_util.create_config("./")
    SpeciesInit()

    database = Database()
    database.add_object(
        object_name="synthetic",
        parallax=(100.0, 1.0),
        flux_density={"MKO/NSFCam.J": (1.0, 0.1)},
        verbose=False,
    )

    return tmp_path


def create_fit_model():
    return FitModel(
        object_name="synthetic",
        model="powerlaw",
        bounds={
            "log_powerlaw_a": (-1.0, 1.0),
            "log_powerlaw_b": (-3.0, 3.0),
            "log_powerlaw_c": (0.5, 2.0),
        },
        inc_phot=True,
        inc_spec=False,
    )


def assert_samples_stored(tag, sampler):
    samples = Database().get_samples(tag)

    assert samples.samples.ndim == 2
    assert samples.samples.shape[1] == 3
    assert samples.samples.shape[0] > 0
    assert samples.ln_prob.shape[0] == samples.samples.shape[0]
    assert samples.attributes["sampler"] == sampler


def disable_grid_check(monkeypatch):
    # Sampler methods and Database.get_samples call this check, which expects
    # an atmospheric grid that the synthetic power-law model does not have.
    monkeypatch.setattr(
        fit_model_module, "check_nearest_spec", lambda *args: None
    )
    monkeypatch.setattr(model_util, "check_nearest_spec", lambda *args: None)


class TestFitModel:
    def test_create_model(self, fit_model_database):
        fit = create_fit_model()

        assert fit.model == "powerlaw"
        assert fit.modelpar == [
            "log_powerlaw_a",
            "log_powerlaw_b",
            "log_powerlaw_c",
        ]
        assert fit.bounds["log_powerlaw_a"] == (-1.0, 1.0)
        assert fit.filter_name == ["MKO/NSFCam.J"]

    def test_run_multinest(self, fit_model_database, monkeypatch):
        if not hasattr(fit_model_module, "pymultinest"):
            pytest.skip("PyMultiNest is not available")

        disable_grid_check(monkeypatch)
        fit = create_fit_model()
        fit.run_multinest(
            tag="multinest",
            n_live_points=10,
            output="multinest/",
        )

        assert_samples_stored("multinest", "multinest")

    def test_run_ultranest(self, fit_model_database, monkeypatch):
        if not hasattr(fit_model_module, "ultranest"):
            pytest.skip("UltraNest is not available")

        disable_grid_check(monkeypatch)
        fit = create_fit_model()
        fit.run_ultranest(
            tag="ultranest",
            min_num_live_points=40,
            output="ultranest/",
            kwargs_ultranest={
                "show_status": False,
                "viz_callback": False,
                "dlogz": 5.0,
                "min_ess": 10,
                "max_ncalls": 1000,
                "cluster_num_live_points": 10,
            },
        )

        assert_samples_stored("ultranest", "ultranest")

    def test_run_dynesty(self, fit_model_database, monkeypatch):
        disable_grid_check(monkeypatch)
        fit = create_fit_model()
        fit.run_dynesty(
            tag="dynesty",
            n_live_points=10,
            output="dynesty/",
            evidence_tolerance=5.0,
        )

        assert_samples_stored("dynesty", "dynesty")

    def test_run_emcee(self, fit_model_database, monkeypatch):
        disable_grid_check(monkeypatch)
        fit = create_fit_model()
        # TODO: Test with guess
        # TODO: Test with normal params
        fit.run_emcee(
            tag="emcee",
            nwalkers=32,
            nsteps=10,
        )

        assert_samples_stored("emcee", "emcee")
