import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from species import SpeciesInit
from species.data.database import Database
from species.fit import fit_model
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


def create_fit_model(fixed=False):
    bounds = {
        "log_powerlaw_b": (-3.0, 3.0),
        "log_powerlaw_c": (0.5, 2.0),
    }

    if fixed:
        bounds["log_powerlaw_c"] = (1.0, 1.0)

    return FitModel(
        object_name="synthetic",
        model="powerlaw",
        bounds=bounds,
        normal_prior={
            "log_powerlaw_a": (0.0, 0.5),
            "log_powerlaw_b": (0.0, 1.0),
        },
        inc_phot=True,
        inc_spec=False,
    )


def assert_samples_stored(tag, sampler, fixed=False):
    samples = Database().get_samples(tag)

    assert samples.samples.ndim == 2
    assert samples.samples.shape[1] == 3
    assert samples.samples.shape[0] > 0
    assert samples.ln_prob.shape[0] == samples.samples.shape[0]
    assert samples.attributes["sampler"] == sampler

    if fixed:
        assert np.all(samples.samples[:, -1] == 1.0)


def disable_grid_check(monkeypatch):
    # Sampler methods and Database.get_samples call this check, which expects
    # an atmospheric grid that the synthetic power-law model does not have.
    monkeypatch.setattr(fit_model, "check_nearest_spec", lambda *args: None)
    monkeypatch.setattr(model_util, "check_nearest_spec", lambda *args: None)


# TODO: Test with multiprocessing
class TestFitModel:
    @pytest.mark.parametrize("fixed", [False, True], ids=["free", "fixed"])
    def test_create_model(self, fit_model_database, fixed):
        fit = create_fit_model(fixed=fixed)

        assert fit.model == "powerlaw"
        assert fit.modelpar == ["log_powerlaw_a", "log_powerlaw_b"] + (
            [] if fixed else ["log_powerlaw_c"]
        )
        assert "log_powerlaw_a" not in fit.bounds
        assert fit.bounds["log_powerlaw_b"] == (-3.0, 3.0)
        assert fit.filter_name == ["MKO/NSFCam.J"]
        assert fit.fix_param == ({"log_powerlaw_c": 1.0} if fixed else {})

    @pytest.mark.parametrize("fixed", [False, True], ids=["free", "fixed"])
    def test_log_prior(self, fit_model_database, fixed):
        fit = create_fit_model(fixed=fixed)
        params = np.array([0.2, 0.0] + ([] if fixed else [1.0]))

        expected = (
            norm.logpdf(params[0], loc=0.0, scale=0.5)
            + truncnorm.logpdf(params[1], -3.0, 3.0, loc=0.0, scale=1.0)
            + (0.0 if fixed else -np.log(1.5))
        )

        assert fit._lnprior_func(params) == pytest.approx(expected)

    @pytest.mark.parametrize("fixed", [False, True], ids=["free", "fixed"])
    def test_run_multinest(self, fit_model_database, monkeypatch, fixed):
        if not hasattr(fit_model, "pymultinest"):
            pytest.skip("PyMultiNest is not available")

        disable_grid_check(monkeypatch)
        fit = create_fit_model(fixed=fixed)
        fit.run_multinest(
            tag="multinest",
            n_live_points=10,
            output="multinest/",
        )

        assert_samples_stored("multinest", "multinest", fixed=fixed)

    @pytest.mark.parametrize("fixed", [False, True], ids=["free", "fixed"])
    def test_run_ultranest(self, fit_model_database, monkeypatch, fixed):
        if not hasattr(fit_model, "ultranest"):
            pytest.skip("UltraNest is not available")

        disable_grid_check(monkeypatch)
        fit = create_fit_model(fixed=fixed)
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

        assert_samples_stored("ultranest", "ultranest", fixed=fixed)

    @pytest.mark.parametrize("fixed", [False, True], ids=["free", "fixed"])
    def test_run_dynesty(self, fit_model_database, monkeypatch, fixed):
        disable_grid_check(monkeypatch)
        fit = create_fit_model(fixed=fixed)
        fit.run_dynesty(
            tag="dynesty",
            n_live_points=10,
            output="dynesty/",
            evidence_tolerance=5.0,
        )

        assert_samples_stored("dynesty", "dynesty", fixed=fixed)

    @pytest.mark.parametrize("fixed", [False, True], ids=["free", "fixed"])
    def test_run_emcee(self, fit_model_database, monkeypatch, fixed):
        disable_grid_check(monkeypatch)
        fit = create_fit_model(fixed=fixed)
        fit.run_emcee(
            tag="emcee",
            nwalkers=32,
            nsteps=10,
        )

        assert_samples_stored("emcee", "emcee", fixed=fixed)

    def test_run_emcee_with_guess(self, fit_model_database, monkeypatch):
        disable_grid_check(monkeypatch)
        fit = create_fit_model()
        guess = {
            "log_powerlaw_a": 0.1,
            "log_powerlaw_b": 1.0,
        }

        fit.run_emcee(
            tag="emcee_guess",
            nwalkers=32,
            nsteps=10,
            guess=guess,
        )

        assert_samples_stored("emcee_guess", "emcee")

    def test_run_emcee_with_guess_outside_bounds(self, fit_model_database):
        fit = create_fit_model()
        guess = {
            "log_powerlaw_a": 0.1,
            "log_powerlaw_b": 4.0,
        }

        with pytest.raises(
            ValueError,
            match="initial guess for 'log_powerlaw_b'.*outside the bounds",
        ):
            fit.run_emcee(
                tag="emcee_invalid_guess",
                nwalkers=32,
                nsteps=10,
                guess=guess,
            )
