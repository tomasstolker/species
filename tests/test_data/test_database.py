import numpy as np
import pytest

from species.data.database import Database


def test_get_mcmc_spectra_is_deprecated_alias(monkeypatch):
    database = Database.__new__(Database)
    expected = []
    calls = {}

    def get_posterior_spectra(**kwargs):
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(database, "get_posterior_spectra", get_posterior_spectra)

    with pytest.warns(DeprecationWarning, match="get_posterior_spectra"):
        result = database.get_mcmc_spectra(
            "tag",
            random=3,
            wavel_range=(1.0, 2.0),
            spec_res=100.0,
            wavel_resample=np.array([1.0, 2.0]),
        )

    assert result is expected
    assert calls["tag"] == "tag"
    assert calls["random"] == 3


def test_get_mcmc_photometry_is_deprecated_alias(monkeypatch):
    database = Database.__new__(Database)
    expected = np.array([1.0])
    calls = {}

    def get_posterior_photometry(**kwargs):
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(database, "get_posterior_photometry", get_posterior_photometry)

    with pytest.warns(DeprecationWarning, match="get_posterior_photometry"):
        result = database.get_mcmc_photometry(
            "tag",
            "MKO/NSFCam.J",
            random=3,
            phot_type="flux",
            flux_units="W m-2 um-1",
        )

    np.testing.assert_array_equal(result, expected)
    assert calls["tag"] == "tag"
    assert calls["filter_name"] == "MKO/NSFCam.J"
    assert calls["random"] == 3
