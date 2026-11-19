import importlib
import sys

import pytest


def test_plot_posterior_module_exports_plotting_functions():
    module = importlib.import_module("species.plot.plot_posterior")

    assert callable(module.plot_posterior)
    assert callable(module.plot_mag_posterior)
    assert callable(module.plot_size_distributions)
    assert callable(module.plot_extinction)


def test_plot_mcmc_is_deprecated_compatibility_module():
    sys.modules.pop("species.plot.plot_mcmc", None)

    with pytest.warns(DeprecationWarning, match="plot.plot_posterior"):
        deprecated_module = importlib.import_module("species.plot.plot_mcmc")

    current_module = importlib.import_module("species.plot.plot_posterior")

    assert deprecated_module.plot_posterior is current_module.plot_posterior
    assert deprecated_module.plot_mag_posterior is current_module.plot_mag_posterior
    assert (
        deprecated_module.plot_size_distributions
        is current_module.plot_size_distributions
    )
    assert deprecated_module.plot_extinction is current_module.plot_extinction
