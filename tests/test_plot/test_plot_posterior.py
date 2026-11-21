import importlib
import sys

import numpy as np
import pytest

from species.plot import plot_posterior

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


def test_index_vary_fixed():
    N = 11
    samples = np.vstack(
        (
            100.0 * np.ones(N),
            np.arange(N),
            100.0 * np.zeros(N),
            np.arange(-(N // 2), N // 2 + 1),
            -2 * np.ones(N),
        )
    ).T
    ndim = samples.shape[-1]
    index_var = plot_posterior.get_vary_indices(samples)
    index_fix = plot_posterior.get_fixed_indices(samples)

    assert index_var == [1, 3]
    assert index_fix == [0, 2, 4]
    assert sorted(index_var + index_fix) == list(range(ndim))

    # Backward compat check with previous code
    index_sel = []
    index_del = []
    for i in range(ndim):
        if np.amin(samples[:, i]) == np.amax(samples[:, i]):
            index_del.append(i)
        else:
            index_sel.append(i)

    assert index_var == index_sel
    assert index_fix == index_del
