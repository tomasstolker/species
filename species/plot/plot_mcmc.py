"""
Deprecated compatibility module for posterior plotting.
"""

import warnings

from species.plot.plot_posterior import (
    plot_extinction,
    plot_mag_posterior,
    plot_posterior,
    plot_size_distributions,
)

warnings.warn(
    "The 'species.plot.plot_mcmc' module is deprecated. Use "
    "'species.plot.plot_posterior' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "plot_extinction",
    "plot_mag_posterior",
    "plot_posterior",
    "plot_size_distributions",
]
