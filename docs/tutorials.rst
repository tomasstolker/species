.. _tutorials:

Tutorials
=========

Below is a set of tutorials highlighting key functionalities of the toolkit.
All tutorials can be downloaded as `Jupyter notebooks <https://github.com/tomasstolker/species/tree/main/docs/tutorials>`_.
Please `create an issue <https://github.com/tomasstolker/species/issues>`_ if you encounter any problems or unexpected behavior.

.. tip::
   Detailed descriptions of all classes, functions, and parameters are available in the
   `API documentation <https://species.readthedocs.io/en/latest/modules.html>`_.


Fitting models and parameter inference
--------------------------------------

- :doc:`tutorials/fitting_model_spectra`
  – Inference of atmospheric parameters using model grids and Bayesian inference.

- :doc:`tutorials/grid_comparison`
  – Compare observed spectra and photometry to a full grid of model spectra.

- :doc:`tutorials/evolution_fit`
  – Inference of bulk parameters using evolutionary tracks.

- :doc:`tutorials/atmospheric_retrieval`
  – Atmospheric retrieval with ``petitRADTRANS`` to constrain chemical abundances, P-T profile, and clouds.

- :doc:`tutorials/emission_line`
  – Analysis of hydrogen emission lines from accreting planets.

- :doc:`tutorials/flux_calibration`
  – Flux calibration by modeling stellar magnitudes with synthetic spectra.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/fitting_model_spectra.ipynb
   tutorials/grid_comparison.ipynb
   tutorials/evolution_fit.ipynb
   tutorials/atmospheric_retrieval.ipynb
   tutorials/emission_line.ipynb
   tutorials/flux_calibration.ipynb


Atmospheric and evolutionary models
-----------------------------------

- :doc:`tutorials/model_spectra`
  – Interpolate and visualize atmospheric model spectra.

- :doc:`tutorials/data_model`
  – Combine observational data and atmospheric spectra in a plot.

- :doc:`tutorials/read_isochrone`
  – Interpolate and visualize evolutionary tracks and isochrones.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/model_spectra.ipynb
   tutorials/data_model.ipynb
   tutorials/read_isochrone.ipynb


Synthetic photometry, magnitudes, and fluxes
--------------------------------------------

- :doc:`tutorials/flux_magnitude`
  – Convert between fluxes and magnitudes.

- :doc:`tutorials/synthetic_photometry`
  – Compute synthetic photometry from a spectrum.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/flux_magnitude.ipynb
   tutorials/synthetic_photometry.ipynb


Color and magnitude diagrams
----------------------------

- :doc:`tutorials/color_magnitude_broadband`
  – Construct a broadband color–magnitude diagram that compares planets with field objects and model isochrones.

- :doc:`tutorials/color_magnitude_narrowband`
  – Construct a narrowband color-magnitude diagram that includes synthetic photometry from a spectral library.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/color_magnitude_broadband.ipynb
   tutorials/color_magnitude_narrowband.ipynb

Spectral libraries
------------------

- :doc:`tutorials/spectral_library`
  – Load and visualize a spectral library of stars and substellar objects.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/spectral_library.ipynb

Data of directly imaged planets
-------------------------------

- :doc:`tutorials/companion_data`
  – Access observational data of directly imaged planets and brown dwarfs.

- :doc:`tutorials/mass_ratio`
  – Extract and visualize the mass ratios of substellar companions.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/companion_data.ipynb
   tutorials/mass_ratio.ipynb
