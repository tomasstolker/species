.. _tutorials:

Tutorials
=========

This page contains a list of tutorials which highlight some of the functionalities of `species`. The tutorials can be downloaded as `Jupyter notebook <https://github.com/tomasstolker/species/tree/main/docs/tutorials>`_. Please `create an issue <https://github.com/tomasstolker/species/issues>`_ if you encounter any problems.

.. tip::
   Details on the various classes, functions, and parameters can be found in the `API documentation <https://species.readthedocs.io/en/latest/modules.html>`_.

.. important::
   Importing the ``species`` package had become slow because of the many classes and functions that were implicitly imported. The initialization of the package has therefore been adjusted. Any functionalities should now be explicitly imported from the modules that they are part of.

**Spectral retrievals**

.. toctree::
   :maxdepth: 1

   tutorials/fitting_model_spectra.ipynb
   tutorials/atmospheric_retrieval.ipynb

**Atmospheric models**

.. toctree::
   :maxdepth: 1

   tutorials/model_spectra.ipynb
   tutorials/data_model.ipynb

**Evolutionary models**

.. toctree::
   :maxdepth: 1

   tutorials/read_isochrone.ipynb

**Photometry**

.. toctree::
   :maxdepth: 1

   tutorials/flux_magnitude.ipynb
   tutorials/synthetic_photometry.ipynb

**Companion data**

.. toctree::
   :maxdepth: 1

   tutorials/companion_data.ipynb
   tutorials/mass_ratio.ipynb

**Color and magnitude diagrams**

.. toctree::
   :maxdepth: 1

   tutorials/color_magnitude_broadband.ipynb
   tutorials/color_magnitude_narrowband.ipynb

**Miscellaneous**

.. toctree::
   :maxdepth: 1

   tutorials/flux_calibration.ipynb
   tutorials/emission_line.ipynb
   tutorials/spectral_library.ipynb
