.. _tutorials:

Tutorials
=========

This page contains a list of tutorials which highlight some of the functionalities of `species`. These examples are also available as `Jupyter notebook <https://github.com/tomasstolker/species/tree/master/docs/tutorials>`_. Some of tutorials are still work in progress and more examples will be added in the future. Please contact Tomas Stolker if you have questions regarding a specific science case (see :ref:`about` section) or `create an issue <https://github.com/tomasstolker/species/issues>`_ if you encounter any problems with the tutorials.

.. warning::
   The last tutorial is somewhat outdated and incomplete because `species` has been under continuous development. If anything is unclear or causes an error, then please have a look at the `API documentation <https://species.readthedocs.io/en/latest/modules.html>`_, which is kept up-to-date, or contact Tomas Stolker (see :ref:`about` section).

.. toctree::
   :maxdepth: 1

   tutorials/flux_magnitude.ipynb
   tutorials/synthetic_photometry.ipynb
   tutorials/spectral_library.ipynb
   tutorials/color_magnitude_broadband.ipynb
   tutorials/color_magnitude_narrowband.ipynb
   tutorials/atmospheric_models.ipynb
   tutorials/data_model.ipynb
   tutorials/flux_calibration.ipynb
   tutorials/fitting_model_spectra.ipynb

.. important::
   A flux calibrated spectrum of Vega is used for the conversion between a flux density and magnitude. The magnitude of Vega is set to 0.03 for all filters. If needed, the magnitude of Vega can be changed with the ``vega_mag`` attribute of a ``SyntheticPhotometry`` object:

   .. code-block:: python

       >>> synphot = species.SyntheticPhotometry('MKO/NSFCam.K')
       >>> synphot.vega_mag = 0.01
