.. _overview:

Overview
========

Introduction
------------

*species* provides a coherent framework for spectral and photometric analysis of directly imaged exoplanets and brown dwarfs. This page contains a overview of the various data that are supported and some of the tools and features that are available.

Supported data
--------------

The toolkit benefits from publicly available data such as atmospheric model spectra, evolutionary tracks, and photometric and spectral libraries.

The relevant data are automatically downloaded and added to the HDF5 database, which acts as the central data storage for a workflow. All data are stored in a fixed format such that the analysis and plotting tools can easily access and process the data.

An overview of the supported data is provided in the :ref:`data` section.

Analysis tools
--------------

After adding the relevant data to the database, the user can take advantage of the suite of tools that have been implemented for spectral and photometric analysis. Below is an (incomplete) list of available features and tools. See also the :ref:`tutorials` section with detailed notebook tutorials.

**Synthetic photometry**

- Converting between fluxes and magnitudes (see :class:`~species.phot.syn_phot.SyntheticPhotometry`).
- Calculating synthetic photometry spectra (see :class:`~species.phot.syn_phot.SyntheticPhotometry`).

**Atmospheric models**

- Interpolating and plotting model spectra (see :class:`~species.read.read_model.ReadModel` and :func:`~species.plot.plot_spectrum.plot_spectrum`).
- Grid retrievals with Bayesian inference (see :class:`~species.fit.fit_model.FitModel` and :mod:`~species.plot.plot_mcmc`).
- Comparing a spectrum with a full grid of model spectra (see :meth:`~species.fit.compare_spectra.CompareSpectra.compare_model`).
- Wrapper for generating spectra with `petitRADTRANS <https://petitradtrans.readthedocs.io>`_ using various parameterizations for P-T structures, abundances, and clouds (see :class:`~species.read.read_radtrans.ReadRadtrans`).
- Free retrievals with a frontend for `petitRADTRANS <https://petitradtrans.readthedocs.io>`_  (see :class:`~species.fit.retrieval.AtmosphericRetrieval`).

**Evolutionary models**

- Computing synthetic fluxes from isochrones and model spectra (see :class:`~species.read.read_isochrone.ReadIsochrone`)

**Color-magnitude diagrams**

- Creating color-magnitude diagrams (see :class:`~species.read.read_color.ReadColorMagnitude` and :class:`~species.plot.plot_color.plot_color_magnitude`).
- Creating color-color diagrams (see :class:`~species.read.read_color.ReadColorColor` and :class:`~species.plot.plot_color.plot_color_color`).

**Flux calibration**

- Flux calibration of photometric and spectroscopic data (see :class:`~species.read.read_calibration.ReadCalibration`, :class:`~species.fit.fit_model.FitModel`, and :class:`~species.fit.fit_spectrum.FitSpectrum`).

**Empirical spectral analysis**

- Empirical comparison of spectra to infer the spectral type (see :meth:`~species.fit.compare_spectra.CompareSpectra.spectral_type`).

**Emission line analysis**

- Analyzing emission lines from accreting planets (see :class:`~species.fit.emission_line.EmissionLine`).
