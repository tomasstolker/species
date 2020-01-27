.. _overview:

Overview
========

Introduction
------------

*species* provides a coherent framework for analyzing spectra and photometry of directly imaged planets and brown dwarfs. This page provides a short introduction on the design of *species*, the features that have been implemented, and the various types of data it can handle.

Supported data
--------------

The software profits from publicly available data such as photometric and spectral libraries, atmospheric model spectra, evolutionary models, photometry of directly imaged companions, and filter transmission profiles. The relevant data are automatically downloaded and can be added to the HDF5 database. All data are stored in a fixed format in the database which has the advantage that the analysis tools and plotting functions can be easily used without having to deal with the original data formats and units.

The following data are currently supported by *species* (support for other data sets can be requested):

- `DRIFT-PHOENIX <http://svo2.cab.inta-csic.es/theory/newov/index.php?model=drift>`_ atmospheric models
- `AMES-Cond <https://phoenix.ens-lyon.fr/Grids/AMES-Cond/>`_ atmospheric models
- `AMES-Dusty <https://phoenix.ens-lyon.fr/Grids/AMES-Dusty/>`_ atmospheric models
- `BT-NextGen <https://phoenix.ens-lyon.fr/Grids/BT-NextGen/SPECTRA/>`_ atmospheric models
- `BT-Settl <https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011/SPECTRA/>`_ atmospheric models
- `petitCODE <http://www.mpia.de/~molliere/#petitcode>`_ atmospheric models
- All filter profiles from the `Filter Profile Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_
- Spectra from the `IRTF Spectral Library <http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/>`_
- Spectra from the `SpeX Prism Spectral Libraries <http://pono.ucsd.edu/~adam/browndwarfs/spexprism/index_old.html>`_
- All isochrone data from the `Phoenix model grids <https://phoenix.ens-lyon.fr/Grids/>`_
- Photometry from the `Database of Ultracool Parallaxes <http://www.as.utexas.edu/~tdupuy/plx/Database_of_Ultracool_Parallaxes.html>`_
- Photometry from `Sandy Leggett <http://www.gemini.edu/staff/sleggett>`_
- Photometry of directly imaged planets and brown dwarfs (see dictionary in :class:`~species.data.companions`)
- Calibration spectrum of `Vega <http://ssb.stsci.edu/cdbs/calspec/>`_

Please give credit to the relevant authors when using these data in a publication. More information is available on the respective websites.

Analysis tools
--------------

After adding the relevant data to the database, the user can take advantage of the suite of tools that have been implemented to analyze the photometric and/or spectral data. Here is an incomplete list of types of analysis for which *species* is suitable:

- Converting between fluxes and magnitudes.
- Calculating synthetic photometry from atmospheric model spectra and spectral libraries.
- Interpolating and plotting atmospheric model spectra.
- Fitting photometry and/or spectra with a grid of atmospheric models or a single calibration spectrum (e.g. an IRTF spectrum) with an MCMC sampling approach.
- Plotting posterior distributions and randomly sampled spectra together with the synthetic photometry, filter transmission curves, and best-fit residuals.
- Fitting photometry and/or spectra in the context of a photometric calibration. The MCMC samples can be used to compute the posterior distribution of the synthetic photometry of a specific filter.
- Creating color-magnitude and color-color diagrams by using photometric libraries, synthetic photometry of spectral libraries, photometry of directly imaged objects, isochrone data.
- Reading evolutionary tracks and calculating synthetic photometry for a given age and range of masses.

Output boxes
------------

Data which are read from the database, as well as the output of various functions, are stored in :class:`~species.core.box.Box` objects. These can be used directly as input for the plotting functionalities of `species`. Alternatively, the user can easily extract the content of a :class:`~species.core.box.Box` and process and plot it to their own needs. The :func:`~species.core.box.Box.open_box` function can be used to see which attributes are inside a :class:`~species.core.box.Box`.

The following example will add already available photometry of PZ Tel B to the database, read the data and properties of companion into an :class:`~species.core.box.ObjectBox`, and list its content.

.. code-block:: python

   import species

   species.SpeciesInit(config_path='./')

   database = species.Database()
   database.add_companion(name='PZ Tel B')

   objectbox = database.get_object(object_name='PZ Tel B')
   objectbox.open_box()

As an example, a dictionary with the apparent magnitudes can be extracted from the :class:`~species.core.box.ObjectBox` in the following way:

.. code-block:: python

   app_mag = objectbox.magnitude
