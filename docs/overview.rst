.. _overview:

Overview
========

Introduction
------------

*species* provides a coherent framework for analyzing photometric and spectral data of planetary and substellar atmospheres. It has been designed for self-luminous objects such as directly imaged exoplanets and brown dwarfs, that is, objects which are not significantly irradiated by an external source. On this page you find a very short introduction on the design of *species*, the features that have been implemented, and the various types of data it can handle.

Supported data
--------------

The software profits from publicly available data such as photometric and spectral libraries, atmospheric model spectra, evolutionary models, photometry of directly imaged companions, and filter transmission curves. The relevant data are automatically downloaded and added to a central HDF5 database on request of the user. All data are stored in a fixed format such that the various analysis tools and plotting functions can be directly used without having to deal with the many different data formats and units in which the original data were stored. The following data are currently supported by *species* (support for other data sets can be requested):

- `DRIFT-PHOENIX <http://svo2.cab.inta-csic.es/theory/newov/index.php?model=drift>`_ atmospheric models
- `AMES-Cond <https://phoenix.ens-lyon.fr/Grids/AMES-Cond/>`_ atmospheric models
- `AMES-Dusty <https://phoenix.ens-lyon.fr/Grids/AMES-Dusty/>`_ atmospheric models
- `BT-NextGen <https://phoenix.ens-lyon.fr/Grids/BT-NextGen/SPECTRA/>`_ atmospheric models
- `BT-Settl <https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011/SPECTRA/>`_ atmospheric models
- All filter profiles from the `Filter Profile Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_
- Spectra from the `IRTF Spectral Library <http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/>`_
- Spectra from the `SpeX Prism Spectral Libraries <http://pono.ucsd.edu/~adam/browndwarfs/spexprism/index_old.html>`_
- All isochrone data from the `Phoenix model grids <https://phoenix.ens-lyon.fr/Grids/>`_
- Photometry from the `Database of Ultracool Parallaxes <http://www.as.utexas.edu/~tdupuy/plx/Database_of_Ultracool_Parallaxes.html>`_
- Photometry from `Sandy Leggett <http://www.gemini.edu/staff/sleggett>`_
- Photometry from `A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence <http://www.pas.rochester.edu/~emamajek>`_
- Photometry of directly images companions
- Vega `calibration spectrum <http://ssb.stsci.edu/cdbs/calspec/>`_

Please give credit to the relevant authors when using these data in a publication. More details will follow on the references but the information is also available on the respective websites.

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

Data which are read from the database, as well as the output of the various *species* functions, are stored in :class:`~species.core.box.Box` objects. These can be inputted directly to the plotting functions but the used can use their content also in their own plotting routine. The :func:`~species.core.box.Box.open_box` function can be used to see which attributes are inside a box. The following example will add the available photometry of PZ Tel B to the database, reads the data, creates an :class:`~species.core.box.ObjectBox`, and shows its content::

   import species

   species.SpeciesInit('./')

   database = species.Database()
   database.add_companion(name='PZ Tel B')

   objectbox = database.get_object(object_name='PZ Tel B')
   objectbox.open_box()

