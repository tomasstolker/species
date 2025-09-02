.. _overview:

Overview
========

Introduction
------------

*species* provides a coherent framework for spectral and photometric analysis of directly imaged exoplanets and brown dwarfs. This page contains a overview of the various data that are supported and some of the tools and features that are available.

Supported data
--------------

The toolkit benefits from publicly available data resources such as atmospheric model spectra, photometric and spectral libraries, evolutionary tracks, and photometry of directly imaged companions. The relevant data are automatically downloaded and added to the HDF5 database, which acts as the central data storage for a workflow. All data are stored in a fixed format such that the analysis and plotting tools can easily access and process the data.

The following data and models are currently supported:

**Atmospheric models**

- `AMES-Cond <https://phoenix.ens-lyon.fr/Grids/AMES-Cond/>`_
- `AMES-Dusty <https://phoenix.ens-lyon.fr/Grids/AMES-Dusty/>`_
- `ATMO (CEQ, NEQ weak, NEQ strong) <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_
- `ATMO Chabrier et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023A&A...671A.119C>`_
- `BT-Cond <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-cond>`_
- `BT-Dusty <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-dusty>`_
- `BT-NextGen <https://phoenix.ens-lyon.fr/Grids/BT-NextGen/SPECTRA/>`_
- `BT-Settl <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-settl>`_
- `BT-Settl-CIFIST <http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=bt-settl-cifist>`_
- `coolTLUSTY Y dwarf <https://ui.adsabs.harvard.edu/abs/2023ApJ...950....8L/abstract>`_
- `DRIFT-PHOENIX <http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=drift>`_
- `Exo-REM <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..172C/abstract>`_
- `Morley et al. (2012) T/Y dwarf spectra <https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract>`_
- `petitCODE <https://www2.mpia-hd.mpg.de/~molliere/grids/>`_
- `petitRADTRANS <https://petitradtrans.readthedocs.io>`_
- `PHOENIX Husser et al. (2013) <https://phoenix.astro.physik.uni-goettingen.de>`_
- `Saumon & Marley (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...689.1327S/abstract>`_
- `Sonora Bobcat <https://zenodo.org/record/5063476>`_
- `Sonora Cholla <https://zenodo.org/record/4450269>`_
- `Sonora Diamondback <https://zenodo.org/records/12735103>`_
- `Sonora Elf Owl L-type <https://zenodo.org/records/10385987>`_
- `Sonora Elf Owl T-type <https://zenodo.org/records/10385821>`_
- `Sonora Elf Owl Y-type <https://zenodo.org/records/10381250>`_

.. tip::
  It is also possible to add your own custom grid of model spectra with :func:`~species.data.database.Database.add_custom_model()`.

.. tip::
  The :func:`~species.data.database.Database.available_models()` method of the :class:`~species.data.database.Database` class can be used for printing a detailed overview of all available model grids:

  .. code-block:: python

     from species import SpeciesInit
     from species.data.database import Database

     SpeciesInit()

     database = Database()
     database.available_models()

**Evolutionary models**

- `AMES-Cond isochrones <https://ui.adsabs.harvard.edu/abs/2003A%26A...402..701B/abstract>`_
- `AMES-Dusty isochrones <https://ui.adsabs.harvard.edu/abs/2000ApJ...542..464C/abstract>`_
- `ATMO isochrones <https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract>`_ (CEQ, NEQ weak, NEQ strong)
- `Baraffe et al. (2015) isochrones <http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/>`_
- `Linder et al. (2019) isochrones <https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..85L/abstract>`_
- `Saumon & Marley (2008) isochrones <https://ui.adsabs.harvard.edu/abs/2008ApJ...689.1327S/abstract>`_
- `Sonora Bobcat isochrones <https://zenodo.org/record/5063476>`_
- `Sonora Diamondback isochrones <https://zenodo.org/records/12735103>`_
- `PARSEC isochrones <https://stev.oapd.inaf.it/cgi-bin/cmd>`_
- Isochrones from the `Phoenix grids <https://phoenix.ens-lyon.fr/Grids/>`_

**Spectral libraries**

- `IRTF Spectral Library <http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/>`_
- `SpeX Prism Spectral Libraries <http://pono.ucsd.edu/~adam/browndwarfs/spexprism/index_old.html>`_
- `SDSS spectra by Kesseli et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJS..230...16K/abstract>`_
- NIR spectra of young M/L dwarfs by `Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...772...79A/abstract>`_
- NIR spectra of young M/L dwarfs by `Bonnefoy et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014A%26A...562A.127B/abstract>`_
- `Spectra of directly imaged planets and brown dwarfs <https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_spectra.json>`_

**Photometric libraries**

- `Database of Ultracool Parallaxes <http://www.as.utexas.edu/~tdupuy/plx/Database_of_Ultracool_Parallaxes.html>`_
- Photometry from `S. Leggett <http://www.gemini.edu/staff/sleggett>`_
- `Magnitudes, stellar properties, and other parameters of directly imaged planets and brown dwarfs <https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json>`_
- Parallaxes, photometry, and spectra from the `SIMPLE database <https://simple-bd-archive.org>`_

**Calibration**

- All filters from the `Filter Profile Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_
- Latest `flux-calibrated spectrum of Vega <https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec>`_

**Dust extinction**

- Extinction models from `dust-extinction <https://dust-extinction.readthedocs.io/en/latest/dust_extinction/choose_model.html>`_
- Dust cross sections computed with `PyMieScatt <https://pymiescatt.readthedocs.io>`_
- Optical constants adopted from `Mollière et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_

Please give credit to the relevant references when using any of the external data in a publication. More information is available on the respective websites. Support for other datasets can be requested by creating an `issue <https://github.com/tomasstolker/species/issues>`_ on the Github page.

Analysis tools
--------------

After adding the relevant data to the database, the user can take advantage of the suite of tools that have been implemented for spectral and photometric analysis. Here is an incomplete list of available features and tools:

- Converting between fluxes and magnitudes (see :class:`~species.phot.syn_phot.SyntheticPhotometry`).
- Calculating synthetic photometry spectra (see :class:`~species.phot.syn_phot.SyntheticPhotometry`).
- Interpolating and plotting model spectra (see :class:`~species.read.read_model.ReadModel` and :func:`~species.plot.plot_spectrum.plot_spectrum`).
- Wrapper for generating spectra with `petitRADTRANS <https://petitradtrans.readthedocs.io>`_ using various parameterizations for P-T structures, abundances, and clouds (see :class:`~species.read.read_radtrans.ReadRadtrans`).
- Grid retrievals with Bayesian inference (see :class:`~species.fit.fit_model.FitModel` and :mod:`~species.plot.plot_mcmc`).
- Comparing a spectrum with a full grid of model spectra (see :meth:`~species.fit.compare_spectra.CompareSpectra.compare_model`).
- Free retrievals with a frontend for `petitRADTRANS <https://petitradtrans.readthedocs.io>`_  (see :class:`~species.fit.retrieval.AtmosphericRetrieval`).
- Creating color-magnitude diagrams (see :class:`~species.read.read_color.ReadColorMagnitude` and :class:`~species.plot.plot_color.plot_color_magnitude`).
- Creating color-color diagrams (see :class:`~species.read.read_color.ReadColorColor` and :class:`~species.plot.plot_color.plot_color_color`).
- Computing synthetic fluxes from isochrones and model spectra (see :class:`~species.read.read_isochrone.ReadIsochrone`)
- Flux calibration of photometric and spectroscopic data (see :class:`~species.read.read_calibration.ReadCalibration`, :class:`~species.fit.fit_model.FitModel`, and :class:`~species.fit.fit_spectrum.FitSpectrum`).
- Empirical comparison of spectra to infer the spectral type (see :meth:`~species.fit.compare_spectra.CompareSpectra.spectral_type`).
- Analyzing emission lines from accreting planets (see :class:`~species.fit.emission_line.EmissionLine`).
