.. _data:

Data
====

On this page you find a list of the supported data. Please give credit to the relevant references when using any of the external data in a publication. Support for other datasets can be requested by creating an `issue <https://github.com/tomasstolker/species/issues>`_ on the GitHub page.

Atmospheric models
------------------

.. json_models:: ../species/data/model_data/model_data.json

.. tip::
  The :func:`~species.data.database.Database.available_models()` method of the :class:`~species.data.database.Database` class can be used for printing a detailed overview of all available model grids:

  .. code-block:: python

     from species import SpeciesInit
     from species.data.database import Database

     SpeciesInit()

     database = Database()
     database.available_models()

.. tip::
  It is also possible to add your own custom grid of model spectra with :func:`~species.data.database.Database.add_custom_model()`.

Evolutionary models
-------------------

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

Spectral libraries
------------------

- `IRTF Spectral Library <http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/>`_
- `SpeX Prism Spectral Libraries <http://pono.ucsd.edu/~adam/browndwarfs/spexprism/index_old.html>`_
- `SDSS spectra by Kesseli et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJS..230...16K/abstract>`_
- NIR spectra of young M/L dwarfs by `Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...772...79A/abstract>`_
- NIR spectra of young M/L dwarfs by `Bonnefoy et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014A%26A...562A.127B/abstract>`_
- `Spectra of directly imaged planets and brown dwarfs <https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_spectra.json>`_

Photometric libraries
---------------------

- `Database of Ultracool Parallaxes <http://www.as.utexas.edu/~tdupuy/plx/Database_of_Ultracool_Parallaxes.html>`_
- Photometry from `S. Leggett <http://www.gemini.edu/staff/sleggett>`_
- `Magnitudes, stellar properties, and other parameters of directly imaged planets and brown dwarfs <https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json>`_
- Parallaxes, photometry, and spectra from the `SIMPLE database <https://simple-bd-archive.org>`_

Calibration
-----------

- Filters from the `Filter Profile Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_
- `HST flux-calibrated spectrum of Vega <https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec>`_

Dust extinction
---------------

- Extinction models from `dust-extinction <https://dust-extinction.readthedocs.io/en/latest/dust_extinction/choose_model.html>`_
- Dust cross sections computed with `PyMieScatt <https://pymiescatt.readthedocs.io>`_
- Optical constants from `Mollière et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
