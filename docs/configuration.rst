.. _configuration:

Configuration
=============

A configuration file with the name `species_config.ini` is required in the working folder. The configuration file contains the global settings, such as the path of the HDF5 database and the location where the data will be dowloaded. These can be provided as absolute paths or relative to the working folder. The file also contains the magnitude of Vega that is used for converting between magnitudes and fluxes. This is what the content of the configuration file may look like:

.. code-block:: ini

   [species]
   database = species_database.hdf5
   data_folder = /path/to/store/data/
   vega_mag = 0.03

In this case, the database is stored in the working folder and an absolute path points to the folder where the data is downloaded.

.. important::
   The configuration file should always be located in the working folder. Are you not sure about your current working folder? Try running Python's `os.getcwd() <https://docs.python.org/3/library/os.html#os-file-dir>`_ function:

      .. code-block:: python

         >>> import os
         >>> os.getcwd()

The workflow with *species* is now initiated with the :class:`~species.core.species_init.SpeciesInit` class:

.. code-block:: python

   >>> import species
   >>> species.SpeciesInit()

A configuration file with default values is automatically created when `species` is initiated and the file is not present in the working folder.

.. tip::
   The same `data_folder` can be used in multiple configuration files. In this way, the data is only downloaded once and easily reused by a new instance of :class:`~species.core.species_init.SpeciesInit`. Also the HDF5 database can be reused by simply including the same `database` in the configuration file (:ref:`database` section).

.. important::
   A flux-calibrated spectrum of Vega is used for the conversion between a flux density and magnitude. The magnitude of Vega is set to 0.03 for all filters by default. If needed, the magnitude of Vega can be adjusted with the ``vega_mag`` parameter in the configuration file.
