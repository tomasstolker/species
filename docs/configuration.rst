.. _configuration:

Configuration
=============

A configuration file with the name `species_config.ini` is required in the working folder. It contains the global settings, such as the path of the HDF5 :ref:`database`, the folder where data will be dowloaded, and the magnitude of Vega.

.. code-block:: ini

   [species]
   database = species_database.hdf5
   data_folder = /path/to/store/data/
   vega_mag = 0.03

Paths should be either absolute or relative to the working folder, so in this example the database is located in the working folder.

.. important::
   The configuration file should either be located in the working folder, so the folder returned by `os.getcwd() <https://docs.python.org/3/library/os.html#os-file-dir>`_, or the file path can be set with the ``SPECIES_CONFIG`` environment variable:

.. code-block:: python

   >>> import os
   >>> os.environ["SPECIES_CONFIG"] = "/path/to/species_config.ini"

The workflow with *species* is initiated with the :class:`~species.core.species_init.SpeciesInit` class:

.. code-block:: python

   >>> import species
   >>> species.SpeciesInit()

A configuration file with default values is created in case this file is not present in the working folder.

.. tip::
   The same `data_folder` can be used in multiple configuration files. In this way, the data is only downloaded once and easily reused by new instances of :class:`~species.core.species_init.SpeciesInit`. Also the HDF5 :ref:`database` can be reused by including the same `database` path in the configuration file.

.. important::
   A flux-calibrated spectrum of Vega is used for the conversion between a flux density and magnitude. The magnitude of Vega is set to 0.03 for all filters by default. If needed, the magnitude of Vega can be adjusted with the ``vega_mag`` parameter in the configuration file.
