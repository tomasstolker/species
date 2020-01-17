.. _configuration:

Configuration
=============

A configuration file with the name `species_config.ini` is required in the working folder. At the moment, the file contains the path of the HDF5 database and the location where all the data is dowloaded before it is stored into the database. These can be provided as absolute paths or relative to the working folder. This is what the content of the configuration file may look like:

.. code-block:: ini

   [species]
   database = species_database.hdf5
   data_folder = /path/to/store/data/

A configuration file with default values is automatically created when `species` is initiated by running :class:`~species.core.setup.SpeciesInit` and no configuration file is present in the working folder, for example::

   import species

   species.SpeciesInit(config_path='./')

.. tip::
   The same `data_folder` can be set in multiple configuration files of `species`. In this way, the data is only downloaded once and easily reused by a new instance of :class:`~species.core.setup.SpeciesInit`. Also the HDF5 database can be reused by simply including the same `database` in the configuration file.
