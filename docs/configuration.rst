.. _configuration:

Configuration
=============

A configuration file with the name `species_config.ini` is required in the working folder. Currently, the configuration file contains the path of the HDF5 database and the location where all the data is dowloaded before it is stored into the database. These can be provided as absolute paths or relative to the working folder. This is what the content of the configuration file may look like:

.. code-block:: ini

   [species]
   database = species_database.hdf5
   data_folder = /path/to/store/data/

In this case the database is stored in the working folder and an absolute path points to the folder for the external data.

.. important::
   The configuration file should always be located in the working folder. Are you not sure about your current working folder? Try running the following Python code.

      .. code-block:: python

         >>> import os
         >>> os.getcwd()

The workflow with *species* can now be initiated with the :class:`~species.core.setup.SpeciesInit` class:

.. code-block:: python

   >>> import species
   >>> species.SpeciesInit()

Alternatively, a configuration file with default values is automatically created when `species` is initiated and the configuration file is not present in the working folder.

.. tip::
   The same `data_folder` can be used in multiple configuration files. In this way, the data is only downloaded once and easily reused by a new instance of :class:`~species.core.setup.SpeciesInit`. Also the HDF5 database can be reused by simply including the same `database` in the configuration file.
