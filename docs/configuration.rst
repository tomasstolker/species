.. _configuration:

Configuration
=============

A configuration file is required in the working folder with the name `species_config.ini`. The file contains the path to the HDF5 database, the path to the configuration file, and the folder where all the data is dowloaded before it is stored into the database.

The paths can be either absolute or relative to the working folder. It is recommended to use the same input folder for different configurations of `species` such that data only has to be downloaded once. As an example, this is what the content of the configuration file may look like:

.. code-block:: ini

   [species]
   database = species_database.hdf5
   config = species_config.ini
   input = /path/to/store/data/

A configuration file with default values is automatically created when `species` is initiated by running :class:`~species.core.setup.SpeciesInit` and no configuration file is present in the working folder, for example::

   import species

   species.SpeciesInit(config_path='./')