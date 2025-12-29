.. _database:

Database
========

The central component of a workflow is the :class:`~species.data.database.Database`. This object is used to store various types of data in the `species_database.hdf5` file, which is located in the working folder or another location set in the `species_config.ini` file. Data needs to be added to a database only once, so databases can be conveniently reused by other workflows.

Want to know which data and attributes have been stored in the database? The :func:`~species.data.database.Database.list_content` method of :class:`~species.data.database.Database` is used for listing the content of the HDF5 file:

.. code-block:: python

   database.list_content()

Data which are read from the database, as well as the output of various functions, are stored in :class:`~species.core.box.Box` objects. These can be used as input for the plotting functionalities of `species` (see examples in the :ref:`tutorials` section). Alternatively, users can extract the content of a :class:`~species.core.box.Box` and process it to their own needs.

The following example will add available photometric data of HR 8799 b to the database, and read the data of the companion into an :class:`~species.core.box.ObjectBox`. The :func:`~species.core.box.Box.open_box` method is used for listing the data in a :class:`~species.core.box.Box`.

.. code-block:: python

   import species

   species.SpeciesInit()

   database = species.Database()
   database.add_companion(name='HR 8799 b')

   objectbox = database.get_object(object_name='HR 8799 b')
   objectbox.open_box()

Data are easily extracted as the attributes of a :class:`~species.core.box.Box` object. For example, in this example a dictionary with the apparent magnitudes is extracted from the :class:`~species.core.box.ObjectBox`:

.. code-block:: python

   app_mag = objectbox.magnitude

To delete a group or dataset from the HDF5 file, there is the :func:`~species.data.database.Database.delete_data` method which takes the path in the HDF5 structure as argument. For example, to remove all previously added photometric data of HR 8799 b:

.. code-block:: python

   database.delete_data("objects/HR 8799 b/photometry")

.. important::
   When data is added to the HDF5 database by an existing name tag, then the existing data is first deleted before the requested data is added to the database. For example, when the AMES-Dusty spectra are present in the ``models/ames-dusty`` group and ``add_model('ames-dusty')`` is executed, then all spectra are first removed from that group before the requested spectra are added. Similarly, if the ``objects/beta Pic b/photometry/Paranal/NACO.Mp`` group contains NACO :math:`M'` data of beta Pic b then these data are first removed if that same filter is used by :func:`~species.data.database.Database.add_object`.
