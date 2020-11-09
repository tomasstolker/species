.. _database:

Database
========

Data which are read from the database, as well as the output of various functions, are stored in :class:`~species.core.box.Box` objects. These can be used directly as input for the plotting functionalities of `species` (see examples in the :ref:`tutorials` section). Alternatively, users can easily extract the content of a :class:`~species.core.box.Box` and process or plot it to their own needs. The :func:`~species.core.box.Box.open_box` function can be used to see which attributes are inside a :class:`~species.core.box.Box`.

The following example will add available photometric data of PZ Tel B to the database, read the data and properties of the companion into an :class:`~species.core.box.ObjectBox`, and list its content.

.. code-block:: python

   import species

   species.SpeciesInit()

   database = species.Database()
   database.add_companion(name='PZ Tel B')

   objectbox = database.get_object(object_name='PZ Tel B')
   objectbox.open_box()

As an example, a dictionary with the apparent magnitudes can be extracted from the :class:`~species.core.box.ObjectBox` in the following way:

.. code-block:: python

   app_mag = objectbox.magnitude

.. important::
   Whenever data is added to the HDF5 database with a name tag that already exists, then existing data is first deleted before the requested data is added to the database. For example, if the AMES-Cond spectra are present in the ``models/ames-cond`` group and ``add_model('ames-cond')`` is executed, then all spectra are first removed from that group before the requested spectra are added. Similarly, if the ``objects/beta Pic b/photometry/Paranal/NACO.Mp`` group contains NACO Mp data of beta Pic b then these data are first removed if that same filter is used by :func:`~species.data.database.Database.add_object`.
