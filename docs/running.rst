.. _running:

Running species
===============

To get a first impression, this page shows what a typical workflow with `species` looks like.

First, the atmospheric model spectra will be downloaded and added to the database. Available photometry of PZ Tel B will be read from the :class:`~species.data.companions` dictionary and added to the database. Alternatively, the :func:`~species.data.database.Database.add_object` function can be used for manually creating an planet or other type of object in the database. Then, the model spectra are read from the database, interpolated at the provided parameter values, and stored in a :class:`~species.core.box.ModelBox`. The data of PZ Tel B is read from the database and stored in a :class:`~species.core.box.ObjectBox`. The :class:`~species.core.box.Box` objects are given to the :func:`~species.plot.plot_spectrum.plot_spectrum` function, together with the filter names for which the photometry is available.

The following code can be executed from the command line or within a `Jupyter notebook <https://jupyter.org/>`_.

.. code-block:: python

   import species

   # initialize species
   species.SpeciesInit('./')

   # create a database object
   database = species.Database()

   # add the AMES-Cond atmospheric models
   database.add_model(model='ames-cond',
                      wavel_range=(0.1, 6.),
                      teff_range=(2800., 3000.),
                      spec_res=1000.)

   # add the photometry of PZ Tel B that is available in species.data.companions
   database.add_companion(name='PZ Tel B')

   # create an object for reading model spectra
   readmodel = species.ReadModel(model='ames-cond',
                                 wavel_range=(0.1, 6.))

   # interpolate the grid of spectra
   modelbox = readmodel.get_model(model_param={'teff': 2900.,
                                               'logg': 4.5,
                                               'feh': 0.0,
                                               'radius': 2.2,
                                               'distance': 47.13},
                                  spec_res=100.)

   # read the photometry of PZ Tel B
   objectbox = database.get_object(object_name='PZ Tel B')

   # plot the spectrum, photometry, and filter transmission curves
   # the objectbox requires colors for the photometry and spectrum (set to None)
   species.plot_spectrum(boxes=(modelbox, objectbox),
                         filters=objectbox.filter,
                         colors=('black', ('tomato', None)),
                         offset=(-0.08, -0.06),
                         xlim=(0.2, 5.5),
                         ylim=(0., 4.8e-14),
                         legend='upper right',
                         output='spectrum.pdf')

.. image:: https://people.phys.ethz.ch/~stolkert/species/example.png
   :width: 100%
   :align: center
