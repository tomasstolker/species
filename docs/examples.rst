.. _examples:

Examples
========

Configuration file
------------------

A configuration file is required in the working folder, for example::

   [species]
   database = species_database.hdf5
   config = species_config.ini
   input = data/

Conversion of photometry units
------------------------------

Calculating the flux density for a given magnitude (and the other way around) is done in the following way::

   import species

   species.SpeciesInit("./")

   synphot = species.SyntheticPhotometry("MKO/NSFCam.J")

   flux, error = synphot.magnitude_to_flux(19.04, 0.40)
   app_mag, _ = synphot.flux_to_magnitude(flux, None)

   print("Apparent flux density [W m-2 micron-1] =", flux)
   print("Apparent magnitude [mag] =", app_mag)

Synthetic photometry of a Planck function
-----------------------------------------

To calculate synthetic photometry from a Planck function for a filter of JWST::

   import species

   species.SpeciesInit('./')

   box = species.get_planck(temperature=200.,
                            radius=1.0,
                            distance=20.,
                            wavelength=(1.0, 30.),
                            specres=1000.)

   synphot = species.SyntheticPhotometry('JWST/MIRI.F1000W')

   phot = synphot.spectrum_to_photometry(box.wavelength, box.flux)
   mag = synphot.spectrum_to_magnitude(box.wavelength, box.flux)

   print('Apparent flux density [W m-2 micron-1] =', phot)
   print('Apparent magnitude [mag] =', mag[0])

Spectral library
----------------

The following code will download the IRTF spectral library and added to the database. Synthetic photometry is then calculated for the first (L0 type) spectrum from the library at the MKO H filter. The spectrum slice is then plotted together with the filter profile and the photometry::

   import species

   species.SpeciesInit('./')

   spectrum = species.ReadSpectrum(spectrum='irtf', filter_name='MKO/NSFCam.H')
   specbox = spectrum.get_spectrum(sptype='L0')

   synphot = species.SyntheticPhotometry(filter_name='MKO/NSFCam.H')

   phot = synphot.spectrum_to_photometry(wavelength=specbox.wavelength,
                                         flux_density=specbox.flux)

   transmission = species.ReadFilter(filter_name='MKO/NSFCam.H')
   wl_mean = transmission.mean_wavelength()

   photbox = species.create_box(boxtype='photometry',
                                name='L0 dwarf',
                                wavelength=wl_mean,
                                flux=phot)

   species.plot_spectrum(boxes=(specbox, photbox),
                         filters=('MKO/NSFCam.H', ),
                         output='photometry.pdf',
                         xlim=(1., 2.5),
                         offset=(-0.08, -0.06))

.. image:: _images/photometry.png
   :width: 80%
   :align: center

Color-magnitude diagram
-----------------------

Here photometric data of 51 Eri b (Rajan et al. 2017) is added to the database. Then a color-magnitude diagram (J-H vs. J) is created from the IRTF spectral library and the data point of 51 Eri b is added to the plot (black square)::

   import species

   species.SpeciesInit('./')

   database = species.Database()
   database.add_companion(name=None)

   object1 = ('beta Pic b', 'Paranal/NACO.J', 'Paranal/NACO.H', 'Paranal/NACO.J')
   object2 = ('51 Eri b', 'MKO/NSFCam.J', 'MKO/NSFCam.H', 'MKO/NSFCam.J')

   colormag = species.ReadColorMagnitude(library=('vlm-plx', ),
                                         filters_color=('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                         filter_mag='MKO/NSFCam.J')

   colorbox = colormag.get_color_magnitude(object_type='field')

   species.plot_color_magnitude(colorbox=colorbox,
                                objects=(object1, object2),
                                label_x='J - H [mag]',
                                label_y='M$_\mathregular{J}$ [mag]',
                                output='color_mag.pdf',
                                legend='upper left')

.. image:: _images/color_mag.png
   :width: 70%
   :align: center

Atmospheric models
------------------

In the last example, the DRIFT-PHOENIX atmospheric models are added to the database. The grid is then interpolated and a spectrum for a given set of parameter values and spectral resolution is computed. The spectrum is then plotted together with several filter curves::

   import species

   species.SpeciesInit('./')

   filters = ('MKO/NSFCam.J', 'MKO/NSFCam.H', 'MKO/NSFCam.K', 'MKO/NSFCam.Lp', 'MKO/NSFCam.Mp')

   model = species.ReadModel(model='drift-phoenix',
                             wavelength=(1.0, 5.0))

   modelbox = model.get_model(model_par={'teff':1510., 'logg':4.1, 'feh':0.1},
                              sampling=('gaussian', (1000, 200.)))

   species.plot_spectrum(boxes=(modelbox, ),
                         filters=filters,
                         output='model1.pdf',
                         offset=(-0.08, -0.07),
                         xlim=(1., 5.),
                         ylim=(0., 1.1e5))

.. image:: _images/model1.png
   :width: 80%
   :align: center

Or, a spectrum with the original spectral resolution can be obtained from the (discrete) model grid::

   modelbox = model.get_data(model_par={'teff':1200., 'logg':4.0, 'feh':0., 'radius':1., 'distance':10.})

   species.plot_spectrum(boxes=(modelbox, ),
                         filters=filters,
                         output='model2.pdf',
                         offset=(-0.08, -0.07),
                         xlim=(1., 5.),
                         ylim=(0., 2.15e-15))

.. image:: _images/model2.png
   :width: 80%
   :align: center