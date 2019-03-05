.. _examples:

Examples
========

Configuration file
------------------

First, a configuration file has to be created in the working folder, for example::

   [species]
   database = species_database.hdf5
   config = species_config.ini
   input = data/

Photometric calibration
-----------------------

Calculating the flux density for a given magnitude (and the other way around) is done in the following way::

   import species

   species.SpeciesInit('./')

   synphot = species.SyntheticPhotometry('MKO/NSFCam.J')
   flux, error = synphot.magnitude_to_flux(19.04, 0.40)
   app_mag, _ = synphot.flux_to_magnitude(flux, None)

L0 type spectrum
----------------

The following code will download the IRTF spectral library and added to the database. Synthetic photometry is then calculated for the first (L0 type) spectrum from the library at the MKO H filter. The spectrum slice is then plotted together with the filter profile and the photometry::

   import species

   species.SpeciesInit('./')

   spectrum = species.ReadSpectrum('irtf', 'MKO/NSFCam.H')
   specbox = spectrum.get_spectrum()

   synphot = species.SyntheticPhotometry('MKO/NSFCam.H')
   phot = synphot.spectrum_to_photometry(wavelength[0, ], flux_density[0, ])

   transmission = species.ReadFilter('MKO/NSFCam.H')
   wl_mean = transmission.mean_wavelength()

   species.plot_spectrum(wavelength[0, ], flux_density[0, ], ('MKO/NSFCam.H', ), ((wl_mean, phot), ), 'photometry.pdf')

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

   object_cmd = ('51 Eri b', 'MKO/NSFCam.J', 'MKO/NSFCam.H', 'MKO/NSFCam.J')

   colormag = species.ReadColorMagnitude(('vlm-plx', ), ('MKO/NSFCam.J', 'MKO/NSFCam.H'), 'MKO/NSFCam.J')
   colorbox = colormag.get_color_magnitude('field')

   species.plot_color_magnitude(colorbox, (object_cmd, ), 'J - H [mag]', 'M$_\mathregular{J}$ [mag]', 'color_mag.pdf')

.. image:: _images/color_mag.png
   :width: 70%
   :align: center

Atmospheric models
------------------

In the last example, the DRIFT-PHOENIX atmospheric models are added to the database. The grid is then interpolated and a spectrum for a given set of parameter values and spectral resolution is computed. The spectrum is then plotted together with several filter curves::

   import species

   species.SpeciesInit('./')

   filters = ('MKO/NSFCam.J', 'MKO/NSFCam.H', 'MKO/NSFCam.K', 'MKO/NSFCam.Lp', 'MKO/NSFCam.Mp')

   model = species.ReadModel('drift-phoenix', (1.0, 5.0))
   modelbox = model.get_model({'teff':1510., 'logg':4.1, 'feh':0.1}, ('gaussian', (1000, 200.)))
   species.plot_spectrum((modelbox, ), filters, 'model1.pdf', None, offset=(-0.08, -0.07), xlim=(1., 5.), ylim=(2e4, 1.1e5))

.. image:: _images/model1.png
   :width: 80%
   :align: center

Or, a spectrum with the original spectral resolution can be obtained from the (discrete) model grid::

   modelbox = model.get_data({'teff':1200., 'logg':4.0, 'feh':0., 'radius':1., 'distance':10.})
   species.plot_spectrum((modelbox, ), filters, 'model2.pdf', None, offset=(-0.08, -0.07), xlim=(1., 5.), ylim=(0., 2.15e-15))

.. image:: _images/model2.png
   :width: 80%
   :align: center