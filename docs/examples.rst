.. _examples:

Examples
========

Photometric calibration
-----------------------

To compute the flux density for a given apparent magnitude and uncertainty, and the other way around, is done in the following way::

   from species import *

   SpeciesInit("./", "./data")

   synphot = SyntheticPhotometry("MKO/NSFCam.J")

   flux, error = synphot.magnitude_to_flux(18.0, 0.5)
   app_mag, _ = synphot.flux_to_magnitude(flux, None)

L0 type spectrum
----------------

The following code will create a database in the current folder and download the IRTF spectral library to a separate data folder. The spectra are then read from the database and synthetic photometry for the MKO H filter is computed for the first (L0 type) spectrum from the library. The spectrum is plotted together with the filter profile and the photometry::

   from species import *

   SpeciesInit("./", "./data")

   spectrum = ReadSpectrum("irtf", "MKO/NSFCam.H")
   wavelength, flux_density = spectrum.get_spectrum()

   synphot = SyntheticPhotometry("MKO/NSFCam.H")
   phot = synphot.spectrum_to_photometry(wavelength[0, ], flux_density[0, ])

   transmission = ReadFilter("MKO/NSFCam.H")
   wl_mean = transmission.mean_wavelength()

   plot_spectrum(wavelength[0, ], flux_density[0, ], ("MKO/NSFCam.H", ), ((wl_mean, phot), ), "photometry.pdf")

.. image:: _images/photometry.png
   :width: 80%
   :align: center

Color-magnitude diagram
-----------------------

In the following example we will add photometric data of 51 Eri b (Rajan et al. 2017) to the database. Then we create a color-magnitude diagram (J-H vs. J) from the IRTF spectral library and add the data point of 51 Eri b (black square)::

   from species import *

   SpeciesInit("./", "./data")

   magnitudes = {"MKO/NSFCam.J":19.04, "MKO/NSFCam.H":18.99, "MKO/NSFCam.K":18.67, "Keck/NIRC2.Lp":16.20, "Keck/NIRC2.Mp":16.1}

   database = Database()
   database.add_object("51 Eri b", 29.43, magnitudes)

   object_cmd1 = (("51 Eri b", "MKO/NSFCam.J", "MKO/NSFCam.H", "MKO/NSFCam.J"), )

   colormag = ReadColorMagnitude(("MKO/NSFCam.J", "MKO/NSFCam.H"), "MKO/NSFCam.J")
   color, mag, sptype = colormag.get_color_magnitude("field")
   plot_color_magnitude(color, mag, sptype, object_cmd1, "J - H [mag]", "J [mag]", "color_mag_j-h_j.pdf")

.. image:: _images/color_mag.png
   :width: 70%
   :align: center

Atmospheric models
------------------

In the last example we add the DRIFT-PHOENIX atmospheric models to the database. Then the grid will be interpolated and we will obtain spectrum for a given set of parameter values and spectral resolution, and plot the spectrum together with several filter curves::

   from species import *

   SpeciesInit("./", "./data")

   filters = ("MKO/NSFCam.J", "MKO/NSFCam.H", "MKO/NSFCam.K", "MKO/NSFCam.Lp", "MKO/NSFCam.Mp")

   model = ReadModel("drift-phoenix", (1.0, 5.0))
   spectrum = model.get_model({'teff':1510., 'logg':4.1, 'feh':0.1}, 100.)
   plot_spectrum(spectrum[0, ], spectrum[1, ], filters, None, "drift-phoenix_filters.pdf")

.. image:: _images/drift-phoenix_filters.png
   :width: 80%
   :align: center

Or, we can also take a spectrum from the (discrete) grid with the original spectral resolution::

   model = ReadModel("drift-phoenix", (1., 5.))
   spectrum = model.get_data({'teff':1200., 'logg':4.0, 'feh':0., 'radius':1., 'distance':10.})
   plot_spectrum(spectrum[0, ], spectrum[1, ], filters, None, "drift-phoenix_teff_1200_logg4.0_feh_0.0.pdf")

.. image:: _images/drift-phoenix_full.png
   :width: 80%
   :align: center