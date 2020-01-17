.. _examples:

Examples
========

This page contains an incomplete overview of the functionalities that have been implemented in `species`. More examples will be added at a later stage. Feel free to contact Tomas Stolker (see :ref:`about`) for questions regarding its usability for your specific science case.

Converting photometry units
---------------------------

To calculated the flux density from a magnitude (and the other way around):

.. code-block:: python

   import species

   species.SpeciesInit('./')

   synphot = species.SyntheticPhotometry('MKO/NSFCam.J')

   flux, error = synphot.magnitude_to_flux(19.04, 0.40)
   app_mag, _ = synphot.flux_to_magnitude(flux, None)

   print('Apparent flux density [W m-2 micron-1] =', flux)
   print('Apparent magnitude [mag] =', app_mag)

Synthetic photometry
--------------------

To calculate synthetic photometry from a Planck function for given filter:

.. code-block:: python

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

The following code will download the IRTF spectral library and add the spectra to the database. An L0 type spectrum is then read from the database and synthetic photometry is computed for the MKO H filter. The spectrum slice is plotted together with the filter profile and the synthetic photometry:

.. code-block:: python

   import species

   species.SpeciesInit('./')

   spectrum = species.ReadSpectrum(spectrum='irtf', filter_name='MKO/NSFCam.H')
   specbox = spectrum.get_spectrum(sptype=['L0', ])

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

.. image:: https://people.phys.ethz.ch/~stolkert/species/photometry.png
   :width: 80%
   :align: center

Color-magnitude diagram
-----------------------

Here photometric data of 51 Eri b (Rajan et al. 2017) is added to the database. Then a color-magnitude diagram (J-H vs. J) is created from the IRTF spectral library and the data point of 51 Eri b is added to the plot (black square):

.. code-block:: python

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

.. image:: https://people.phys.ethz.ch/~stolkert/species/color_mag.png
   :width: 70%
   :align: center

Atmospheric models
------------------

In the last example, the DRIFT-PHOENIX atmospheric models are added to the database. The grid is then interpolated and a spectrum for a given set of parameter values and spectral resolution is computed. The spectrum is then plotted together with several filter curves:

.. code-block:: python

   import species

   species.SpeciesInit('./')

   filters = ('MKO/NSFCam.J', 'MKO/NSFCam.H', 'MKO/NSFCam.K', 'MKO/NSFCam.Lp', 'MKO/NSFCam.Mp')

   model = species.ReadModel(model='drift-phoenix',
                             wavelength=(1.0, 5.0))

   modelbox = model.get_model(model_par={'teff':1510., 'logg':4.1, 'feh':0.1},
                              specres=200.)

   species.plot_spectrum(boxes=(modelbox, ),
                         filters=filters,
                         output='model1.pdf',
                         offset=(-0.08, -0.07),
                         xlim=(1., 5.),
                         ylim=(0., 1.1e5))

.. image:: https://people.phys.ethz.ch/~stolkert/species/model1.png
   :width: 80%
   :align: center

Or, a spectrum with the original spectral resolution can be obtained from the (discrete) model grid:

.. code-block:: python

   modelbox = model.get_data(model_par={'teff':1200., 'logg':4.0, 'feh':0., 'radius':1., 'distance':10.})

   species.plot_spectrum(boxes=(modelbox, ),
                         filters=filters,
                         output='model2.pdf',
                         offset=(-0.08, -0.07),
                         xlim=(1., 5.),
                         ylim=(0., 2.15e-15))

.. image:: https://people.phys.ethz.ch/~stolkert/species/model2.png
   :width: 80%
   :align: center

Photometric calibration
-----------------------

In this example, the 2MASS photometry of PZ Tel A is fitted with a IRTF spectrum of a G8V type star (which can be downloaded from the IRTF website). The plots show the posterior distribution scaling parameter that was fitted and randomly selected spectra from the posterior distribution with the best-fit synthetic photometry and the observed photometry (which are overlapping). The residuals are shown in terms of the uncertainty of the 2MASS photometry. The following code will run the MCMC, extrapolate the spectrum a bit  and create the plots:

.. code-block:: python

   import species

   species.SpeciesInit('./')

   distance = 47.13 # [pc]

   magnitudes = {'2MASS/2MASS.J':(6.856, 0.021),
                 '2MASS/2MASS.H':(6.486, 0.049),
                 '2MASS/2MASS.Ks':(6.366, 0.024)}

   filters = tuple(magnitudes.keys())

   database = species.Database()

   database.add_object(object_name='PZ Tel A',
                       distance=distance,
                       app_mag=magnitudes)

   database.add_calibration(filename='input/G8V_HD75732.txt',
                            tag='G8V_HD75732')

   fit = species.FitSpectrum(objname='PZ Tel A',
                             filters=None,
                             spectrum='G8V_HD75732',
                             bounds={'scaling':(0., 1e0)})

   fit.run_mcmc(nwalkers=200,
                nsteps=1000,
                guess={'scaling':5e-1},
                tag='pztel')

   species.plot_walkers(tag='pztel',
                        output='plot/walkers.pdf',
                        nsteps=None,
                        offset=(-0.25, -0.08))

   species.plot_posterior(tag='pztel',
                          burnin=500,
                          title=None,
                          output='plot/posterior.pdf',
                          offset=(-0.3, -0.10),
                          title_fmt='.4f')

   objectbox = database.get_object(object_name='PZ Tel A',
                                   filter_id=None)

   samples = database.get_mcmc_spectra(tag='pztel',
                                       burnin=500,
                                       random=30,
                                       wavelength=(0.1, 50.0))

   best = {'scaling':0.1199}

   synphot = species.multi_photometry(datatype='calibration',
                                      spectrum='G8V_HD75732',
                                      filters=filters,
                                      parameters=best)

   residuals = species.get_residuals(datatype='calibration',
                                     spectrum='G8V_HD75732',
                                     parameters=best,
                                     filters=filters,
                                     objectbox=objectbox,
                                     inc_phot=True,
                                     inc_spec=False)

   readcalib = species.ReadCalibration(spectrum='G8V_HD75732',
                                       filter_name=None)

   spectrum = readcalib.get_spectrum(parameters=best,
                                     extrapolate=False,
                                     min_wavelength=2.5)

   species.plot_spectrum(boxes=(samples, spectrum, objectbox, synphot),
                         filters=filters,
                         output='plot/spectrum.pdf',
                         colors=('gray', 'black', ('black', ), 'black', 'tomato', 'teal'),
                         residuals=residuals,
                         xlim=(0.8, 2.5),
                         ylim=(-1.5e-12, 2.1e-11),
                         scale=('linear', 'linear'),
                         title=r'G8V HD75732 - PZ Tel A',
                         offset=(-0.3, -0.08))

If we need to know the magnitude of PZ Tel A in a specific filter (e.g. VLT/NACO Mp), we can create synthetic photometry in the following way:

.. code-block:: python

   synphot = species.SyntheticPhotometry('Paranal/NACO.Mp')
   mag = synphot.spectrum_to_magnitude(spectrum.wavelength, spectrum.flux)
   phot = synphot.spectrum_to_photometry(spectrum.wavelength, spectrum.flux)

   print('NACO Mp [mag] =', mag[0])
   print('NACO Mp [W m-2 micron-1] =', phot)

Which gives:

.. code-block:: none

   NACO Mp [mag] = 6.407877593040467
   NACO Mp [W m-2 micron-1] = 5.9164296e-14

.. image:: https://people.phys.ethz.ch/~stolkert/species/posterior.png
   :width: 40%
   :align: center

.. image:: https://people.phys.ethz.ch/~stolkert/species/spectrum.png
   :width: 90%
   :align: center

Fitting photometry
------------------

In this example we fit the available photometry of beta Pic b with the DRIFT-PHOENIX atmospheric models and sample the posterior distributions of the model parameters with MCMC.

.. code-block:: python

   import species

   species.SpeciesInit('./')

   database = species.Database()

   database.add_model(model='drift-phoenix')

   database.add_companion(name='beta Pic b')

   database.add_filter(filter_id='LCO/VisAO.Ys',
                       filename='../data/VisAO_Ys_filter_curve.dat')

   database.add_object(object_name='beta Pic b',
                       distance=None,
                       app_mag={'LCO/VisAO.Ys': (15.53, 0.34)})  # Males et al. (2014),

   objectbox = database.get_object(object_name='beta Pic b',
                                   filter_id=None,
                                   inc_phot=True,
                                   inc_spec=False)

   fit = species.FitModel(objname='beta Pic b',
                          filters=None,
                          model='drift-phoenix',
                          bounds=None,
                          inc_phot=True,
                          inc_spec=False)

   fit.run_mcmc(nwalkers=200,
                nsteps=1000,
                guess={'teff': 1800, 'logg': None, 'feh': None, 'radius': 1.3},
                tag='betapic',
                prior=('mass', 13., 3.))

   species.plot_walkers(tag='betapic',
                        nsteps=None,
                        offset=(-0.24, -0.09),
                        output='plot/walkers.pdf')

   species.plot_posterior(tag='betapic',
                          burnin=500,
                          title=r'DRIFT-PHOENIX - $\beta$ Pic b',
                          offset=(-0.25, -0.25),
                          limits=((1500., 1920.), (3.4, 4.7), (-0.6, 0.3), (1.1, 1.8)),
                          output='plot/posterior.pdf')

   samples = database.get_mcmc_spectra(tag='betapic',
                                       burnin=500,
                                       random=30,
                                       wavelength=(0.7, 6.5),
                                       specres=50.)

   median = database.get_median_sample('betapic', burnin=500)

   drift = species.ReadModel(model='drift-phoenix', wavelength=(0.7, 6.5))

   model = drift.get_model(model_par=median, specres=50.)

   model = species.add_luminosity(model)

   residuals = species.get_residuals(datatype='model',
                                     spectrum='drift-phoenix',
                                     parameters=median,
                                     filters=None,
                                     objectbox=objectbox,
                                     inc_phot=True,
                                     inc_spec=False)

   synphot = species.multi_photometry(datatype='model',
                                      spectrum='drift-phoenix',
                                      filters=objectbox.filter,
                                      parameters=median)

   species.plot_spectrum(boxes=(samples, model, objectbox, synphot),
                         filters=objectbox.filter,
                         residuals=residuals,
                         colors=('gray', 'tomato', ('black', ), 'black'),
                         xlim=(0.7, 6.0),
                         ylim=(-1.2e-15, 1.3e-14),
                         scale=('linear', 'linear'),
                         title=r'DRIFT-PHOENIX - $\beta$ Pic b',
                         offset=(-0.25, -0.06),
                         output='plot/spectrum.pdf')

.. image:: https://people.phys.ethz.ch/~stolkert/species/betapic.png
   :width: 100%
   :align: center

Isochrone data
--------------

When creating a color-magnitude diagram, various data can be combined such as photometry of isolated brown dwarfs, synthetic photometry of spectra, individual objects, and isochrone data from evolutionary models. Isochrones from the |phoenix| website can be imported into the database after which the related atmospheric models can be used to calculate synthetic photometry for a given age and a range of masses. Alternatively, it is also possible to interpolate the magnitudes of the isochrone data directly. The example below reads and interpolates the AMES-Cond and AMES-Dusty isochrones at 20 Myr, uses these evolutionary data for the computation of synthetic photometry, and plots the isochrones in a color-magnitude diagram together with photometry of field dwarfs and directly imaged companions.

.. code-block:: python

   import species
   import numpy as np

   mass = np.logspace(-1., 4., 100)  # [Mjup]

   species.SpeciesInit('./')

   database = species.Database()

   # Add the relevant data to the database

   database.add_companion(name=None)

   database.add_photometry(library='vlm-plx')
   database.add_photometry(library='leggett')

   database.add_model(model='ames-cond',
                      wavelength=(0.5, 10.),
                      teff=(100., 4000.),
                      specres=1000.)

   database.add_model(model='ames-dusty',
                      wavelength=(0.5, 10.),
                      teff=(100., 4000.),
                      specres=1000.)

   database.add_isochrones(filename='/path/to/model.AMES-dusty.M-0.0.MKO.Vega',
                           tag='iso_dusty')

   database.add_isochrones(filename='/path/to/model.AMES-Cond-2000.M-0.0.MKO.Vega',
                           tag='iso_cond')

   # Create synthetic photometry for isochrones

   readiso1 = species.ReadIsochrone(tag='iso_cond')
   readiso2 = species.ReadIsochrone(tag='iso_dusty')

   modelcolor1 = readiso1.get_color_magnitude(age=20.,
                                              mass=mass,
                                              model='ames-cond',
                                              filters_color=('MKO/NSFCam.H', 'MKO/NSFCam.Lp'),
                                              filter_mag='MKO/NSFCam.Lp')

   modelcolor2 = readiso2.get_color_magnitude(age=20.,
                                              mass=mass,
                                              model='ames-dusty',
                                              filters_color=('MKO/NSFCam.H', 'MKO/NSFCam.Lp'),
                                              filter_mag='MKO/NSFCam.Lp')

   # Directly imaged companions

   objects = (('beta Pic b', 'Paranal/NACO.H', 'Paranal/NACO.Lp', 'Paranal/NACO.Lp'),
              ('HIP 65426 b', 'Paranal/SPHERE.IRDIS_D_H23_2', 'Paranal/NACO.Lp', 'Paranal/NACO.Lp'),
              ('PZ Tel B', 'Paranal/NACO.H', 'Paranal/NACO.Lp', 'Paranal/NACO.Lp'),
              ('HD 206893 B', 'Paranal/SPHERE.IRDIS_B_H', 'Paranal/NACO.Lp', 'Paranal/NACO.Lp'),
              ('51 Eri b', 'MKO/NSFCam.H', 'Keck/NIRC2.Lp', 'Keck/NIRC2.Lp'),
              ('HR 8799 b', 'Keck/NIRC2.H', 'Paranal/NACO.Lp', 'Paranal/NACO.Lp'),
              ('HR 8799 c', 'Keck/NIRC2.H', 'Paranal/NACO.Lp', 'Paranal/NACO.Lp'),
              ('HR 8799 d', 'Keck/NIRC2.H', 'Paranal/NACO.Lp', 'Paranal/NACO.Lp'),
              ('GSC 06214 B', 'MKO/NSFCam.H', 'MKO/NSFCam.Lp', 'MKO/NSFCam.Lp'),
              ('ROXs 42 Bb', 'Keck/NIRC2.H', 'Keck/NIRC2.Lp', 'Keck/NIRC2.Lp'))

   # Field dwarfs from photometric libraries

   colormag = species.ReadColorMagnitude(library=('vlm-plx', 'leggett'),
                                         filters_color=('MKO/NSFCam.H', 'MKO/NSFCam.Lp'),
                                         filter_mag='MKO/NSFCam.Lp')

   colorbox = colormag.get_color_magnitude(object_type='field')

   # Make color-magnitude diagram

   species.plot_color_magnitude(colorbox=colorbox,
                                objects=objects,
                                isochrones=None,
                                models=(modelcolor1, modelcolor2),
                                label_x='H - L$^\prime$ [mag]',
                                label_y='M$_\mathregular{L\prime}$ [mag]',
                                xlim=(-0, 5),
                                ylim=(15.65, 4),
                                offset=(-0.07, -0.1),
                                legend='upper right',
                                output='isochrones.pdf')

.. image:: https://people.phys.ethz.ch/~stolkert/species/isochrone.png
   :width: 60%
   :align: center

.. |phoenix| raw:: html

   <a href="https://phoenix.ens-lyon.fr/Grids/" target="_blank">PHOENIX</a>
