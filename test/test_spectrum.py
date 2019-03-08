import os
import shutil

import numpy as np

import species
from . import tools


class TestPhotometry:

    def setup_class(self):
        self.limit = 1e-10

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        os.remove('photometry.pdf')
        shutil.rmtree("data/")

    def test_species_init(self):
        tools.create_config()
        species.SpeciesInit("./")

    def test_spectral_library(self):
        spectrum = species.ReadSpectrum(spectrum='irtf', filter_name='MKO/NSFCam.H')
        specbox = spectrum.get_spectrum(sptype='L0')

        wavelength = specbox.wavelength[0]
        flux = specbox.flux[0]

        assert wavelength.shape == (1063, )
        assert flux.shape == (1063, )

        assert np.allclose(np.sum(wavelength), 1692.8604, rtol=1e-7, atol=0.)
        assert np.allclose(np.sum(flux), 4.5681937e-11, rtol=1e-8, atol=0.)

        spectrum = species.ReadSpectrum(spectrum='irtf', filter_name='MKO/NSFCam.H')
        specbox = spectrum.get_spectrum(sptype='L0')

        synphot = species.SyntheticPhotometry(filter_name='MKO/NSFCam.H')
        phot = synphot.spectrum_to_photometry(wavelength=wavelength,
                                              flux_density=flux)

        assert np.allclose(phot, 4.567475834590298e-14, rtol=self.limit, atol=0.)

        transmission = species.ReadFilter(filter_name='MKO/NSFCam.H')
        wl_mean = transmission.mean_wavelength()

        assert np.allclose(wl_mean, 1.6298258, rtol=1e-7, atol=0.)

        photbox = species.create_box(boxtype='photometry',
                                     name='L0 dwarf',
                                     wavelength=wl_mean,
                                     flux=phot)

        assert photbox.name == 'L0 dwarf'
        assert np.allclose(photbox.wavelength, 1.6298258, rtol=1e-7, atol=0.)
        assert np.allclose(photbox.flux, 4.567475834590298e-14, rtol=self.limit, atol=0.)

        species.plot_spectrum(boxes=(specbox, photbox),
                              filters=('MKO/NSFCam.H', ),
                              output='photometry.pdf',
                              xlim=(1., 2.5),
                              offset=(-0.08, -0.06))

        assert os.path.exists('photometry.pdf')
