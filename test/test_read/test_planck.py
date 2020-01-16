import os
import shutil
import pytest

import numpy as np

import species
from species.util import test_util


class TestPlanck:

    def setup_class(self):
        self.limit = 1e-10

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config()
        species.SpeciesInit('./')

    def test_read_planck(self):
        read_planck = species.ReadPlanck('MKO/NSFCam.J')
        assert read_planck.wavel_range == pytest.approx((1.1308, 1.3812))

        read_planck = species.ReadPlanck((1., 5.))
        assert read_planck.wavel_range == (1., 5.)

    def test_get_spectrum(self):
        read_planck = species.ReadPlanck('MKO/NSFCam.J')
        modelbox = read_planck.get_spectrum({'teff': 2000., 'radius': 1., 'distance': 10.}, 100.)

        assert modelbox.model == 'planck'
        assert modelbox.wavelength.shape == (22, )
        assert modelbox.flux.shape == (22, )

        assert np.allclose(np.sum(modelbox.wavelength), 27.67246963534303, rtol=self.limit, atol=0.)
        assert np.allclose(np.sum(modelbox.flux), 1.827363592502394e-12, rtol=self.limit, atol=0.)

    def test_get_flux(self):
        read_planck = species.ReadPlanck('MKO/NSFCam.J')

        flux = read_planck.get_flux({'teff': 2000., 'radius': 1., 'distance': 10.})
        assert flux == 8.322445907073985e-14

        synphot = species.SyntheticPhotometry('MKO/NSFCam.J')
        flux = read_planck.get_flux({'teff': 2000., 'radius': 1., 'distance': 10.}, synphot=synphot)
        assert flux == 8.322445907073985e-14
