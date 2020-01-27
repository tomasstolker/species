import os
import shutil

import numpy as np

import species
from species.util import test_util


class TestModel:

    def setup_class(self):
        self.limit = 1e-10
        self.test_path = os.path.dirname(__file__) + '/'
        self.model_param = {'teff': 2200., 'logg': 4.5, 'radius': 1., 'distance': 10.}

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_model(self):
        database = species.Database()
        database.add_model('ames-cond', teff_range=(2000., 2500))

        read_model = species.ReadModel('ames-cond')
        assert read_model.model == 'ames-cond'

    def test_get_model(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        model_box = read_model.get_model(self.model_param, spec_res=100., magnitude=False)

        assert np.allclose(np.sum(model_box.wavelength), 410.47345, rtol=1e-8, atol=0.)
        assert np.allclose(np.sum(model_box.flux), 8.032991225624241e-12, rtol=self.limit, atol=0.)

        model_box = read_model.get_model(self.model_param, spec_res=100., magnitude=True)

        assert np.allclose(np.sum(model_box.wavelength), 410.47345, rtol=1e-8, atol=0.)
        assert np.allclose(np.sum(model_box.flux), 2860.100473487432, rtol=self.limit, atol=0.)

    def test_get_data(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        model_box = read_model.get_data(self.model_param)

        assert np.allclose(np.sum(model_box.wavelength), 444.07788, rtol=1e-8, atol=0.)
        assert np.allclose(np.sum(model_box.flux), 8.2915255e-12, rtol=1e-8, atol=0.)

    def test_get_flux(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        flux = read_model.get_flux(self.model_param)

        assert np.allclose(flux, 3.4948402850252455e-14, rtol=self.limit, atol=0.)

    def test_get_magnitude(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        magnitude = read_model.get_magnitude(self.model_param)

        assert np.allclose(magnitude[0], 11.306912946044667, rtol=self.limit, atol=0.)
        assert np.allclose(magnitude[1], 11.306912946044667, rtol=self.limit, atol=0.)

    def test_get_bounds(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        bounds = read_model.get_bounds()

        assert bounds['teff'] == (2000., 2500.)
        assert bounds['logg'] == (0., 6.)

    def test_get_wavelengths(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        wavelengths = read_model.get_wavelengths()

        assert np.allclose(np.sum(wavelengths), 99995.52, rtol=1e-7, atol=0.)

    def test_get_points(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        points = read_model.get_points()

        assert np.sum(points['teff']) == 13500.
        assert np.sum(points['logg']) == 39.

    def test_get_parameters(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        parameters = read_model.get_parameters()

        assert parameters == ['teff', 'logg']
