import os
import shutil
import pytest

import numpy as np

import species
from species.util import test_util


class TestCalibration:

    def setup_class(self):
        self.limit = 1e-10
        self.test_path = os.path.dirname(__file__) + '/'
        self.model_param = {'scaling': 1.}

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_calibration(self):
        database = species.Database()
        database.add_spectrum('vega')

        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.H')
        assert read_calib.wavel_range == pytest.approx((1.44, 1.88))

    def test_interpolate_spectrum(self):
        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.H')
        interp_spectrum = read_calib.interpolate_spectrum()

        assert interp_spectrum.x.shape == (268, )
        assert interp_spectrum.y.shape == (268, )

        assert np.allclose(np.sum(interp_spectrum.x), 442.18668, rtol=1e-8, atol=0.)
        assert np.allclose(np.sum(interp_spectrum.y), 3.1471103e-07, rtol=1e-8, atol=0.)

    def test_resample_spectrum(self):
        read_calib = species.ReadCalibration('vega')
        spec_box = read_calib.resample_spectrum(np.linspace(1., 2., 10), apply_mask=True)

        assert np.sum(spec_box.wavelength) == 15.
        assert np.allclose(np.sum(spec_box.flux), 2.288734689409295e-08, rtol=self.limit, atol=0.)

    def test_get_spectrum(self):
        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.Lp')
        spec_box = read_calib.get_spectrum(self.model_param, apply_mask=True, spec_res=100.)

        assert np.allclose(np.sum(spec_box.wavelength), 79.7966233185033, rtol=self.limit, atol=0.)
        assert np.allclose(np.sum(spec_box.flux), 1.0942469537490926e-09, rtol=self.limit, atol=0.)

        with pytest.warns(UserWarning) as warning:
            spec_box = read_calib.get_spectrum(self.model_param, apply_mask=True, spec_res=1000.,
                                               extrapolate=True, min_wavelength=None)

        assert len(warning) == 2

        assert np.allclose(np.sum(spec_box.wavelength), 2594.77301502914, rtol=self.limit, atol=0.)
        assert np.allclose(np.sum(spec_box.flux), 1.519444387600341e-08, rtol=self.limit, atol=0.)

    def test_get_flux(self):
        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.H')
        flux = read_calib.get_flux(model_param=self.model_param)

        assert np.allclose(flux, 1.1329024e-09, rtol=1e-7, atol=0.)

    def test_get_magnitude(self):
        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.H')
        magnitude = read_calib.get_magnitude(model_param=self.model_param)

        assert magnitude[0] == 0.03
        assert magnitude[1] is None
