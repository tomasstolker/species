import os
import shutil
import urllib.request

import numpy as np

import species
from species.util import test_util


class TestIsochrone:

    def setup_class(self):
        self.limit = 1e-10
        self.test_path = os.path.dirname(__file__) + '/'

        filename = 'model.AMES-Cond-2000.M-0.0.NaCo.Vega'

        url = 'https://phoenix.ens-lyon.fr/Grids/AMES-Cond/ISOCHRONES/' \
              'model.AMES-Cond-2000.M-0.0.NaCo.Vega'

        urllib.request.urlretrieve(url, filename)

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        os.remove('model.AMES-Cond-2000.M-0.0.NaCo.Vega')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_isochrone(self):
        database = species.Database()
        database.add_isochrones('model.AMES-Cond-2000.M-0.0.NaCo.Vega', 'ames-cond_isochrone')

        database.add_model('ames-cond',
                           wavel_range=(1., 5.),
                           spec_res=100.,
                           teff_range=(2000., 2500))

        read_isochrone = species.ReadIsochrone('ames-cond_isochrone')
        assert read_isochrone.tag == 'ames-cond_isochrone'

    def test_get_isochrone(self):
        read_isochrone = species.ReadIsochrone('ames-cond_isochrone')

        isochrone_box = read_isochrone.get_isochrone(100.,
                                                     np.linspace(10., 100., 10),
                                                     ('J', 'H'),
                                                     'J')

        assert isochrone_box.color.shape == (10, )
        assert isochrone_box.magnitude.shape == (10, )

        assert np.allclose(np.sum(isochrone_box.color), 3.8664061764797584,
                           rtol=self.limit, atol=0.)

        assert np.allclose(np.sum(isochrone_box.magnitude), 110.53956798127874,
                           rtol=self.limit, atol=0.)

        assert np.allclose(np.sum(isochrone_box.teff), 23004.82954860802,
                           rtol=self.limit, atol=0.)

        assert np.allclose(np.sum(isochrone_box.logg), 47.474750185397234,
                           rtol=self.limit, atol=0.)

    def test_get_color_magnitude(self):
        read_isochrone = species.ReadIsochrone('ames-cond_isochrone')

        colormag_box = read_isochrone.get_color_magnitude(100.,
                                                          np.linspace(35., 45., 10),
                                                          'ames-cond',
                                                          ('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                                          'MKO/NSFCam.J')

        assert colormag_box.object_type == 'model'
        assert colormag_box.color.shape == (10, )
        assert colormag_box.magnitude.shape == (10, )

        assert np.allclose(np.sum(colormag_box.color), 2.5051430554779106,
                           rtol=self.limit, atol=0.)

        assert np.allclose(np.sum(colormag_box.magnitude), 109.60059994388715,
                           rtol=self.limit, atol=0.)

        assert np.allclose(np.sum(colormag_box.sptype), 400.,
                           rtol=self.limit, atol=0.)

    def test_get_color_color(self):
        read_isochrone = species.ReadIsochrone('ames-cond_isochrone')

        colorcolor_box = read_isochrone.get_color_color(100.,
                                                        np.linspace(35., 45., 10),
                                                        'ames-cond',
                                                        (('MKO/NSFCam.J', 'MKO/NSFCam.H'),
                                                         ('MKO/NSFCam.H', 'MKO/NSFCam.Ks')))

        assert colorcolor_box.object_type == 'model'
        assert colorcolor_box.color1.shape == (10, )
        assert colorcolor_box.color2.shape == (10, )

        assert np.allclose(np.sum(colorcolor_box.color1), 2.5051430554779106,
                           rtol=self.limit, atol=0.)

        assert np.allclose(np.sum(colorcolor_box.color2), 3.3549617411553143,
                           rtol=self.limit, atol=0.)

        assert np.allclose(np.sum(colorcolor_box.sptype), 400.,
                           rtol=self.limit, atol=0.)
