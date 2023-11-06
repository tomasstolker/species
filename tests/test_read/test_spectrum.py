import os
import shutil

import pytest
import numpy as np

from species import SpeciesInit
# from species.data.database import Database
# from species.read.read_spectrum import ReadSpectrum
# from species.plot.plot_spectrum import plot_spectrum
from species.util import test_util


class TestSpectrum:
    def setup_class(self):
        self.limit = 1e-8
        self.test_path = os.path.dirname(__file__) + "/"

    def teardown_class(self):
        os.remove("species_database.hdf5")
        os.remove("species_config.ini")
        # os.remove("spectrum.pdf")
        shutil.rmtree("data/")

    def test_species_init(self):
        test_util.create_config("./")
        SpeciesInit()

    # def test_read_spectrum(self):
    #     database = Database()
    #
    #     with pytest.warns(UserWarning):
    #         database.add_spectrum(
    #             "irtf",
    #             sptypes=[
    #                 "L",
    #             ],
    #         )
    #
    #     read_spectrum = ReadSpectrum("irtf", filter_name="MKO/NSFCam.H")
    #     assert read_spectrum.wavel_range == pytest.approx(
    #         (1.382, 1.8656), rel=1e-6, abs=0.0
    #     )
    #
    # def test_get_spectrum(self):
    #     read_spectrum = ReadSpectrum("irtf", filter_name="MKO/NSFCam.H")
    #     spec_box = read_spectrum.get_spectrum(
    #         sptypes=[
    #             "L0",
    #         ],
    #         exclude_nan=True,
    #     )
    #
    #     assert spec_box.wavelength[0].shape == (1063,)
    #     assert spec_box.flux[0].shape == (1063,)
    #
    #     assert np.sum(spec_box.wavelength[0]) == pytest.approx(
    #         1692.8604, rel=1e-7, abs=0.0
    #     )
    #     assert np.sum(spec_box.flux[0]) == pytest.approx(
    #         4.5681937e-11, rel=1e-7, abs=0.0
    #     )
    #
    #     plot_spectrum(
    #         boxes=[
    #             spec_box,
    #         ],
    #         filters=[
    #             "MKO/NSFCam.H",
    #         ],
    #         output="spectrum.pdf",
    #         xlim=(1.0, 2.5),
    #         offset=(-0.08, -0.06),
    #     )
    #
    #     assert os.path.exists("spectrum.pdf")
