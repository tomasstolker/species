"""
Module with reading functionalities for atmospheric models from petitRADTRANS. See
Mollière et al. 2019 for details about the retrieval code.
"""

import os
import configparser

import numpy as np

from petitRADTRANS import Radtrans
from petitRADTRANS_ck_test_speed import nat_cst as nc
from petitRADTRANS_ck_test_speed import Radtrans as RadtransScatter

from species.core import box, constants
from species.read import read_filter
from species.util import read_util, retrieval_util


class ReadRadtrans:
    """
    Class for reading a model spectrum from the database.
    """

    def __init__(self,
                 line_species=['H2O', 'CO', 'CH4'],
                 cloud_species=[],
                 scattering=False,
                 wavel_range=None,
                 filter_name=None):
        """
        Parameters
        ----------
        line_species : list
            List with the line species.
        cloud_species : list
            List with the cloud species. No clouds are used if an empty list is provided.
        scattering : bool
            Include scattering in the radiative transfer.
        wavel_range : tuple(float, float), None
            Wavelength range (um). The wavelength range is set to 0.8-10 um if set to None or
            not used if ``filter_name`` is not None.
        filter_name : str, None
            Filter name that is used for the wavelength range. The ``wavel_range`` is used if
            ''filter_name`` is set to None.

        Returns
        -------
        NoneType
            None
        """

        self.filter_name = filter_name
        self.wavel_range = wavel_range
        self.scattering = scattering

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()

        elif self.wavel_range is None:
            self.wavel_range = (0.8, 10.)

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        # create mock p-t profile

        temp_params = {}
        temp_params['log_delta'] = -6.
        temp_params['log_gamma'] = 1.
        temp_params['t_int'] = 750.
        temp_params['t_equ'] = 0.
        temp_params['log_p_trans'] = -3.
        temp_params['alpha'] = 0.

        self.pressure, _ = nc.make_press_temp(temp_params)

        # create Radtrans object

        if self.scattering:
            self.rt_object = RadtransScatter(line_species=line_species,
                                             rayleigh_species=['H2', 'He'],
                                             cloud_species=cloud_species,
                                             continuum_opacities=['H2-H2', 'H2-He'],
                                             wlen_bords_micron=wavel_range,
                                             mode='c-k',
                                             test_ck_shuffle_comp=self.scattering,
                                             do_scat_emis=self.scattering)

        else:
            self.rt_object = Radtrans(line_species=line_species,
                                      rayleigh_species=['H2', 'He'],
                                      cloud_species=cloud_species,
                                      continuum_opacities=['H2-H2', 'H2-He'],
                                      wlen_bords_micron=wavel_range,
                                      mode='c-k')

        # create RT arrays of appropriate lengths by using every three pressure points
        self.rt_object.setup_opa_structure(self.pressure[::3])

    def get_model(self,
                  model_param,
                  spec_res=None,
                  wavel_resample=None):
        """
        Function for extracting a model spectrum by linearly interpolating the model grid. The
        parameters values should lie within the boundaries of the grid points that are stored
        in the database. The stored grid points can be inspected with
        :func:`~species.read.read_model.ReadModel.get_points`.

        Parameters
        ----------
        model_param : dict
            Model parameters and values. The values should be within the boundaries of the grid.
            The grid boundaries of the available spectra in the database can be obtained with
            :func:`~species.read.read_model.ReadModel.get_bounds()`.
        spec_res : float, None
            Spectral resolution, achieved by smoothing with a Gaussian kernel. The original
            wavelength points are used if both ``spec_res`` and ``wavel_resample`` are set to None.
        wavel_resample : numpy.ndarray
            Wavelength points (um) to which the spectrum is resampled. Only used if
            ``spec_res`` is set to None.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        if spec_res is not None and wavel_resample is not None:
            raise ValueError('The \'spec_res\' and \'wavel_resample\' parameters can not be used '
                             'simultaneously. Please set one of them to None.')

        if 'tint' in model_param:
            temp, _, _ = retrieval_util.pt_ret_model(
                np.array([model_param['t1'], model_param['t2'], model_param['t3']]),
                10.**model_param['log_delta'], model_param['alpha'], model_param['tint'],
                self.pressure, model_param['metallicity'], model_param['c_o_ratio'])

        else:
            knot_press = np.logspace(np.log10(self.pressure[0]), np.log10(self.pressure[-1]), 15)

            knot_temp = []
            for i in range(15):
                knot_temp.append(model_param[f't{i}'])

            temp = retrieval_util.pt_spline_interp(knot_press, knot_temp, self.pressure)

        if 'log_p_quench' in model_param:
            log_p_quench = model_param['log_p_quench']
        else:
            log_p_quench = -10.

        if self.scattering:
            pass

        else:
            if 'c_o_ratio' in model_param and 'metallicity' in model_param:
                wavelength, flux = retrieval_util.calc_spectrum_clear(
                    self.rt_object, self.pressure, temp, model_param['logg'], model_param['c_o_ratio'],
                    model_param['metallicity'], log_p_quench, None, half=True)

            else:
                abund = {}
                for ab_item in self.rt_object.line_species:
                    abund[ab_item] = model_param[ab_item]

                wavelength, flux = retrieval_util.calc_spectrum_clear(
                    self.rt_object, self.pressure, temp, model_param['logg'], None,
                    None, None, abund, half=True)

        if 'radius' in model_param:
            model_param['mass'] = read_util.get_mass(model_param)

            if 'distance' in model_param:
                scaling = (model_param['radius']*constants.R_JUP)**2 / \
                          (model_param['distance']*constants.PARSEC)**2

                flux *= scaling

        if spec_res is not None:
            # convolve with Gaussian LSF
            flux = retrieval_util.convolve(wavelength, flux, spec_res)

        return box.create_box(boxtype='model',
                              model='petitradtrans',
                              wavelength=wavelength,
                              flux=flux,
                              parameters=model_param,
                              quantity='flux')
