"""
Module with reading functionalities for atmospheric models from petitRADTRANS. See
Mollière et al. 2019 for details about the retrieval code.
"""

import os
import configparser

from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS_ck_test_speed.radtrans import Radtrans as RadtransScatter

from species.analysis import photometry
from species.core import box, constants
from species.read import read_filter
from species.util import read_util, retrieval_util


class ReadRadtrans:
    """
    Class for reading a model spectrum from the database.
    """

    @typechecked
    def __init__(self,
                 line_species: Optional[List[str]] = None,
                 cloud_species: Optional[List[str]] = None,
                 scattering: bool = False,
                 wavel_range: Optional[Tuple[float, float]] = None,
                 filter_name: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        line_species : list, None
            List with the line species. No line species are used if set to ``None``.
        cloud_species : list, None
            List with the cloud species. No clouds are used if set to ``None``.
        scattering : bool
            Include scattering in the radiative transfer.
        wavel_range : tuple(float, float), None
            Wavelength range (um). The wavelength range is set to 0.8-10 um if set to ``None`` or
            not used if ``filter_name`` is not ``None``.
        filter_name : str, None
            Filter name that is used for the wavelength range. The ``wavel_range`` is used if
            ''filter_name`` is set to ``None``.

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
            self.wavel_range = (0.9*self.wavel_range[0], 1.2*self.wavel_range[1])

        elif self.wavel_range is None:
            self.wavel_range = (0.8, 10.)

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        if line_species is None:
            self.line_species = []
        else:
            self.line_species = line_species

        if cloud_species is None:
            self.cloud_species = []
            n_pressure = 180

        else:
            self.cloud_species = cloud_species
            # n_pressure = 1440
            n_pressure = 180

        # create 180 pressure layers in log space
        self.pressure = np.logspace(-6, 3, n_pressure)

        # create Radtrans object

        if self.scattering:
            self.rt_object = RadtransScatter(line_species=self.line_species,
                                             rayleigh_species=['H2', 'He'],
                                             cloud_species=self.cloud_species,
                                             continuum_opacities=['H2-H2', 'H2-He'],
                                             wlen_bords_micron=self.wavel_range,
                                             mode='c-k',
                                             test_ck_shuffle_comp=self.scattering,
                                             do_scat_emis=self.scattering)

        else:
            self.rt_object = Radtrans(line_species=self.line_species,
                                      rayleigh_species=['H2', 'He'],
                                      cloud_species=self.cloud_species,
                                      continuum_opacities=['H2-H2', 'H2-He'],
                                      wlen_bords_micron=self.wavel_range,
                                      mode='c-k')

        # create RT arrays of appropriate lengths by using every three pressure points
        self.rt_object.setup_opa_structure(self.pressure[::3])

    @typechecked
    def get_model(self,
                  model_param: dict,
                  spec_res: Optional[float] = None,
                  wavel_resample: Optional[np.ndarray] = None,
                  plot_contribution: Optional[str] = None) -> box.ModelBox:
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
            wavelength points are used if both ``spec_res`` and ``wavel_resample`` are set to
            ``None``.
        wavel_resample : numpy.ndarray
            Wavelength points (um) to which the spectrum is resampled. Only used if
            ``spec_res`` is set to ``None``.
        plot_contribution : str, None
            Filename for the plot of the emission contribution function. The plot is not created if
            set to ``None``.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        if spec_res is not None and wavel_resample is not None:
            raise ValueError('The \'spec_res\' and \'wavel_resample\' parameters can not be used '
                             'simultaneously. Please set one of them to None.')

        if plot_contribution:
            contribution = True
        else:
            contribution = False

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

            knot_temp = np.asarray(knot_temp)

            temp = retrieval_util.pt_spline_interp(knot_press, knot_temp, self.pressure)

        if 'log_p_quench' in model_param:
            log_p_quench = model_param['log_p_quench']
        else:
            log_p_quench = -10.

        if len(self.cloud_species) > 0:

            cloud_fractions = {}
            for item in self.cloud_species:
                cloud_fractions[item] = model_param[f'{item[:-3].lower()}_fraction']

            log_x_base = retrieval_util.log_x_cloud_base(model_param['c_o_ratio'],
                                                         model_param['metallicity'],
                                                         cloud_fractions)

            wavelength, flux, emission_contr = retrieval_util.calc_spectrum_clouds(
                self.rt_object, self.pressure, temp, model_param['c_o_ratio'],
                model_param['metallicity'], log_p_quench, log_x_base, model_param['fsed'],
                model_param['kzz'], model_param['logg'], model_param['sigma_lnorm'],
                chemistry='equilibrium', half=True, plotting=False, contribution=contribution)

        elif 'c_o_ratio' in model_param and 'metallicity' in model_param:
            wavelength, flux, emission_contr = retrieval_util.calc_spectrum_clear(
                self.rt_object, self.pressure, temp, model_param['logg'],
                model_param['c_o_ratio'], model_param['metallicity'], log_p_quench,
                None, half=True, chemistry='equilibrium', contribution=contribution)

        else:
            abund = {}
            for ab_item in self.rt_object.line_species:
                abund[ab_item] = model_param[ab_item]

            wavelength, flux, emission_contr = retrieval_util.calc_spectrum_clear(
                self.rt_object, self.pressure, temp, model_param['logg'], None,
                None, None, abund, half=True, chemistry='free', contribution=contribution)

        if 'radius' in model_param:
            model_param['mass'] = read_util.get_mass(model_param)

            if 'distance' in model_param:
                scaling = (model_param['radius']*constants.R_JUP)**2 / \
                          (model_param['distance']*constants.PARSEC)**2

                flux *= scaling

        if spec_res is not None:
            # convolve with Gaussian LSF
            flux = retrieval_util.convolve(wavelength, flux, spec_res)

        if plot_contribution is not None:
            mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
            mpl.rcParams['font.family'] = 'serif'

            plt.rc('axes', edgecolor='black', linewidth=2.5)

            plt.figure(1, figsize=(8., 4.))
            gridsp = mpl.gridspec.GridSpec(1, 1)
            gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

            ax = plt.subplot(gridsp[0, 0])

            ax.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                           direction='in', width=1, length=5, labelsize=12, top=True,
                           bottom=True, left=True, right=True)

            ax.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                           direction='in', width=1, length=3, labelsize=12, top=True,
                           bottom=True, left=True, right=True)

            ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=13)
            ax.set_ylabel('Pressure (bar)', fontsize=13)

            ax.get_xaxis().set_label_coords(0.5, -0.09)
            ax.get_yaxis().set_label_coords(-0.07, 0.5)

            ax.set_yscale('log')
            ax.set_xscale('log')

            xx_grid, yy_grid = np.meshgrid(wavelength, self.pressure[::3])
            ax.contourf(xx_grid, yy_grid, emission_contr, 30, cmap=plt.cm.bone_r)

            ax.set_xlim(np.amin(wavelength), np.amax(wavelength))
            ax.set_ylim(np.amax(self.pressure[::3]), np.amin(self.pressure[::3]))

            plt.savefig(plot_contribution, bbox_inches='tight')
            plt.clf()
            plt.close()

        return box.create_box(boxtype='model',
                              model='petitradtrans',
                              wavelength=wavelength,
                              flux=flux,
                              parameters=model_param,
                              quantity='flux')

    @typechecked
    def get_flux(self,
                 model_param: Dict[str, float]) -> Tuple[float, None]:
        """
        Function for calculating the average flux density for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Model parameters and values.

        Returns
        -------
        float
            Average flux (W m-2 um-1).
        float, None
            Uncertainty (W m-2 um-1), which is set to ``None``.
        """

        spectrum = self.get_model(model_param)

        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(spectrum.wavelength, spectrum.flux)
