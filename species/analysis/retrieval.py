"""
Module with functionalities for atmospheric retrieval with petitRADTRANS (Mollière et al. 2019).
More details on the retrieval code are available at https://petitradtrans.readthedocs.io.
"""

import os
import json
import warnings
import configparser

import h5py
import tqdm
import pymultinest
import numpy as np
import matplotlib.pyplot as plt

from rebin_give_width import rebin_give_width

from petitRADTRANS import Radtrans
from petitRADTRANS_ck_test_speed import nat_cst as nc
from petitRADTRANS_ck_test_speed import Radtrans as RadtransScatter

from species.analysis import photometry
from species.data import database
from species.core import box, constants
from species.read import read_object
from species.util import retrieval_util


os.environ['OMP_NUM_THREADS'] = '1'


class AtmosphericRetrieval:
    """
    Class for atmospheric retrieval with petitRADTRANS.
    """

    def __init__(self,
                 object_name,
                 line_species,
                 cloud_species,
                 scattering,
                 output_name):
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        line_species : list
            List with the line species.
        cloud_species : list
            List with the cloud species. No clouds are used if an empty list is provided.
        scattering : bool
            Include scattering in the radiative transfer.
        output_name : str
            Output name that is used for the output files from MultiNest.

        Returns
        -------
        NoneType
            None
        """

        # input parameters

        self.object_name = object_name
        self.line_species = line_species
        self.cloud_species = cloud_species
        self.scattering = scattering
        self.output_name = output_name

        # get object data

        self.object = read_object.ReadObject(self.object_name)
        self.distance = self.object.get_distance()[0]  # [pc]

        species_db = database.Database()
        objectbox = species_db.get_object(object_name, None)
        filters = objectbox.filters

        # get photometric data

        self.objphot = []
        self.synphot = []

        for item in filters:
            obj_phot = self.object.get_photometry(item)
            self.objphot.append((obj_phot[2], obj_phot[3]))

            sphot = photometry.SyntheticPhotometry(item)
            self.synphot.append(sphot)

        if not self.objphot:
            self.objphot = None

        if not self.synphot:
            self.synphot = None

        # get spectroscopic data

        self.spectrum = self.object.get_spectrum()

        if self.spectrum is None:
            raise ValueError('A spectrum is required for the atmospheric retrieval.')

        # set wavelength bins and add to spectrum dictionary

        self.wavel_min = []
        self.wavel_max = []

        for key, value in self.spectrum.items():
            dict_val = list(value)
            wavel_data = dict_val[0][:, 0]

            wavel_bins = np.zeros_like(wavel_data)
            wavel_bins[:-1] = np.diff(wavel_data)
            wavel_bins[-1] = wavel_bins[-2]

            dict_val.append(wavel_bins)
            self.spectrum[key] = dict_val

            # min and max wavelength for Radtrans object

            self.wavel_min.append(wavel_data[0])
            self.wavel_max.append(wavel_data[-1])

        # mock p-t profile for Radtrans object

        temp_params = {}
        temp_params['log_delta'] = -6.
        temp_params['log_gamma'] = 1.
        temp_params['t_int'] = 750.
        temp_params['t_equ'] = 0.
        temp_params['log_p_trans'] = -3.
        temp_params['alpha'] = 0.

        self.pressure, _ = nc.make_press_temp(temp_params)

        # initiate parameter list and counters

        self.parameters = []
        self.count_scale = 0
        self.count_error = 0

    def set_parameters(self,
                       bounds):
        """
        Function to set the list with parameters.

        Parameters
        ----------
        bounds : dict
            Dictionary with the parameter boundaries.

        Returns
        -------
        NoneType
            None
        """

        # generic parameters

        self.parameters.append('logg')
        self.parameters.append('radius')

        # p-t profile parameters

        self.parameters.append('tint')
        self.parameters.append('t1')
        self.parameters.append('t2')
        self.parameters.append('t3')
        self.parameters.append('alpha')
        self.parameters.append('log_delta')

        # abundance parameters

        self.parameters.append('feh')
        self.parameters.append('co')
        self.parameters.append('log_p_quench')

        # cloud parameters

        if len(self.cloud_species) > 0:
            self.parameters.append('fe_fraction')
            self.parameters.append('mgsio3_fraction')
            self.parameters.append('fsed')
            self.parameters.append('kzz')
            self.parameters.append('sigma_lnorm')

        # add the flux scaling parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][0] is not None:
                    self.parameters.append(f'scale_{item}')
                    self.count_scale += 1

        # add the error offset parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][1] is not None:
                    self.parameters.append(f'error_{item}')
                    self.count_error += 1

    def run_multinest(self,
                      tag,
                      bounds,
                      live_points=2000,
                      efficiency=0.05,
                      resume=False,
                      plotting=False):
        """
        Function to sample the posterior distribution with MultiNest. See also
        https://github.com/farhanferoz/MultiNest.

        Parameters
        ----------
        tag : str
            Database tag where the results will be stored.
        bounds : dict
            Dictionary with the prior boundaries.
        live_points : int
            Number of live points.
        efficiency : float
            Sampling efficiency.
        resume : bool
            Resume from a previous run.
        plotting : bool
            Plot sample results for testing.

        Returns
        -------
        NoneType
            None
        """

        # set initial number of parameters (not including the flux scaling and error offeset)

        if len(self.cloud_species) == 0:
            n_param = 11
        else:
            n_param = 16

        # create list with parameters for MultiNest

        self.set_parameters(bounds)

        # delete the cloud parameters from the boundaries dictionary in case of no cloud species

        if len(self.cloud_species) == 0:
            if 'fe_fraction' in bounds:
                del bounds['fe_fraction']

            if 'mgsio3_fraction' in bounds:
                del bounds['mgsio3_fraction']

            if 'fsed' in bounds:
                del bounds['fsed']

            if 'kzz' in bounds:
                del bounds['kzz']

            if 'sigma_lnorm' in bounds:
                del bounds['sigma_lnorm']

        # create Ratrans object

        print('Setting up petitRADTRANS...')

        if self.scattering:
            self.rt_object = RadtransScatter(line_species=self.line_species,
                                             rayleigh_species=['H2', 'He'],
                                             cloud_species=self.cloud_species,
                                             continuum_opacities=['H2-H2', 'H2-He'],
                                             wlen_bords_micron=(0.95*min(self.wavel_min),
                                                                1.05*max(self.wavel_max)),
                                             mode='c-k',
                                             test_ck_shuffle_comp=self.scattering,
                                             do_scat_emis=self.scattering)

        else:
            self.rt_object = Radtrans(line_species=self.line_species,
                                      rayleigh_species=['H2', 'He'],
                                      cloud_species=self.cloud_species,
                                      continuum_opacities=['H2-H2', 'H2-He'],
                                      wlen_bords_micron=(0.95*min(self.wavel_min),
                                                         1.05*max(self.wavel_max)),
                                      mode='c-k')

        # create RT arrays of appropriate lengths by using every three pressure points

        self.rt_object.setup_opa_structure(self.pressure[::3])

        def prior(cube, ndim, nparams):
            """
            Function to transform the unit cube into the parameter cube.

            Parameters
            ----------
            cube : pymultinest.run.LP_c_double
                Unit cube.
            ndim : int
                Number of dimensions.
            nparams : int
                Number of parameters.

            Returns
            -------
            float
                The logarithm of the prior probability.
            """

            # if ndim != nparams:
            #     raise ValueError('The number of dimensions and parameters should be equal.')

            # initiate the logarithm of the prior
            log_prior = 0

            # surface gravity (dex)
            if 'logg' in bounds:
                logg = bounds['logg'][0] + (bounds['logg'][1]-bounds['logg'][0])*cube[0]
            else:
                # default: 2-5.5 dex
                logg = 2. + 3.5*cube[0]

            # planet radius (Rjup)
            if 'radius' in bounds:
                radius = bounds['radius'][0] + (bounds['radius'][1]-bounds['radius'][0])*cube[1]
            else:
                # defaul: 0.8-2 Rjup
                radius = 0.8 + 1.2*cube[1]

            # internal temperature (K) of the Eddington model
            # see Eq. 2 in Mollière et al. in prep.
            if 'tint' in bounds:
                tint = bounds['tint'][0] + (bounds['tint'][1]-bounds['tint'][0])*cube[2]
            else:
                # default: 500-3000 K
                tint = 500.+2500.*cube[2]

            # connection temperature (K)
            t_connect = (3./4.*tint**4.*(0.1+2./3.))**0.25

            # the temperature (K) at temp_3 is scaled down from t_connect
            temp_3 = t_connect*(1-cube[5])

            # the temperature (K) at temp_2 is scaled down from temp_3
            temp_2 = temp_3*(1-cube[4])

            # the temperature (K) at temp_1 is scaled down from temp_2
            temp_1 = temp_2*(1-cube[3])

            # alpha: power law index in tau = delta * press_cgs**alpha
            # see Eq. 1 in Mollière et al. in prep.
            if 'alpha' in bounds:
                alpha = bounds['alpha'][0] + (bounds['alpha'][1]-bounds['alpha'][0])*cube[6]
            else:
                # default: 1-2
                alpha = 1. + cube[6]

            # photospheric pressure (bar)
            # default: 1e-3-1e2 bar
            p_phot = 1e1**(-3. + 5.*cube[7])

            # delta: proportionality factor in tau = delta * press_cgs**alpha
            # see Eq. 1 in Mollière et al. in prep.
            delta = (p_phot*1e6)**(-alpha)
            log_delta = np.log10(delta)

            # metallicity (dex) for the nabla_ad interpolation
            if 'feh' in bounds:
                feh = bounds['feh'][0] + (bounds['feh'][1]-bounds['feh'][0])*cube[8]
            else:
                # default: -1.5-1.5 dex
                feh = -1.5 + 3.*cube[8]

            # carbon-to-oxygen ratio for the nabla_ad interpolation
            if 'co' in bounds:
                co_ratio = bounds['co'][0] + (bounds['co'][1]-bounds['co'][0])*cube[9]
            else:
                # default: 0.1-1.6
                co_ratio = 0.1 + 1.5*cube[9]

            # quench pressure (bar)
            # default: 1e-6-1e3 bar
            log_p_quench = -6. + 9.*cube[10]

            if len(self.cloud_species) > 0:
                # cloud base mass fractions of Fe (iron)
                # relative to the maximum values allowed from elemental abundances
                # see Eq. 3 in Mollière et al. in prep.
                # default: 0.05-1.
                fe_fraction = np.log10(0.05)+(np.log10(1.)-np.log10(0.05))*cube[11]

                # cloud base mass fractions of MgSiO3 (enstatite)
                # relative to the maximum values allowed from elemental abundances
                # see Eq. 3 in Mollière et al. in prep.
                # default: 0.05-1.
                mgsio3_fraction = np.log10(0.05)+(np.log10(1.)-np.log10(0.05))*cube[12]

                # sedimentation parameter
                # ratio of the settling and mixing velocities of the cloud particles
                # see Eq. 3 in Mollière et al. in prep.
                if 'fsed' in bounds:
                    fsed = bounds['fsed'][0] + (bounds['fsed'][1]-bounds['fsed'][0])*cube[13]
                else:
                    # default: 0-10
                    fsed = 10.*cube[13]

                # eddy diffusion coefficient, log(Kzz)
                if 'kzz' in bounds:
                    kzz = bounds['kzz'][0] + (bounds['kzz'][1]-bounds['kzz'][0])*cube[14]
                else:
                    # default: 5-13
                    kzz = 5. + 8.*cube[14]

                # width of the log-normal particle size distribution TODO (um?)
                if 'sigma_lnorm' in bounds:
                    sigma_lnorm = bounds['sigma_lnorm'][0] + (bounds['sigma_lnorm'][1] -
                                                              bounds['sigma_lnorm'][0])*cube[15]
                else:
                    # default: 1.05-3. TODO um (?)
                    sigma_lnorm = 1.05 + 1.95*cube[15]

            # put the new parameter values back into the cube

            cube[0] = logg
            cube[1] = radius
            cube[2] = tint
            cube[3] = temp_1
            cube[4] = temp_2
            cube[5] = temp_3
            cube[6] = alpha
            cube[7] = log_delta
            cube[8] = feh
            cube[9] = co_ratio
            cube[10] = log_p_quench

            if len(self.cloud_species) > 0:
                cube[11] = fe_fraction
                cube[12] = mgsio3_fraction
                cube[13] = fsed
                cube[14] = kzz
                cube[15] = sigma_lnorm

            # add flux scaling parameter if the boundaries are provided

            count = 0

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][0] is not None:
                        cube[n_param+count] = bounds[item][0][0] + \
                            (bounds[item][0][1]-bounds[item][0][0])*cube[n_param+count]

                        count += 1

            # add error inflation parameter if the boundaries are provided

            count = 0

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][1] is not None:
                        cube[n_param+self.count_scale+count] = bounds[item][1][0] + \
                            (bounds[item][1][1]-bounds[item][1][0]) * \
                            cube[n_param+self.count_scale+count]

                        count += 1

            return log_prior

        def loglike(cube, ndim, nparams):
            """
            Function for the logarithm of the likelihood, computed from the parameter cube.

            Parameters
            ----------
            cube : pymultinest.run.LP_c_double
                Unit cube.
            ndim : int
                Number of dimensions.
            nparams : int
                Number of parameters.

            Returns
            -------
            float
                Logarithm of the likelihood function.
            """

            # if ndim != nparams:
            #     raise ValueError('The number of dimensions and parameters should be equal.')

            # mandatory parameters
            logg, radius = cube[0:2]
            tint, temp_1, temp_2, temp_3, alpha, log_delta = cube[2:8]
            feh, co_ratio, log_p_quench = cube[8:11]

            if len(self.cloud_species) > 0:
                # optional cloud parameters
                fe_fraction, mgsio3_fraction, fsed, kzz, sigma_lnorm = cube[11:16]

            # create dictionary with flux scaling parameters

            count = 0
            scaling = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][0] is not None:
                    scaling[item] = cube[n_param+count]
                    count += 1

                else:
                    scaling[item] = 1.

            # create dictionary with error offset parameters

            count = 0
            err_offset = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][1] is not None:
                    err_offset[item] = cube[n_param+self.count_scale+count]
                    count += 1

                else:
                    err_offset[item] = 0.

            # initiate the logarithm of the likelihood
            log_likelihood = 0.

            # create a p-t profile

            # try:
            temp, _, _ = retrieval_util.pt_ret_model(np.array([temp_1, temp_2, temp_3]),
                                                     10.**log_delta,
                                                     alpha,
                                                     tint,
                                                     self.pressure,
                                                     feh,
                                                     co_ratio)

            # except:
            #     return -np.inf

            # return zero probability if the minimum temperature is negative

            if np.min(temp) < 0.:
                return -np.inf

            # calculate the emission spectrum

            # try:
            if len(self.cloud_species) > 0:
                # cloudy atmosphere

                # mass fraction of Fe
                x_fe = retrieval_util.return_XFe(feh, co_ratio)

                # logarithm of the cloud base mass fraction of Fe
                log_x_base_fe = np.log10(1e1**fe_fraction*x_fe)

                # mass fraction of MgSiO3
                x_mgsio3 = retrieval_util.return_XMgSiO3(feh, co_ratio)

                # logarithm of the cloud base mass fraction of MgSiO3
                log_x_base_mgsio3 = np.log10(1e1**mgsio3_fraction*x_mgsio3)

                # wlen_micron, flux_lambda, Pphot_esti, tau_pow, tau_cloud = \
                wlen_micron, flux_lambda = retrieval_util.calc_spectrum_clouds(
                    self.rt_object, self.pressure, temp, co_ratio, feh, log_p_quench,
                    log_x_base_fe, log_x_base_mgsio3, fsed, fsed, kzz,
                    logg, sigma_lnorm, half=True, plotting=plotting)

            else:
                # clear atmosphere

                # log_x_base_fe = -1e10
                # log_x_base_mgsio3 = -1e10
                # fsed = 1e10
                # kzz = -1e10
                # sigma_lnorm = 10.

                wlen_micron, flux_lambda = retrieval_util.calc_spectrum_clear(
                    self.rt_object, self.pressure, temp, logg, co_ratio, feh, log_p_quench,
                    half=True)

            # except:
            #     return -np.inf

            # return zero probability if the spectrum contains NaN values

            if np.sum(np.isnan(flux_lambda)) > 0:
                if len(flux_lambda) > 1:
                    warnings.warn('Spectrum with NaN values encountered.')

                return -np.inf

            # scale the emitted spectrum to the observation
            flux_lambda *= (radius*constants.R_JUP / (self.distance*constants.PARSEC))**2.

            for key, value in self.spectrum.items():
                # get spectrum
                data_wavel = value[0][:, 0]
                data_flux = value[0][:, 1]
                data_error = value[0][:, 2]

                # get inverted covariance matrix
                data_cov_inv = value[2]

                # get spectral resolution
                spec_res = value[3]

                # get wavelength binds
                data_wavel_bins = value[4]

                # fitted error component
                err_fit = 10.**err_offset[key]

                # convolve with Gaussian LSF
                flux_smooth = retrieval_util.convolve(wlen_micron,
                                                      flux_lambda,
                                                      spec_res)

                # resample to the observation
                flux_rebinned = rebin_give_width(wlen_micron,
                                                 flux_smooth,
                                                 data_wavel,
                                                 data_wavel_bins)

                # difference between the observed and modeled spectrum
                diff = flux_rebinned - scaling[key]*data_flux

                if data_cov_inv is not None:
                    # calculate the log-likelihood with the covariance matrix
                    # TODO include err_fit in the covariance matrix
                    log_likelihood += -np.dot(diff, data_cov_inv.dot(diff))/2.

                else:
                    # calculate the log-likelihood without the covariance matrix
                    var_infl = data_error**2.+err_fit**2
                    log_likelihood += -0.5*np.sum(diff**2/var_infl + np.log(2.*np.pi*var_infl))

                if plotting:
                    plt.errorbar(data_wavel, scaling[key]*data_flux, yerr=data_error+err_fit,
                                 marker='o', ms=3, color='tab:blue', markerfacecolor='tab:blue')

                    plt.plot(data_wavel, flux_rebinned, marker='o', ms=3, color='tab:orange')

            if plotting:
                plt.plot(wlen_micron, flux_smooth, color='black', zorder=-20)
                plt.xlabel(r'Wavelength [$\mu$m]')
                plt.ylabel(r'Flux [W m$^{-2}$ $\mu$m$^{-1}$]')
                plt.savefig('spectrum.pdf', bbox_inches='tight')
                plt.clf()

            return log_likelihood

        # store the model parameters in a JSON file

        with open(f'{self.output_name}_params.json', 'w') as json_file:
            json.dump(self.parameters, json_file)

        # store the Radtrans arguments in a JSON file

        radtrans_dict = {}
        radtrans_dict['line_species'] = self.line_species
        radtrans_dict['cloud_species'] = self.cloud_species
        radtrans_dict['scattering'] = self.scattering
        radtrans_dict['distance'] = self.distance

        with open(f'{self.output_name}_radtrans.json', 'w', encoding='utf-8') as json_file:
            json.dump(radtrans_dict, json_file, ensure_ascii=False, indent=4)

        # run the nested sampling with MultiNest

        print('Sampling the posterior distribution with MultiNest...')

        pymultinest.run(loglike,
                        prior,
                        len(self.parameters),
                        outputfiles_basename=f'{self.output_name}_',
                        resume=resume,
                        verbose=True,
                        const_efficiency_mode=True,
                        sampling_efficiency=efficiency,
                        n_live_points=live_points,
                        evidence_tolerance=0.5)
