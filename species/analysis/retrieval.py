"""
Module with functionalities for atmospheric retrieval with petitRADTRANS (Mollière et al. 2019).
More details on the retrieval code are available at https://petitradtrans.readthedocs.io.
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'

import json
import warnings

import pymultinest
import numpy as np
import matplotlib.pyplot as plt

import rebin_give_width as rgw

from petitRADTRANS_ck_test_speed import Radtrans
from petitRADTRANS_ck_test_speed import nat_cst as nc

from species.analysis import photometry
from species.data import database
from species.read import read_object
from species.util import retrieval_util


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
            Account for scattering in the radiative transfer.
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

        wavel_min = []
        wavel_max = []

        for key, value in self.spectrum.items():
            dict_val = list(value)
            wavel_data = dict_val[0][:, 0]

            wavel_bins = np.zeros_like(wavel_data)
            wavel_bins[:-1] = np.diff(wavel_data)
            wavel_bins[-1] = wavel_bins[-2]

            dict_val.append(wavel_bins)
            self.spectrum[key] = dict_val

            # min and max wavelength for Radtrans object

            wavel_min.append(wavel_data[0])
            wavel_max.append(wavel_data[-1])

        # mock p-t profile for Radtrans object

        temp_params = {}
        temp_params['log_delta'] = -6.
        temp_params['log_gamma'] = 1.
        temp_params['t_int'] = 750.
        temp_params['t_equ'] = 0.
        temp_params['log_p_trans'] = -3.
        temp_params['alpha'] = 0.

        self.pressure, _ = nc.make_press_temp(temp_params)

        # Ratrans object

        self.rt_object = Radtrans(line_species=self.line_species,
                                  rayleigh_species=['H2', 'He'],
                                  cloud_species=self.cloud_species,
                                  continuum_opacities=['H2-H2', 'H2-He'],
                                  wlen_bords_micron=(0.99*min(wavel_min), 1.01*max(wavel_max)),
                                  mode='c-k',
                                  test_ck_shuffle_comp=self.scattering,
                                  do_scat_emis=self.scattering)

        # create RT arrays of appropriate lengths by using every three pressure points

        self.rt_object.setup_opa_structure(self.pressure[::3])

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

        self.parameters = []

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

        self.parameters.append('co')
        self.parameters.append('feh')
        self.parameters.append('log_p_quench')

        # cloud parameters

        if len(self.cloud_species) > 0:
            self.parameters.append('fe_fraction')
            self.parameters.append('mgsio3_fraction')
            self.parameters.append('fsed')
            self.parameters.append('kzz')
            self.parameters.append('sigma_lnorm')

        # add the flux scaling and error offset parameters

        self.count_scale = 0
        self.count_error = 0

        for item in self.spectrum.keys():
            if item in bounds:
                if bounds[item][0] is not None:
                    self.parameters.append(f'scale_{item}')
                    self.count_scale += 1

                if bounds[item][1] is not None:
                    self.parameters.append(f'error_{item}')
                    self.count_error += 1

    def run_multinest(self,
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
        bounds : dict
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
            self.n_param = 11
        else:
            self.n_param = 16

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

            # the temperature (K) at t3 is scaled down from t_connect
            t3 = t_connect*(1-cube[5])

            # the temperature (K) at t2 is scaled down from t3
            t2 = t3*(1-cube[4])

            # the temperature (K) at t1 is scaled down from t2
            t1 = t2*(1-cube[3])

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
                co = bounds['co'][0] + (bounds['co'][1]-bounds['co'][0])*cube[9]
            else:
                # default: 0.1-1.6
                co = 0.1 + 1.5*cube[9]

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
                    sigma_lnorm = bounds['sigma_lnorm'][0] + (bounds['sigma_lnorm'][1]-bounds['sigma_lnorm'][0])*cube[15]
                else:
                    # default: 1.05-3. TODO um (?)
                    sigma_lnorm = 1.05 + 1.95*cube[15]

            # put the new parameter values back into the cube

            cube[0] = logg
            cube[1] = radius
            cube[2] = tint
            cube[3] = t1
            cube[4] = t2
            cube[5] = t3
            cube[6] = alpha
            cube[7] = log_delta
            cube[8] = feh
            cube[9] = co
            cube[10] = log_p_quench

            if len(self.cloud_species) > 0:
                cube[11] = fe_fraction
                cube[12] = mgsio3_fraction
                cube[13] = fsed
                cube[14] = kzz
                cube[15] = sigma_lnorm

            # add flux scaling parameter if the boundaries are provided

            count = 0

            for item in self.spectrum.keys():
                if item in bounds:
                    if bounds[item][0] is not None:
                        cube[self.n_param+count] = bounds[item][0][0] + \
                            (bounds[item][0][1]-bounds[item][0][0])*cube[self.n_param+count]

                        count += 1

            # add error inflation parameter if the boundaries are provided

            count = 0

            for item in self.spectrum.keys():
                if item in bounds:
                    if bounds[item][1] is not None:
                        cube[self.n_param+self.count_scale+count] = bounds[item][1][0] + \
                            (bounds[item][1][1]-bounds[item][1][0])*cube[self.n_param+self.count_scale+count]

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

            # mandatory parameters
            logg, radius, tint, t1, t2, t3, alpha, log_delta, feh, co, log_p_quench = cube[:11]

            if len(self.cloud_species) > 0:
                # optional cloud parameters
                fe_fraction, mgsio3_fraction, fsed, kzz, sigma_lnorm = cube[11:16]

            # create dictionary with flux scaling parameters

            count = 0
            scaling = {}

            for i, item in enumerate(self.spectrum.keys()):
                if item in bounds and bounds[item][0] is not None:
                    scaling[item] = cube[self.n_param+count]
                    count += 1

                else:
                    scaling[item] = 1.

            # create dictionary with error offset parameters

            count = 0
            err_offset = {}

            for i, item in enumerate(self.spectrum.keys()):
                if item in bounds and bounds[item][1] is not None:
                    err_offset[item] = cube[self.n_param+self.count_scale+count]
                    count += 1

                else:
                    err_offset[item] = 0.

            # initiate the logarithm of the likelihood
            log_likelihood = 0.

            # create a p-t profile

            try:
                temp, pphot, t_connect = retrieval_util.pt_ret_model(np.array([t1, t2, t3]),
                    1e1**log_delta, alpha, tint, self.pressure, feh, co)

            except:
                return -np.inf

            # return zero probability if the minimum temperature is negative

            if np.min(temp) < 0.:
                return -np.inf

            # calculate the emission spectrum

            try:
                if len(self.cloud_species) > 0:
                    # cloudy atmosphere

                    # mass fraction of Fe
                    XFe = retrieval_util.return_XFe(feh, co)

                    # logarithm of the cloud base mass fraction of Fe
                    log_X_cloud_base_Fe = np.log10(1e1**fe_fraction*XFe)

                    # mass fraction of MgSiO3
                    XMgSiO3 = retrieval_util.return_XMgSiO3(feh, co)

                    # logarithm of the cloud base mass fraction of MgSiO3
                    log_X_cloud_base_MgSiO3 = np.log10(1e1**mgsio3_fraction*XMgSiO3)

                else:
                    # clear atmosphere

                    log_X_cloud_base_Fe = -1e10
                    log_X_cloud_base_MgSiO3 = -1e10
                    fsed = 1e10
                    kzz = -1e10
                    sigma_lnorm = 10.

                # wlen_micron, flux_lambda, Pphot_esti, tau_pow, tau_cloud = \
                wlen_micron, flux_lambda = retrieval_util.calc_spectrum_clouds(self.rt_object,
                    self.pressure, temp, co, feh, log_p_quench, log_X_cloud_base_Fe,
                    log_X_cloud_base_MgSiO3, fsed, fsed, kzz, logg, sigma_lnorm, half=True,
                    plotting=plotting)

            except:
                return -np.inf

            # return zero probability if the spectrum contains NaN values

            if np.sum(np.isnan(flux_lambda)) > 0:
                if len(flux_lambda) > 1:
                    warnings.warn('Spectrum with NaN values encountered.')

                return -np.inf

            # scale the emitted spectrum to the observation
            flux_lambda = flux_lambda * (radius*nc.r_jup_mean/(self.distance*nc.pc))**2.

            for key, value in self.spectrum.items():
                data_wavel = value[0][:, 0]
                data_flux = value[0][:, 1]
                data_error = value[0][:, 2]
                data_cov_inv = value[2]
                spec_res = value[3]
                data_wavel_bins = value[4]

                # convolve with Gaussian LSF
                flux_smooth = retrieval_util.convolve(wlen_micron, flux_lambda, spec_res)

                # resample to observation
                flux_rebinned = rgw.rebin_give_width(wlen_micron, flux_smooth, data_wavel, data_wavel_bins)

                if plotting:
                    plt.errorbar(data_wavel, scaling[key]*data_flux, yerr=data_error+10.**err_offset[key],
                                 marker='o', ms=3, color='tab:blue', markerfacecolor='tab:blue')

                    plt.plot(data_wavel, flux_rebinned, marker='o', ms=3, color='tab:orange')

                # Calculate log-likelihood
                diff = flux_rebinned - scaling[key]*data_flux

                if data_cov_inv is not None:
                    log_likelihood += -np.dot(diff, data_cov_inv.dot(diff))/2.
                else:
                    log_likelihood += -np.sum((diff**2/(data_error**2.+(10.**err_offset[key])**2)))/2.

            if plotting:
                plt.plot(wlen_micron, flux_smooth, color='black', zorder=-20)
                plt.xlabel('Wavelength [$\mu$m]')
                plt.ylabel('Flux [W m$^{-2}$ $\mu$m$^{-1}$]')
                plt.savefig('spectrum.pdf', bbox_inches='tight')
                plt.clf()

            return log_likelihood

        # store the model parameters in a JSON file

        json.dump(self.parameters, open(f'{self.output_name}_params.json', 'w'))

        # run the nested sampling with MultiNest

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