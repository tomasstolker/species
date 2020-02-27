"""
Module with functionalities for atmospheric retrieval with petitRADTRANS.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import json
import warnings

import pymultinest
import numpy as np
import matplotlib.pyplot as plt

from species.analysis import photometry
from species.data import database
from species.read import read_object
from species.util import retrieval_util


class AtmosphericRetrieval:
    """
    Text
    """

    def __init__(self,
                 object_name,
                 line_species,
                 cloud_species,
                 output_name,
                 pm_path,
                 radtrans_path,
                 rebin_path):
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.

        Returns
        -------
        NoneType
            None
        """

        sys.path.append(radtrans_path)
        sys.path.append(rebin_path)

        from petitRADTRANS_ck_test_speed import Radtrans
        from petitRADTRANS_ck_test_speed import nat_cst as nc

        self.output_name = output_name
        self.pm_path = pm_path
        self.radtrans_path = radtrans_path
        self.line_species = line_species
        self.cloud_species = cloud_species

        self.object = read_object.ReadObject(object_name)
        self.distance = self.object.get_distance()[0]*nc.pc  # [cm]

        species_db = database.Database()
        objectbox = species_db.get_object(object_name, None)
        filters = objectbox.filters

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

        self.spectrum = self.object.get_spectrum()

        if self.spectrum is None:
            raise ValueError('A spectrum is required for the atmospheric retrieval.')

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

            wavel_min.append(wavel_data[0])
            wavel_max.append(wavel_data[-1])

        self.parameters = []
        self.parameters.append('t1')
        self.parameters.append('t2')
        self.parameters.append('t3')
        self.parameters.append('log_delta')
        self.parameters.append('alpha')
        self.parameters.append('tint')
        self.parameters.append('CO')
        self.parameters.append('FeH')
        self.parameters.append('log_p_quench')
        self.parameters.append('log_X_cloud_base_Fe_fraction')
        self.parameters.append('log_X_cloud_base_MgSiO3_fraction')
        self.parameters.append('fsed')
        self.parameters.append('Kzz')
        self.parameters.append('logg')
        self.parameters.append('radius')
        self.parameters.append('sigma_lnorm')
        # self.parameters.append('log_sigma_alpha')

        # Create mock PT profile for Radtrans object

        temp_params = {}
        temp_params['log_delta'] = -6.
        temp_params['log_gamma'] = 1.
        temp_params['t_int'] = 750.
        temp_params['t_equ'] = 0.
        temp_params['log_p_trans'] = -3.
        temp_params['alpha'] = 0.

        self.pressure, _ = nc.make_press_temp(temp_params)

        # Create Ratrans object

        self.rt_object = Radtrans(line_species=self.line_species,
                                  rayleigh_species=['H2', 'He'],
                                  continuum_opacities=['H2-H2', 'H2-He'],
                                  cloud_species=self.cloud_species,
                                  mode='c-k',
                                  wlen_bords_micron=(0.99*min(wavel_min), 1.01*max(wavel_max)),
                                  test_ck_shuffle_comp=True,
                                  do_scat_emis=True)

        print(self.rt_object.__dict__)

        # Create the RT arrays of appropriate lengths

        self.rt_object.setup_opa_structure(self.pressure[::3])

        # if half:
        #     rt_object.setup_opa_structure(self.pressure[::3])
        # else:
        #     rt_object.setup_opa_structure(self.pressure)

    def run_mcmc(self,
                 bounds,
                 live_points=2000,
                 efficiency=0.05,
                 resume=False,
                 plotting=False):

        import rebin_give_width as rgw
        from petitRADTRANS_ck_test_speed import nat_cst as nc

        for item in self.spectrum.keys():
            if item in bounds:
                self.parameters.append(f'{item}')

        def prior(cube, ndim, nparams):
            # tint from 500 to 3000 K
            if 'tint' in bounds:
                tint = bounds['tint'][0] + (bounds['tint'][1]-bounds['tint'][0])*cube[5]
            else:
                tint = 500.+2500.*cube[5]

            # analytically calculate the connection temperature
            t_connect_calc = (3./4.*tint**4.*(0.1+2./3.))**0.25

            # Scale to temperatures below that for t3
            t3 = t_connect_calc*(1-cube[2])

            # Scale to temperatures below that for t2
            t2 = t3*(1-cube[1])

            # Scale to temperatures below that for t1
            t1 = t2*(1-cube[0])

            # [Fe/H] between -1.5 and 1.5
            if 'feh' in bounds:
                feh = bounds['feh'][0] + (bounds['feh'][1]-bounds['feh'][0])*cube[7]
            else:
                feh = -1.5 + 3.*cube[7]

            # C/O between 0.1 and 1.6
            if 'co' in bounds:
                co = bounds['co'][0] + (bounds['co'][1]-bounds['co'][0])*cube[6]
            else:
                co = 0.1 + 1.5*cube[6]

            # log(g) between 2 and 5.5
            if 'logg' in bounds:
                logg = bounds['logg'][0] + (bounds['logg'][1]-bounds['logg'][0])*cube[13]
            else:
                logg = 2. + 3.5*cube[13]

            # alpha between 1 and 2
            if 'alpha' in bounds:
                alpha = bounds['alpha'][0] + (bounds['alpha'][1]-bounds['alpha'][0])*cube[4]
            else:
                alpha = 1. + cube[4]

            # Photospheric pressure between 1e-3 and 100 bar
            Pphot_bar = 1e1**(-3. + 5.*cube[3])

            # Use this to calculate delta, and from this log(delta)
            delta = (Pphot_bar*1e6)**(-alpha)
            log_delta = np.log10(delta)

            # Quench pressure between 1e-6 and 1e3 bar
            log_p_quench = -6. + 9.*cube[8]

            # Cloud base mass fractions equal to maximum values allowed from elemental abundances
            # times 0.05 to 1.
            log_X_cloud_base_Fe_fraction = np.log10(0.05)+(np.log10(1.)-np.log10(0.05))*cube[9]
            log_X_cloud_base_MgSiO3_fraction = np.log10(0.05)+(np.log10(1.)-np.log10(0.05))*cube[10]

            # fseds between 0 and 10
            if 'fsed' in bounds:
                fsed = bounds['fsed'][0] + (bounds['fsed'][1]-bounds['fsed'][0])*cube[11]
            else:
                fsed = 10.*cube[11]

            # logg(Kzz)s between 5 and 13
            if 'kzz' in bounds:
                kzz = bounds['kzz'][0] + (bounds['kzz'][1]-bounds['kzz'][0])*cube[12]
            else:
                kzz = 5. + 8.*cube[12]

            # Planetary radius between 0.8 and 2 Rjup
            if 'radius' in bounds:
                radius = bounds['radius'][0] + (bounds['radius'][1]-bounds['radius'][0])*cube[14]
            else:
                radius = 0.8 + 1.2*cube[14]

            # Width of the log-normal particle size distribution allowed to vary between 1.05 and 3.
            if 'sigma_lnorm' in bounds:
                sigma_lnorm = bounds['sigma_lnorm'][0] + (bounds['sigma_lnorm'][1]-bounds['sigma_lnorm'][0])*cube[15]
            else:
                sigma_lnorm = 1.05 + 1.95*cube[15]

            # Put the new parameter values back into the cube
            cube[0] = t1
            cube[1] = t2
            cube[2] = t3
            cube[3] = log_delta
            cube[4] = alpha
            cube[5] = tint
            cube[6] = co
            cube[7] = feh
            cube[8] = log_p_quench
            cube[9] = log_X_cloud_base_Fe_fraction
            cube[10] = log_X_cloud_base_MgSiO3_fraction
            cube[11] = fsed
            cube[12] = kzz
            cube[13] = logg
            cube[14] = radius
            cube[15] = sigma_lnorm

            for i, item in enumerate(self.spectrum.keys()):
                if item in bounds:
                    cube[16+i] = bounds[f'{item}'][0] + (bounds[f'{item}'][1]-bounds[f'{item}'][0])*cube[16+i]

            # Width of the permitted alpha value
            # log_sigma_alpha = 1.-5.*cube[16]
            # cube[16] = log_sigma_alpha

            return

        def loglike(cube, ndim, nparams):
            t1, t2, t3, log_delta, alpha, tint = cube[:6]
            co = cube[6]
            feh = cube[7]
            log_p_quench = cube[8]
            log_X_cloud_base_Fe_fraction = cube[9]
            log_X_cloud_base_MgSiO3_fraction = cube[10]

            # Retrieve only one fsed
            fsed_Fe = cube[11]
            fsed_MgSiO3 = cube[11]

            kzz = cube[12]
            logg = cube[13]
            radius = cube[14]
            sigma_lnorm = cube[15]

            scaling = {}
            for i, item in enumerate(self.spectrum.keys()):
                if item in bounds:
                    scaling[item] = cube[16+i]
                else:
                    scaling[item] = 1.

            # log_sigma_alpha = cube[16]

            # Prior check all input params
            log_prior = 0.

            XFe = retrieval_util.return_XFe(feh, co)
            log_X_cloud_base_Fe = np.log10(1e1**log_X_cloud_base_Fe_fraction*XFe)

            XMgSiO3 = retrieval_util.return_XMgSiO3(feh, co)
            log_X_cloud_base_MgSiO3 = np.log10(1e1**log_X_cloud_base_MgSiO3_fraction*XMgSiO3)

            # Calculate the log-likelihood
            log_likelihood = 0.

            try:
                temp, pphot, t_connect = retrieval_util.pt_ret_model(
                    np.array([t1,t2,t3]), 1e1**log_delta, alpha, tint, self.pressure, feh, co, pm_path=self.pm_path)

            except:
                return -np.inf

            if np.min(temp) < 0.:
                return -np.inf

            try:
                wlen_micron, flux_lambda, Pphot_esti, tau_pow, tau_cloud = \
                    retrieval_util.calc_emission_spectrum(self.rt_object, self.pressure, temp, co, feh,
                    log_p_quench, log_X_cloud_base_Fe, log_X_cloud_base_MgSiO3, fsed_Fe,
                    fsed_MgSiO3, kzz, logg, sigma_lnorm, half=True, plotting=plotting,
                    pm_path=self.pm_path, radtrans_path=self.radtrans_path)

                # if (pphot/Pphot_esti) > 5.:
                #     return -np.inf

                # if np.abs(alpha-tau_pow) > 0.12:
                #     return -np.inf

                # if (pphot/Pphot_esti) > 5.:
                #     return -np.inf

                # sigma_alpha = 1e1**log_sigma_alpha
                # log_prior += -(alpha-tau_pow)**2./(sigma_alpha)**2/2. - 0.5*np.log(2.*np.pi*sigma_alpha**2.)

            except:
                return -np.inf

            # Return -inf if retrieval model returns NaN values
            if np.sum(np.isnan(flux_lambda)) > 0:
                if len(flux_lambda) > 1:
                    warnings.warn('Spectrum with NaN values encountered.')

                return -np.inf

            # Convert to observation
            flux_lambda = flux_lambda * (radius*nc.r_jup_mean/self.distance)**2.

            for instrument, value in self.spectrum.items():
                # convolve with Gaussian LSF
                # flux_take = retrieval_util.convolve(wlen_micron, flux_lambda, data_resolution[instrument])

                data_wavel = value[0][:, 0]
                data_flux = value[0][:, 1]
                data_error = value[0][:, 2]
                data_cov_inv = value[2]
                data_wavel_bins = value[3]

                # Rebin to observation
                # flux_rebinned = rgw.rebin_give_width(wlen_micron, flux_take, data_wavel, data_wavel_bins)
                flux_rebinned = rgw.rebin_give_width(wlen_micron, flux_lambda, data_wavel, data_wavel_bins)

                if plotting:
                    plt.errorbar(data_wavel, scaling[instrument]*data_flux/1e-16, data_error/1e-16, fmt='o', zorder=-20, color='red')
                    plt.plot(data_wavel, flux_rebinned/1e-16, 's', zorder=-20, color='blue')

                # Calculate log-likelihood
                diff = flux_rebinned - scaling[instrument]*data_flux

                if data_cov_inv is not None:
                    log_likelihood += -np.dot(diff, data_cov_inv.dot(diff))/2.
                else:
                    log_likelihood += -np.sum((diff/data_error)**2.)/2.

            if plotting:
                plt.plot(wlen_micron, flux_lambda/1e-16, color = 'black')
                plt.xscale('log')
                plt.savefig('spectrum.pdf', bbox_inches='tight')
                plt.clf()

            return log_prior + log_likelihood

        json.dump(self.parameters, open(f'{self.output_name}_params.json', 'w'))

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
