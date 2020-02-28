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
                 scattering,
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
        self.scattering = scattering

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

        # spectrum
        self.parameters.append('logg')
        self.parameters.append('radius')

        # p-t profile
        self.parameters.append('tint')
        self.parameters.append('t1')
        self.parameters.append('t2')
        self.parameters.append('t3')
        self.parameters.append('alpha')
        self.parameters.append('log_delta')

        # abundances
        self.parameters.append('feh')
        self.parameters.append('co')
        self.parameters.append('log_p_quench')

        # clouds
        if len(cloud_species) > 0:
            self.parameters.append('fe_fraction')
            self.parameters.append('mgsio3_fraction')
            self.parameters.append('fsed')
            self.parameters.append('kzz')
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
                                  test_ck_shuffle_comp=self.scattering,
                                  do_scat_emis=self.scattering)

        # Create the RT arrays of appropriate lengths

        self.rt_object.setup_opa_structure(self.pressure[::3])

        # if half:
        #     rt_object.setup_opa_structure(self.pressure[::3])
        # else:
        #     rt_object.setup_opa_structure(self.pressure)

    def run_multinest(self,
                      bounds,
                      live_points=2000,
                      efficiency=0.05,
                      resume=False,
                      plotting=False):

        import rebin_give_width as rgw
        from petitRADTRANS_ck_test_speed import nat_cst as nc

        if len(self.cloud_species) == 0:
            self.n_param = 11
        else:
            self.n_param = 16

        count_scale = 0
        count_error = 0

        for item in self.spectrum.keys():
            if item in bounds:
                if bounds[item][0] is not None:
                    self.parameters.append(f'scale_{item}')
                    count_scale += 1

                if bounds[item][1] is not None:
                    self.parameters.append(f'error_{item}')
                    count_error += 1

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
            if 'logg' in bounds:
                logg = bounds['logg'][0] + (bounds['logg'][1]-bounds['logg'][0])*cube[0]
            else:
                # log(g) between 2 and 5.5
                logg = 2. + 3.5*cube[0]

            if 'radius' in bounds:
                radius = bounds['radius'][0] + (bounds['radius'][1]-bounds['radius'][0])*cube[1]
            else:
                # radius between 0.8 and 2 Rjup
                radius = 0.8 + 1.2*cube[1]

            if 'tint' in bounds:
                tint = bounds['tint'][0] + (bounds['tint'][1]-bounds['tint'][0])*cube[2]
            else:
                # tint between 500 and 3000 K
                tint = 500.+2500.*cube[2]

            # analytically calculate the connection temperature
            t_connect_calc = (3./4.*tint**4.*(0.1+2./3.))**0.25

            # scale to temperatures below that for t3
            t3 = t_connect_calc*(1-cube[5])

            # scale to temperatures below that for t2
            t2 = t3*(1-cube[4])

            # scale to temperatures below that for t1
            t1 = t2*(1-cube[3])

            if 'alpha' in bounds:
                alpha = bounds['alpha'][0] + (bounds['alpha'][1]-bounds['alpha'][0])*cube[6]
            else:
                # alpha between 1 and 2
                alpha = 1. + cube[6]

            # Photospheric pressure between 1e-3 and 100 bar
            Pphot_bar = 1e1**(-3. + 5.*cube[7])

            # Use this to calculate delta, and from this log(delta)
            delta = (Pphot_bar*1e6)**(-alpha)
            log_delta = np.log10(delta)

            if 'feh' in bounds:
                feh = bounds['feh'][0] + (bounds['feh'][1]-bounds['feh'][0])*cube[8]
            else:
                # [Fe/H] between -1.5 and 1.5
                feh = -1.5 + 3.*cube[8]

            if 'co' in bounds:
                co = bounds['co'][0] + (bounds['co'][1]-bounds['co'][0])*cube[9]
            else:
                # C/O between 0.1 and 1.6
                co = 0.1 + 1.5*cube[9]

            # Quench pressure between 1e-6 and 1e3 bar
            log_p_quench = -6. + 9.*cube[10]

            if len(self.cloud_species) > 0:
                # Cloud base mass fractions equal to maximum values allowed from elemental abundances
                # times 0.05 to 1.
                log_X_cloud_base_Fe_fraction = np.log10(0.05)+(np.log10(1.)-np.log10(0.05))*cube[11]
                log_X_cloud_base_MgSiO3_fraction = np.log10(0.05)+(np.log10(1.)-np.log10(0.05))*cube[12]

                # fseds between 0 and 10
                if 'fsed' in bounds:
                    fsed = bounds['fsed'][0] + (bounds['fsed'][1]-bounds['fsed'][0])*cube[13]
                else:
                    fsed = 10.*cube[13]

                # logg(kzz)s between 5 and 13
                if 'kzz' in bounds:
                    kzz = bounds['kzz'][0] + (bounds['kzz'][1]-bounds['kzz'][0])*cube[14]
                else:
                    kzz = 5. + 8.*cube[14]

                # Width of the log-normal particle size distribution allowed to vary between 1.05 and 3.
                if 'sigma_lnorm' in bounds:
                    sigma_lnorm = bounds['sigma_lnorm'][0] + (bounds['sigma_lnorm'][1]-bounds['sigma_lnorm'][0])*cube[15]
                else:
                    sigma_lnorm = 1.05 + 1.95*cube[15]

            # Put the new parameter values back into the cube

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
                cube[11] = log_X_cloud_base_Fe_fraction
                cube[12] = log_X_cloud_base_MgSiO3_fraction
                cube[13] = fsed
                cube[14] = kzz
                cube[15] = sigma_lnorm

            # flux scaling parameter

            count = 0

            for item in self.spectrum.keys():
                if item in bounds:
                    if bounds[item][0] is not None:
                        cube[self.n_param+count] = bounds[item][0][0] + \
                            (bounds[item][0][1]-bounds[item][0][0])*cube[self.n_param+count]

                        count += 1

            # error inflation parameter

            count = 0

            for item in self.spectrum.keys():
                if item in bounds:
                    if bounds[item][1] is not None:
                        cube[self.n_param+count_scale+count] = bounds[item][1][0] + \
                            (bounds[item][1][1]-bounds[item][1][0])*cube[self.n_param+count_scale+count]

                        count += 1

            # Width of the permitted alpha value
            # log_sigma_alpha = 1.-5.*cube[16]
            # cube[16] = log_sigma_alpha

        def loglike(cube, ndim, nparams):
            logg, radius, tint, t1, t2, t3, alpha, log_delta, feh, co, log_p_quench = cube[:11]

            if len(self.cloud_species) > 0:
                log_X_cloud_base_Fe_fraction, log_X_cloud_base_MgSiO3_fraction, fsed, kzz, sigma_lnorm = cube[11:16]

            scaling = {}
            for i, item in enumerate(self.spectrum.keys()):
                count = 0
                if item in bounds and bounds[item][0] is not None:
                    scaling[item] = cube[self.n_param+count]
                    count += 1
                else:
                    scaling[item] = 1.

            err_offset = {}
            for i, item in enumerate(self.spectrum.keys()):
                count = 0
                if item in bounds and bounds[item][1] is not None:
                    err_offset[item] = cube[self.n_param+count_scale+count]
                    count += 1
                else:
                    err_offset[item] = 0.

            # log_sigma_alpha = cube[16]

            # Prior check all input params
            log_prior = 0.

            # Calculate the log-likelihood
            log_likelihood = 0.

            try:
                temp, pphot, t_connect = retrieval_util.pt_ret_model(np.array([t1, t2, t3]),
                    1e1**log_delta, alpha, tint, self.pressure, feh, co, pm_path=self.pm_path)

            except:
                return -np.inf

            if np.min(temp) < 0.:
                return -np.inf

            try:
                if len(self.cloud_species) > 0:
                    XFe = retrieval_util.return_XFe(feh, co)
                    log_X_cloud_base_Fe = np.log10(1e1**log_X_cloud_base_Fe_fraction*XFe)

                    XMgSiO3 = retrieval_util.return_XMgSiO3(feh, co)
                    log_X_cloud_base_MgSiO3 = np.log10(1e1**log_X_cloud_base_MgSiO3_fraction*XMgSiO3)

                    # wlen_micron, flux_lambda, Pphot_esti, tau_pow, tau_cloud = \
                    wlen_micron, flux_lambda = retrieval_util.calc_spectrum_clouds(self.rt_object,
                        self.pressure, temp, co, feh, log_p_quench, log_X_cloud_base_Fe,
                        log_X_cloud_base_MgSiO3, fsed, fsed, kzz, logg, sigma_lnorm, half=True,
                        plotting=plotting, pm_path=self.pm_path, radtrans_path=self.radtrans_path)

                else:
                    # wlen_micron, flux_lambda, Pphot_esti, tau_pow, tau_cloud = \
                    wlen_micron, flux_lambda = retrieval_util.calc_spectrum_clouds(self.rt_object,
                    self.pressure, temp, co, feh, log_p_quench, 10., 10., 10., 10., 10., logg, 10.,
                    half=True, plotting=plotting, pm_path=self.pm_path,
                    radtrans_path=self.radtrans_path)

                # wlen_micron, flux_lambda = retrieval_util.calc_spectrum_clear(self.rt_object, self.pressure,
                #     temp, logg, co, feh, log_p_quench, pm_path=self.pm_path, radtrans_path=self.radtrans_path)

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

            for key, value in self.spectrum.items():
                # convolve with Gaussian LSF
                # flux_take = retrieval_util.convolve(wlen_micron, flux_lambda, data_resolution[key])

                data_wavel = value[0][:, 0]
                data_flux = value[0][:, 1]
                data_error = value[0][:, 2]
                data_cov_inv = value[2]
                data_wavel_bins = value[3]

                # Rebin to observation
                # flux_rebinned = rgw.rebin_give_width(wlen_micron, flux_take, data_wavel, data_wavel_bins)
                flux_rebinned = rgw.rebin_give_width(wlen_micron, flux_lambda, data_wavel, data_wavel_bins)

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
                plt.plot(wlen_micron, flux_lambda, color='black', zorder=-20)
                plt.xlabel('Wavelength [$\mu$m]')
                plt.ylabel('Flux [W m$^{-2}$ $\mu$m$^{-1}$]')
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
