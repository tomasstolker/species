"""
Module with a frontend for atmospheric retrieval with ``petitRADTRANS`` (see Mollière et al.
2019). Details on the retrieval code are available at https://petitradtrans.readthedocs.io.
The frontend contains major contributions by Paul Mollière (MPIA).
"""

import os
import json
import time
import warnings

from typing import List, Optional, Tuple, Union

import pymultinest
import numpy as np
import matplotlib.pyplot as plt

from typeguard import typechecked
from scipy.stats import invgamma
from rebin_give_width import rebin_give_width

from petitRADTRANS.radtrans import Radtrans
from poor_mans_nonequ_chem_FeH.poor_mans_nonequ_chem.poor_mans_nonequ_chem import \
    interpol_abundances

from species.analysis import photometry
from species.data import database
from species.core import constants
from species.read import read_filter, read_object
from species.util import retrieval_util, dust_util


os.environ['OMP_NUM_THREADS'] = '1'


class AtmosphericRetrieval:
    """
    Class for atmospheric retrieval with ``petitRADTRANS``.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 line_species: Optional[list],
                 cloud_species: Optional[list],
                 output_folder: str,
                 wavel_range: Optional[Tuple[float, float]],
                 scattering: bool = True,
                 inc_spec: Union[bool, List[str]] = True,
                 inc_phot: Union[bool, List[str]] = False) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        line_species : list, None
            List with the line species. No line species are used if set to ``None``.
        cloud_species : list, None
            List with the cloud species. No cloud species are used if set to ``None``.
        output_folder : str
            Folder name that is used for the output files from ``MultiNest``. The folder should
            already exist.
        wavel_range : tuple(float, float), None
            The wavelength range (um) of the forward model. Should be a bit broader than the
            minimum and maximum wavelength of the data. The wavelength range is set automatically
            if the argument is set to ``None``.
        scattering : bool
            Include scattering in the radiative transfer. Scattering is not required if no cloud
            species are selected.
        inc_spec : bool, list(str)
            Include spectroscopic data in the fit. If a boolean, either all (``True``) or none
            (``False``) of the data are selected. If a list, a subset of spectrum names (as stored
            in the database with :func:`~species.data.database.Database.add_object`) can be
            provided.
        inc_phot : bool, list(str)
            Include photometric data in the fit. If a boolean, either all (``True``) or none
            (``False``) of the data are selected. If a list, a subset of filter names (as stored in
            the database) can be provided.

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
        self.output_folder = output_folder

        # get object data

        self.object = read_object.ReadObject(self.object_name)
        self.distance = self.object.get_distance()[0]  # [pc]

        print(f'Object: {self.object_name}')
        print(f'Distance: {self.distance}')

        if self.line_species is None:
            print('Line species: None')
            self.line_species = []

        else:
            print('Line species:')
            for item in self.line_species:
                print(f'   - {item}')

        if self.cloud_species is None:
            print('Cloud species: None')
            self.cloud_species = []

        else:
            print('Cloud species:')
            for item in self.cloud_species:
                print(f'   - {item}')

        print(f'Scattering: {self.scattering}')

        species_db = database.Database()

        objectbox = species_db.get_object(object_name,
                                          inc_phot=True,
                                          inc_spec=True)

        # scattering is not required without cloud species

        if self.scattering and len(self.cloud_species) == 0:
            raise ValueError('Scattering is not required if there are no cloud species selected.')

        # get photometric data

        self.objphot = []
        self.synphot = []

        if isinstance(inc_phot, bool):
            if inc_phot:
                # Select all filters if True
                species_db = database.Database()
                inc_phot = objectbox.filters

            else:
                inc_phot = []

        if len(objectbox.filters) != 0:
            print('Photometric data:')

        for item in inc_phot:
            obj_phot = self.object.get_photometry(item)
            self.objphot.append(np.array([obj_phot[2], obj_phot[3]]))

            print(f'   - {item} (W m-2 um-1) = {obj_phot[2]:.2e} +/- {obj_phot[3]:.2e}')

            sphot = photometry.SyntheticPhotometry(item)
            self.synphot.append(sphot)

        # get spectroscopic data

        if isinstance(inc_spec, bool):
            if inc_spec:
                # Select all filters if True
                species_db = database.Database()
                objectbox = species_db.get_object(object_name)
                inc_spec = list(objectbox.spectrum.keys())

            else:
                inc_spec = []

        if inc_spec:
            # Select all spectra
            self.spectrum = self.object.get_spectrum()

            # Select the spectrum names that are not in inc_spec
            spec_remove = []
            for item in self.spectrum:
                if item not in inc_spec:
                    spec_remove.append(item)

            # Remove the spectra that are not included in inc_spec
            for item in spec_remove:
                del self.spectrum[item]

        if not inc_spec or self.spectrum is None:
            raise ValueError('At least one spectrum is required for AtmosphericRetrieval. Please '
                             'add a spectrum with the add_object method of Database. ')

        # set wavelength bins and add to spectrum dictionary

        self.wavel_min = []
        self.wavel_max = []

        print('Spectroscopic data:')

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

            print(f'   - {key}')
            print(f'     Wavelength range (um) = {wavel_data[0]:.2f} - {wavel_data[-1]:.2f}')
            print(f'     Spectral resolution = {self.spectrum[key][3]:.2f}')

        # set the wavelength range for the Radtrans object

        if wavel_range is None:
            self.wavel_range = (0.95*min(self.wavel_min), 1.15*max(self.wavel_max))

        else:
            self.wavel_range = (wavel_range[0], wavel_range[1])

        # create the pressure layers for the Radtrans object

        if len(self.cloud_species) > 0:
            # initiate many pressure layers for the refinement around the cloud decks
            n_pressure = 180
            # n_pressure = 1440

        else:
            # initiate fewer pressure layers for a cloudless atmosphere
            n_pressure = 180

        self.pressure = np.logspace(-6, 3, n_pressure)

        print(f'Initiating {self.pressure.size} pressure levels (bar): '
              f'{self.pressure[0]:.2e} - {self.pressure[-1]:.2e}')

        # initiate parameter list and counters

        self.parameters = []

    @typechecked
    def set_parameters(self,
                       bounds: dict,
                       chemistry: str,
                       quenching: bool,
                       pt_profile: str,
                       fit_corr: List[str]) -> None:
        """
        Function to set the list with parameters.

        Parameters
        ----------
        bounds : dict
            Dictionary with the parameter boundaries.
        chemistry : str
            The chemistry type: 'equilibrium' for equilibrium chemistry or 'free' for retrieval
            of free abundances (but constant with altitude).
        quenching : bool
            Fitting a quenching pressure.
        pt_profile : str
            The parametrization for the pressure-temperature profile ('molliere', 'free', or
            'monotonic').
        fit_corr : list(str), None
            List with spectrum names for which the correlation length and fractional amplitude are
            fitted (see Wang et al. 2020).

        Returns
        -------
        NoneType
            None
        """

        # check if clouds are used in combination with equilibrium chemistry

        if len(self.cloud_species) > 0 and chemistry != 'equilibrium':
            raise ValueError('Clouds are currently only implemented in combination with '
                             'equilibrium chemistry.')

        # check if the Mollière P/T profile is used in combination with equilibrium chemistry

        if pt_profile == 'molliere' and chemistry != 'equilibrium':
            raise ValueError('The \'molliere\' P/T parametrization can only be used in '
                             'combination with equilibrium chemistry.')

        # generic parameters

        self.parameters.append('logg')
        self.parameters.append('radius')

        # p-t profile parameters

        if pt_profile == 'molliere':
            self.parameters.append('tint')
            self.parameters.append('t1')
            self.parameters.append('t2')
            self.parameters.append('t3')
            self.parameters.append('alpha')
            self.parameters.append('log_delta')

        elif pt_profile in ['free', 'monotonic']:
            for i in range(15):
                self.parameters.append(f't{i}')

            if pt_profile == 'free':
                self.parameters.append('gamma_r')
                self.parameters.append('beta_r')

        # abundance parameters

        if chemistry == 'equilibrium':

            self.parameters.append('metallicity')
            self.parameters.append('c_o_ratio')

        elif chemistry == 'free':

            for item in self.line_species:
                self.parameters.append(item)

        if quenching:
            self.parameters.append('log_p_quench')

        # cloud parameters

        if len(self.cloud_species) > 0:
            if 'Fe(c)_cd' in self.cloud_species:
                if 'fe_tau' in bounds:
                    self.parameters.append('fe_tau')
                else:
                    self.parameters.append('fe_fraction')

            if 'MgSiO3(c)_cd' in self.cloud_species:
                if 'mgsio3_tau' in bounds:
                    self.parameters.append('mgsio3_tau')
                else:
                    self.parameters.append('mgsio3_fraction')

            if 'Al2O3(c)_cd' in self.cloud_species:
                if 'al2o3_tau' in bounds:
                    self.parameters.append('al2o3_tau')
                else:
                    self.parameters.append('al2o3_fraction')

            if 'Na2S(c)_cd' in self.cloud_species:
                if 'na2s_tau' in bounds:
                    self.parameters.append('na2s_tau')
                else:
                    self.parameters.append('na2s_fraction')

            if 'KCL(c)_cd' in self.cloud_species:
                if 'kcl_tau' in bounds:
                    self.parameters.append('kcl_tau')
                else:
                    self.parameters.append('kcl_fraction')

            self.parameters.append('fsed')
            self.parameters.append('kzz')
            self.parameters.append('sigma_lnorm')

        # add the flux scaling parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][0] is not None:
                    self.parameters.append(f'scaling_{item}')

        # add the error offset parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][1] is not None:
                    self.parameters.append(f'error_{item}')

        # add the wavelength calibration parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][2] is not None:
                    self.parameters.append(f'wavelength_{item}')

        # add extinction parameters

        if 'ism_ext' in bounds:
            self.parameters.append('ism_ext')

        if 'ism_red' in bounds:
            if 'ism_ext' not in bounds:
                raise ValueError('The \'ism_red\' parameter can only be used in combination '
                                 'with \'ism_ext\'.')

            self.parameters.append('ism_red')

        # add covariance parameters

        for item in self.spectrum:
            if item in fit_corr:
                self.parameters.append(f'corr_len_{item}')
                self.parameters.append(f'corr_amp_{item}')

        # list all parameters

        print(f'Fitting {len(self.parameters)} parameters:')

        for item in self.parameters:
            print(f'   - {item}')

    @typechecked
    def run_multinest(self,
                      bounds: dict,
                      chemistry: str = 'equilibrium',
                      quenching: bool = True,
                      pt_profile: str = 'molliere',
                      fit_corr: Optional[List[str]] = None,
                      live_points: int = 2000,
                      resume: bool = False,
                      plotting: bool = False,
                      check_isothermal: bool = False) -> None:
        """
        Function to run the ``PyMultiNest`` wrapper of the ``MultiNest`` sampler. While
        ``PyMultiNest`` can be installed with ``pip`` from the PyPI repository, ``MultiNest``
        has to to be build manually. See the ``PyMultiNest`` documentation for details:
        http://johannesbuchner.github.io/PyMultiNest/install.html. Note that the library path
        of ``MultiNest`` should be set to the environmental variable ``LD_LIBRARY_PATH`` on a
        Linux machine and ``DYLD_LIBRARY_PATH`` on a Mac. Alternatively, the variable can be
        set before importing the ``species`` package, for example:

        .. code-block:: python

            >>> import os
            >>> os.environ['DYLD_LIBRARY_PATH'] = '/path/to/MultiNest/lib'
            >>> import species

        Parameters
        ----------
        bounds : dict
            Dictionary with the prior boundaries.
        chemistry : str
            The chemistry type: 'equilibrium' for equilibrium chemistry or 'free' for retrieval
            of free abundances (but constant with altitude).
        quenching : bool
            Fitting a quenching pressure.
        pt_profile : str
            The parametrization for the pressure-temperature profile ('molliere', 'free', or
            'monotonic').
        fit_corr : list(str), None
            List with spectrum names for which the correlation length and fractional amplitude are
            fitted (see Wang et al. 2020).
        live_points : int
            Number of live points.
        resume : bool
            Resume from a previous run.
        plotting : bool
            Plot sample results for testing.
        check_isothermal : bool
            Check if there is an isothermal region below 1 bar. If so, discard the sample.

        Returns
        -------
        NoneType
            None
        """

        # Create the output folder if required

        if not os.path.exists(self.output_folder):
            raise ValueError(f'The output folder (\'{self.output_folder}\') does not exist.')

        # List with spectra for which the correlated noise is fitted

        if fit_corr is None:
            fit_corr = []

        for item in self.spectrum:
            if item in fit_corr:
                bounds[f'corr_len_{item}'] = (-3., 0.)  # log10(corr_len) (um)
                bounds[f'corr_amp_{item}'] = (0., 1.)

        # Create list with parameters for MultiNest

        if quenching and chemistry != 'equilibrium':
            raise ValueError('The \'quenching\' parameter can only be used in combination with '
                             'chemistry=\'equilibrium\'.')

        self.set_parameters(bounds, chemistry, quenching, pt_profile, fit_corr)

        # Create a dictionary with the cube indices of the parameters

        cube_index = {}
        for i, item in enumerate(self.parameters):
            cube_index[item] = i

        # Delete the cloud parameters from the boundaries dictionary in case of no cloud species

        if len(self.cloud_species) == 0:
            for item in ['fe', 'mgsio3', 'al2o3', 'na2s', 'kcl']:
                if f'{item}_fraction' in bounds:
                    del bounds[f'{item}_fraction']

                if f'{item}_tau' in bounds:
                    del bounds[f'{item}_tau']

            if 'fsed' in bounds:
                del bounds['fsed']

            if 'kzz' in bounds:
                del bounds['kzz']

            if 'sigma_lnorm' in bounds:
                del bounds['sigma_lnorm']

        # Delete C/H and O/H boundaries if the chemistry is not free

        if chemistry != 'free':
            if 'c_h_ratio' in bounds:
                del bounds['c_h_ratio']

            if 'o_h_ratio' in bounds:
                del bounds['o_h_ratio']

        # Create Ratrans object

        print('Setting up petitRADTRANS...')

        # the names in self.cloud_species are converted
        rt_object = Radtrans(line_species=self.line_species,
                             rayleigh_species=['H2', 'He'],
                             cloud_species=self.cloud_species,
                             continuum_opacities=['H2-H2', 'H2-He'],
                             wlen_bords_micron=self.wavel_range,
                             mode='c-k',
                             test_ck_shuffle_comp=self.scattering,
                             do_scat_emis=self.scattering)

        # create RT arrays of 60 pressure layers

        if len(self.cloud_species) > 0:
            # rt_object.setup_opa_structure(self.pressure[::24])
            rt_object.setup_opa_structure(self.pressure[::3])

        else:
            rt_object.setup_opa_structure(self.pressure[::3])

            print(f'Decreasing the number of pressure levels: {self.pressure.size} -> '
                  f'{self.pressure[::3].size}.')

        if pt_profile in ['free', 'monotonic']:
            knot_press = np.logspace(np.log10(self.pressure[0]), np.log10(self.pressure[-1]), 15)
        else:
            knot_press = None

        @typechecked
        def prior(cube,
                  n_dim: int,
                  n_param: int) -> None:
            """
            Function to transform the unit cube into the parameter cube.

            Parameters
            ----------
            cube : LP_c_double
                Unit cube.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            NoneType
                None
            """

            # surface gravity (dex)
            if 'logg' in bounds:
                logg = bounds['logg'][0] + (bounds['logg'][1]-bounds['logg'][0])*cube[cube_index['logg']]
            else:
                # default: 2 - 5.5 dex
                logg = 2. + 3.5*cube[cube_index['logg']]

            cube[cube_index['logg']] = logg

            # planet radius (Rjup)
            if 'radius' in bounds:
                radius = bounds['radius'][0] + (bounds['radius'][1]-bounds['radius'][0])*cube[cube_index['radius']]
            else:
                # defaul: 0.8-2 Rjup
                radius = 0.8 + 1.2*cube[cube_index['radius']]

            cube[cube_index['radius']] = radius

            if pt_profile == 'molliere':

                # Internal temperature (K) of the Eddington approximation (middle altitudes)
                # see Eq. 2 in Mollière et al. (2020)
                if 'tint' in bounds:
                    tint = bounds['tint'][0] + (bounds['tint'][1]-bounds['tint'][0])*cube[cube_index['tint']]
                else:
                    # Default: 500 - 3000 K
                    tint = 500. + 2500.*cube[cube_index['tint']]

                cube[cube_index['tint']] = tint

                # Connection temperature (K)
                t_connect = (3./4.*tint**4.*(0.1+2./3.))**0.25

                # The temperature (K) at temp_3 is scaled down from t_connect
                temp_3 = t_connect*(1-cube[cube_index['t3']])
                cube[cube_index['t3']] = temp_3

                # The temperature (K) at temp_2 is scaled down from temp_3
                temp_2 = temp_3*(1-cube[cube_index['t2']])
                cube[cube_index['t2']] = temp_2

                # The temperature (K) at temp_1 is scaled down from temp_2
                temp_1 = temp_2*(1-cube[cube_index['t1']])
                cube[cube_index['t1']] = temp_1

                # alpha: power law index in tau = delta * press_cgs**alpha
                # see Eq. 1 in Mollière et al. (2020)
                if 'alpha' in bounds:
                    alpha = bounds['alpha'][0] + (bounds['alpha'][1]-bounds['alpha'][0])*cube[cube_index['alpha']]
                else:
                    # Default: 1 - 2
                    alpha = 1. + cube[cube_index['alpha']]

                cube[cube_index['alpha']] = alpha

                # Photospheric pressure (bar)
                # Default: 1e-3 - 1e2 bar
                p_phot = 10.**(-3. + 5.*cube[cube_index['log_delta']])

                # delta: proportionality factor in tau = delta * press_cgs**alpha
                # see Eq. 1 in Mollière et al. (2020)
                delta = (p_phot*1e6)**(-alpha)
                log_delta = np.log10(delta)

                cube[cube_index['log_delta']] = log_delta

            elif pt_profile == 'free':
                # 15 temperature (K) knots
                for i in range(15):
                    # default: 0 - 8000 K
                    cube[cube_index[f't{i}']] = 8000.*cube[cube_index[f't{i}']]

                # penalization of wiggles in the P-T profile
                # inverse Gamma: a=1, b=5e-5
                beta_r = cube[cube_index['beta_r']]
                gamma_r = invgamma.ppf(cube[cube_index['gamma_r']], a=1., scale=beta_r)
                cube[cube_index['beta_r']] = beta_r
                cube[cube_index['gamma_r']] = gamma_r

            elif pt_profile == 'monotonic':
                # 15 temperature (K) knots
                cube[cube_index['t14']] = 10000.*cube[cube_index['t14']]

                for i in range(13, -1, -1):
                    cube[cube_index[f't{i}']] = cube[cube_index[f't{i+1}']] * \
                        (1.-cube[cube_index[f't{i}']])

                    # Increasing temperature steps with increasing pressure
                    # if i == 13:
                    #     cube[cube_index[f't{i}']] = cube[cube_index[f't{i+1}']] * \
                    #         (1.-cube[cube_index[f't{i}']])
                    #
                    # else:
                    #     temp_diff = cube[cube_index[f't{i+2}']] - cube[cube_index[f't{i+1}']]
                    #
                    #     if cube[cube_index[f't{i+1}']] - temp_diff < 0.:
                    #         temp_diff = cube[cube_index[f't{i+1}']]
                    #
                    #     cube[cube_index[f't{i}']] = cube[cube_index[f't{i+1}']] - \
                    #         cube[cube_index[f't{i}']]*temp_diff

            if chemistry == 'equilibrium':
                # metallicity (dex) for the nabla_ad interpolation
                if 'metallicity' in bounds:
                    metallicity = bounds['metallicity'][0] + (bounds['metallicity'][1]-bounds['metallicity'][0])*cube[cube_index['metallicity']]
                else:
                    # default: -1.5 - 1.5 dex
                    metallicity = -1.5 + 3.*cube[cube_index['metallicity']]

                cube[cube_index['metallicity']] = metallicity

                # carbon-to-oxygen ratio for the nabla_ad interpolation
                if 'c_o_ratio' in bounds:
                    c_o_ratio = bounds['c_o_ratio'][0] + (bounds['c_o_ratio'][1]-bounds['c_o_ratio'][0])*cube[cube_index['c_o_ratio']]
                else:
                    # default: 0.1 - 1.6
                    c_o_ratio = 0.1 + 1.5*cube[cube_index['c_o_ratio']]

                cube[cube_index['c_o_ratio']] = c_o_ratio

            elif chemistry == 'free':
                # log10 abundances of the line species
                log_x_abund = {}

                for item in self.line_species:
                    if item in bounds:
                        cube[cube_index[item]] = bounds[item][0] + (bounds[item][1]-bounds[item][0])*cube[cube_index[item]]

                    elif item not in ['K', 'K_lor_cut', 'K_burrows']:
                        # default: -10. - 0. dex
                        cube[cube_index[item]] = -10.*cube[cube_index[item]]

                        # add the log10 of the mass fraction to the abundace dictionary
                        log_x_abund[item] = cube[cube_index[item]]

                if 'Na' in self.line_species or 'Na_lor_cut' in self.line_species or \
                        'Na_burrows' in self.line_species:
                    log_x_k_abund = retrieval_util.potassium_abundance(log_x_abund)

                if 'K' in self.line_species:
                    cube[cube_index['K']] = log_x_k_abund

                elif 'K_lor_cut' in self.line_species:
                    cube[cube_index['K_lor_cut']] = log_x_k_abund

                elif 'K_burrows' in self.line_species:
                    cube[cube_index['K_burrows']] = log_x_k_abund

            # CO/CH4 quenching pressure (bar)
            if quenching:
                if 'log_p_quench' in bounds:
                    log_p_quench = bounds['log_p_quench'][0] + (bounds['log_p_quench'][1]-bounds['log_p_quench'][0])*cube[cube_index['log_p_quench']]
                else:
                    # Default: -6 - 3. (i.e. 1e-6 - 1e3 bar)
                    log_p_quench = -6. + 9.*cube[cube_index['log_p_quench']]

                cube[cube_index['log_p_quench']] = log_p_quench

            if len(self.cloud_species) > 0:
                # Sedimentation parameter: ratio of the settling and mixing velocities of the
                # cloud particles (used in Eq. 3 of Mollière et al. 2020)
                if 'fsed' in bounds:
                    fsed = bounds['fsed'][0] + (bounds['fsed'][1]-bounds['fsed'][0])*cube[cube_index['fsed']]
                else:
                    # Default: 0 - 10
                    fsed = 10.*cube[cube_index['fsed']]

                cube[cube_index['fsed']] = fsed

                # Log10 of the eddy diffusion coefficient (cm2 s-1)
                if 'kzz' in bounds:
                    kzz = bounds['kzz'][0] + (bounds['kzz'][1]-bounds['kzz'][0])*cube[cube_index['kzz']]
                else:
                    # Default: 5 - 13
                    kzz = 5. + 8.*cube[cube_index['kzz']]

                cube[cube_index['kzz']] = kzz

                # Geometric standard deviation of the log-normal size distribution
                if 'sigma_lnorm' in bounds:
                    sigma_lnorm = bounds['sigma_lnorm'][0] + (bounds['sigma_lnorm'][1] -
                                                              bounds['sigma_lnorm'][0])*cube[cube_index['sigma_lnorm']]
                else:
                    # Default: 1.05 - 3.
                    sigma_lnorm = 1.05 + 1.95*cube[cube_index['sigma_lnorm']]

                cube[cube_index['sigma_lnorm']] = sigma_lnorm

                # Cloud mass fractions at the cloud base, relative to the maximum values allowed
                # from elemental abundances (see Eq. 3 in Mollière et al. 2020)

                if 'Fe(c)' in self.cloud_species:

                    if 'fe_fraction' in bounds:
                        fe_fraction = bounds['fe_fraction'][0] + (bounds['fe_fraction'][1] -
                            bounds['fe_fraction'][0])*cube[cube_index['fe_fraction']]

                    else:
                        # default: 0.05 - 1.
                        fe_fraction = np.log10(0.05) + (np.log10(1.) -
                            np.log10(0.05))*cube[cube_index['fe_fraction']]

                    cube[cube_index['fe_fraction']] = fe_fraction

                if 'MgSiO3(c)' in self.cloud_species:

                    if 'mgsio3_fraction' in bounds:
                        mgsio3_fraction = bounds['mgsio3_fraction'][0] + (bounds['mgsio3_fraction'][1] -
                            bounds['mgsio3_fraction'][0])*cube[cube_index['mgsio3_fraction']]

                    else:
                        # default: 0.05 - 1.
                        mgsio3_fraction = np.log10(0.05) + (np.log10(1.) -
                            np.log10(0.05))*cube[cube_index['mgsio3_fraction']]

                    cube[cube_index['mgsio3_fraction']] = mgsio3_fraction

                if 'Al2O3(c)' in self.cloud_species:

                    if 'al2o3_fraction' in bounds:
                        al2o3_fraction = bounds['al2o3_fraction'][0] + (bounds['al2o3_fraction'][1] -
                            bounds['al2o3_fraction'][0])*cube[cube_index['al2o3_fraction']]

                        cube[cube_index['al2o3_fraction']] = al2o3_fraction

                    elif 'al2o3_tau' in bounds:
                        al2o3_tau = bounds['al2o3_tau'][0] + (bounds['al2o3_tau'][1] -
                            bounds['al2o3_tau'][0])*cube[cube_index['al2o3_tau']]

                        cube[cube_index['al2o3_tau']] = al2o3_tau

                    else:
                        # Default: 0.05 - 1.
                        al2o3_fraction = np.log10(0.05) + (np.log10(1.) -
                            np.log10(0.05))*cube[cube_index['al2o3_fraction']]

                        cube[cube_index['al2o3_fraction']] = al2o3_fraction

                if 'Na2S(c)' in self.cloud_species:

                    if 'na2s_fraction' in bounds:
                        na2s_fraction = bounds['na2s_fraction'][0] + (bounds['na2s_fraction'][1] -
                            bounds['na2s_fraction'][0])*cube[cube_index['na2s_fraction']]

                    else:
                        # default: 0.05 - 1.
                        na2s_fraction = np.log10(0.05) + (np.log10(1.) -
                            np.log10(0.05))*cube[cube_index['na2s_fraction']]

                    cube[cube_index['na2s_fraction']] = na2s_fraction

                if 'KCL(c)' in self.cloud_species:

                    if 'kcl_fraction' in bounds:
                        kcl_fraction = bounds['kcl_fraction'][0] + (bounds['kcl_fraction'][1] -
                            bounds['kcl_fraction'][0])*cube[cube_index['kcl_fraction']]

                    else:
                        # default: 0.05 - 1.
                        kcl_fraction = np.log10(0.05) + (np.log10(1.) -
                            np.log10(0.05))*cube[cube_index['kcl_fraction']]

                    cube[cube_index['kcl_fraction']] = kcl_fraction

            # add flux scaling parameter if the boundaries are provided

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][0] is not None:
                        cube[cube_index[f'scaling_{item}']] = bounds[item][0][0] + \
                            (bounds[item][0][1]-bounds[item][0][0])*cube[cube_index[f'scaling_{item}']]

            # add error inflation parameter if the boundaries are provided

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][1] is not None:
                        cube[cube_index[f'error_{item}']] = bounds[item][1][0] + \
                            (bounds[item][1][1]-bounds[item][1][0]) * \
                            cube[cube_index[f'error_{item}']]

            # add wavelength calibration parameter if the boundaries are provided

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][2] is not None:
                        cube[cube_index[f'wavelength_{item}']] = bounds[item][2][0] + \
                            (bounds[item][2][1]-bounds[item][2][0]) * \
                            cube[cube_index[f'wavelength_{item}']]

            # add covariance parameters if any spectra are provided to fit_corr

            for item in self.spectrum:
                if item in fit_corr:
                    cube[cube_index[f'corr_len_{item}']] = bounds[f'corr_len_{item}'][0] + \
                        (bounds[f'corr_len_{item}'][1]-bounds[f'corr_len_{item}'][0]) * \
                        cube[cube_index[f'corr_len_{item}']]

                    cube[cube_index[f'corr_amp_{item}']] = bounds[f'corr_amp_{item}'][0] + \
                        (bounds[f'corr_amp_{item}'][1]-bounds[f'corr_amp_{item}'][0]) * \
                        cube[cube_index[f'corr_amp_{item}']]

            # ISM extinction

            if 'ism_ext' in bounds:
                ism_ext = bounds['ism_ext'][0] + (bounds['ism_ext'][1]-bounds['ism_ext'][0])*cube[cube_index['ism_ext']]

                cube[cube_index['ism_ext']] = ism_ext

            if 'ism_red' in bounds:
                ism_red = bounds['ism_red'][0] + (bounds['ism_red'][1]-bounds['ism_red'][0])*cube[cube_index['ism_red']]

                cube[cube_index['ism_red']] = ism_red

        @typechecked
        def loglike(cube,
                    n_dim: int,
                    n_param: int) -> float:
            """
            Function for the logarithm of the likelihood, computed from the parameter cube.

            Parameters
            ----------
            cube : LP_c_double
                Unit cube.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            float
                Sum of the logarithm of the prior and likelihood.
            """

            # initiate the logarithm of the prior and likelihood

            ln_prior = 0.
            ln_like = 0.

            # create dictionary with flux scaling parameters

            scaling = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][0] is not None:
                    scaling[item] = cube[cube_index[f'scaling_{item}']]
                else:
                    scaling[item] = 1.

            # create dictionary with error offset parameters

            err_offset = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][1] is not None:
                    err_offset[item] = cube[cube_index[f'error_{item}']]
                else:
                    err_offset[item] = None

            # create dictionary with wavelength calibration parameters

            wavel_cal = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][2] is not None:
                    wavel_cal[item] = cube[cube_index[f'wavelength_{item}']]
                else:
                    wavel_cal[item] = 0.

            # create dictionary with covariance parameters

            corr_len = {}
            corr_amp = {}

            for item in self.spectrum:
                if f'corr_len_{item}' in bounds:
                    corr_len[item] = 10.**cube[cube_index[f'corr_len_{item}']]  # (um)

                if f'corr_amp_{item}' in bounds:
                    corr_amp[item] = cube[cube_index[f'corr_amp_{item}']]

            # Check if the cloud optical depth is a free parameter

            calc_tau_cloud = False

            for item in self.cloud_species:
                if item[:-3].lower()+'_tau' in bounds:
                    calc_tau_cloud = True

            # Prepare the scaling based on the cloud optical depth

            if calc_tau_cloud:
                # Create the P/T profile
                temperature, knot_temp = retrieval_util.create_pt_profile(
                    cube, cube_index, pt_profile, self.pressure, knot_press)

                if 'log_p_quench' in cube_index:
                    # Quenching pressure (bar)
                    quench_pressure = 10.**cube[cube_index['log_p_quench']]
                else:
                    quench_pressure = None

                # Interpolate the abundances, following chemical equilibrium
                abund_in = interpol_abundances(np.full(self.pressure.size, cube[cube_index['c_o_ratio']]),
                                               np.full(self.pressure.size, cube[cube_index['metallicity']]),
                                               temperature,
                                               self.pressure,
                                               Pquench_carbon=quench_pressure)

                # Extract the mean molecular weight
                mmw = abund_in['MMW']

                # Set the kappa_zero argument, required by Radtrans.mix_opa_tot
                rt_object.kappa_zero = None

            # Create the P-T profile

            temp, knot_temp = retrieval_util.create_pt_profile(
                cube, cube_index, pt_profile, self.pressure, knot_press)

            if check_isothermal:
                # Get knot indices where the pressure is larger than 1 bar
                indices = np.where(knot_press > 1.)[0]

                # Remove last index because temp_diff.size = knot_press.size - 1
                indices = indices[:-1]

                temp_diff = np.diff(knot_temp)
                temp_diff = temp_diff[indices]

                small_temp = np.where(temp_diff < 100.)[0]

                if len(small_temp) > 0:
                    # Return zero probability if there is a temperature step smaller than 10 K
                    return -np.inf

            if pt_profile == 'free':
                temp_sum = np.sum((knot_temp[2:] + knot_temp[:-2] - 2.*knot_temp[1:-1])**2.)
                # temp_sum = np.sum((temp[::3][2:] + temp[::3][:-2] - 2.*temp[::3][1:-1])**2.)

                ln_prior += -1.*temp_sum/(2.*cube[cube_index['gamma_r']]) - \
                    0.5*np.log(2.*np.pi*cube[cube_index['gamma_r']])

            # return zero probability if the minimum temperature is negative

            if np.min(temp) < 0.:
                return -np.inf

            # set the quenching pressure
            if quenching:
                log_p_quench = cube[cube_index['log_p_quench']]
            else:
                log_p_quench = -10.

            # calculate the emission spectrum

            start = time.time()

            if len(self.cloud_species) > 0:
                # cloudy atmosphere

                cloud_fractions = {}
                for item in self.cloud_species:
                    if f'{item[:-3].lower()}_fraction' in self.parameters:
                        cloud_fractions[item] = cube[cube_index[f'{item[:-3].lower()}_fraction']]

                    elif f'{item[:-3].lower()}_tau' in self.parameters:
                        cloud_fractions[item] = retrieval_util.scale_cloud_abund(
                            cube, cube_index, rt_object, self.pressure, temp, mmw, chemistry,
                            abund_in, item, cube[cube_index[f'{item[:-3].lower()}_tau']])

                log_x_base = retrieval_util.log_x_cloud_base(cube[cube_index['c_o_ratio']],
                                                             cube[cube_index['metallicity']],
                                                             cloud_fractions)

                # the try-except is required to catch numerical precision errors with the clouds
                # try:
                wlen_micron, flux_lambda, _ = retrieval_util.calc_spectrum_clouds(
                    rt_object, self.pressure, temp, cube[cube_index['c_o_ratio']],
                    cube[cube_index['metallicity']], log_p_quench, log_x_base,
                    cube[cube_index['fsed']], cube[cube_index['kzz']], cube[cube_index['logg']],
                    cube[cube_index['sigma_lnorm']], chemistry=chemistry, half=True,
                    plotting=plotting, contribution=False)

                # except:
                #     return -np.inf

            else:
                # clear atmosphere

                if chemistry == 'equilibrium':
                    wlen_micron, flux_lambda, _ = retrieval_util.calc_spectrum_clear(
                        rt_object, self.pressure, temp, cube[cube_index['logg']],
                        cube[cube_index['c_o_ratio']], cube[cube_index['metallicity']],
                        log_p_quench, None, chemistry=chemistry, half=True, contribution=False)

                elif chemistry == 'free':
                    # create a dictionary with the mass fractions
                    log_x_abund = {}
                    for item in self.line_species:
                        log_x_abund[item] = cube[cube_index[item]]

                    # check if the sum of fractional abundances is smaller than unity

                    if np.sum(10.**np.asarray(list(log_x_abund.values()))) > 1.:
                        return -np.inf

                    # check if the C/H and O/H ratios are within the prior boundaries

                    if 'c_h_ratio' or 'o_h_ratio' in bounds:
                        c_h_ratio, o_h_ratio = retrieval_util.calc_metal_ratio(log_x_abund)

                    if 'c_h_ratio' in bounds and (c_h_ratio < bounds['c_h_ratio'][0] or
                                                  c_h_ratio > bounds['c_h_ratio'][1]):

                        return -np.inf

                    if 'o_h_ratio' in bounds and (o_h_ratio < bounds['o_h_ratio'][0] or
                                                  o_h_ratio > bounds['o_h_ratio'][1]):

                        return -np.inf

                    # calculate the emission spectrum

                    wlen_micron, flux_lambda, _ = retrieval_util.calc_spectrum_clear(
                        rt_object, self.pressure, temp, cube[cube_index['logg']],
                        None, None, None, log_x_abund, chemistry, half=True, contribution=False)

            end = time.time()

            print(f'\rRadiative transfer time: {end-start:.2e} s', end='', flush=True)

            # return zero probability if the spectrum contains NaN values

            if np.sum(np.isnan(flux_lambda)) > 0:
                # if len(flux_lambda) > 1:
                #     warnings.warn('Spectrum with NaN values encountered.')

                return -np.inf

            # scale the emitted spectrum to the observation
            flux_lambda *= (cube[cube_index['radius']]*constants.R_JUP / (self.distance*constants.PARSEC))**2.

            for i, item in enumerate(self.spectrum.keys()):
                # shift the wavelengths of the data with the fitted calibration parameter
                data_wavel = self.spectrum[item][0][:, 0] + wavel_cal[item]

                # flux density
                data_flux = self.spectrum[item][0][:, 1]

                # variance with optional inflation
                if err_offset[item] is None:
                    data_var = self.spectrum[item][0][:, 2]**2
                else:
                    data_var = (self.spectrum[item][0][:, 2] + 10.**err_offset[item])**2

                # apply ISM extinction to the model spectrum
                if 'ism_ext' in self.parameters:
                    if 'ism_red' in self.parameters:
                        ism_reddening = cube[cube_index['ism_red']]

                    else:
                        # Use default interstellar reddening (R_V = 3.1)
                        ism_reddening = 3.1

                    flux_lambda = dust_util.apply_ism_ext(wlen_micron,
                                                          flux_lambda,
                                                          cube[cube_index['ism_ext']],
                                                          ism_reddening)

                # convolve with Gaussian LSF
                flux_smooth = retrieval_util.convolve(wlen_micron,
                                                      flux_lambda,
                                                      self.spectrum[item][3])

                # resample to the observation
                flux_rebinned = rebin_give_width(wlen_micron,
                                                 flux_smooth,
                                                 data_wavel,
                                                 self.spectrum[item][4])

                # difference between the observed and modeled spectrum
                flux_diff = flux_rebinned - scaling[item]*data_flux

                if self.spectrum[item][2] is not None:
                    # Use the inverted covariance matrix

                    if err_offset[item] is None:
                        data_cov_inv = self.spectrum[item][2]

                    else:
                        # Ratio of the inflated and original uncertainties
                        sigma_ratio = np.sqrt(data_var) / self.spectrum[item][0][:, 2]
                        sigma_j, sigma_i = np.meshgrid(sigma_ratio, sigma_ratio)

                        # Calculate the inversion of the infalted covariances
                        data_cov_inv = np.linalg.inv(self.spectrum[item][1]*sigma_i*sigma_j)

                    # Use the inverted covariance matrix
                    dot_tmp = np.dot(flux_diff, np.dot(data_cov_inv, flux_diff))
                    ln_like += -0.5*dot_tmp - 0.5*np.nansum(np.log(2.*np.pi*data_var))

                else:
                    if item in fit_corr:
                        # Covariance model (Wang et al. 2020)
                        wavel_j, wavel_i = np.meshgrid(data_wavel, data_wavel)

                        error = np.sqrt(data_var)  # (W m-2 um-1)
                        error_j, error_i = np.meshgrid(error, error)

                        cov_matrix = corr_amp[item]**2 * error_i * error_j * \
                            np.exp(-(wavel_i-wavel_j)**2 / (2.*corr_len[item]**2)) + \
                            (1.-corr_amp[item]**2) * np.eye(data_wavel.shape[0])*error_i**2

                        dot_tmp = np.dot(flux_diff, np.dot(np.linalg.inv(cov_matrix), flux_diff))

                        ln_like += -0.5*dot_tmp - 0.5*np.nansum(np.log(2.*np.pi*data_var))

                    else:
                        # calculate the log-likelihood without the covariance matrix
                        ln_like += -0.5*np.sum(flux_diff**2/data_var + np.log(2.*np.pi*data_var))

                if plotting:
                    plt.errorbar(data_wavel, scaling[item]*data_flux, yerr=np.sqrt(data_var),
                                 marker='o', ms=3, color='tab:blue', markerfacecolor='tab:blue', alpha=0.2)

                    plt.plot(data_wavel, flux_rebinned, marker='o', ms=3, color='tab:orange', alpha=0.2)

            for i, obj_item in enumerate(self.objphot):
                # Calculate the photometric flux from the model spectrum
                phot_flux, _ = self.synphot[i].spectrum_to_flux(wlen_micron, flux_lambda)

                if plotting:
                    read_filt = read_filter.ReadFilter(self.synphot[i].filter_name)

                    plt.errorbar(read_filt.mean_wavelength(), phot_flux, xerr=read_filt.filter_fwhm(),
                                 marker='s', ms=5., color='tab:green', mfc='white')

                if obj_item.ndim == 1:
                    # Filter with one flux
                    ln_like += -0.5 * (obj_item[0] - phot_flux)**2 / obj_item[1]**2

                    if plotting:
                        plt.errorbar(read_filt.mean_wavelength(), obj_item[0], xerr=read_filt.filter_fwhm(),
                                     yerr=obj_item[1], marker='s', ms=5., color='tab:green', mfc='tab:green')

                else:
                    # Filter with multiple fluxes
                    for j in range(obj_item.shape[1]):
                        ln_like += -0.5 * (obj_item[0, j] - phot_flux)**2 / obj_item[1, j]**2

            if plotting:
                plt.plot(wlen_micron, flux_smooth, color='black', zorder=-20)
                plt.xlabel(r'Wavelength ($\mu$m)')
                plt.ylabel(r'Flux (W m$^{-2}$ $\mu$m$^{-1}$)')
                plt.savefig('spectrum.pdf', bbox_inches='tight')
                plt.clf()

            return ln_prior + ln_like

        # store the model parameters in a JSON file

        json_filename = os.path.join(self.output_folder, 'params.json')
        print(f'Storing the model parameters: {json_filename}')

        with open(json_filename, 'w') as json_file:
            json.dump(self.parameters, json_file)

        # store the Radtrans arguments in a JSON file

        radtrans_filename = os.path.join(self.output_folder, 'radtrans.json')
        print(f'Storing the Radtrans arguments: {radtrans_filename}')

        radtrans_dict = {}
        radtrans_dict['line_species'] = self.line_species
        radtrans_dict['cloud_species'] = self.cloud_species
        radtrans_dict['distance'] = self.distance
        radtrans_dict['scattering'] = self.scattering
        radtrans_dict['chemistry'] = chemistry
        radtrans_dict['quenching'] = quenching
        radtrans_dict['pt_profile'] = pt_profile

        with open(radtrans_filename, 'w', encoding='utf-8') as json_file:
            json.dump(radtrans_dict, json_file, ensure_ascii=False, indent=4)

        # run the nested sampling with MultiNest

        print('Sampling the posterior distribution with MultiNest...')

        pymultinest.run(loglike,
                        prior,
                        len(self.parameters),
                        outputfiles_basename=os.path.join(self.output_folder, ''),
                        resume=resume,
                        verbose=True,
                        const_efficiency_mode=True,
                        sampling_efficiency=0.05,
                        n_live_points=live_points,
                        evidence_tolerance=0.5)
