"""
Module with functionalities for the analysis of emission lines.
"""

import configparser
import os

from typing import Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ultranest

from astropy import units as u
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import StdDevUncertainty
from scipy.interpolate import interp1d
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from typeguard import typechecked

from species.core import constants
from species.data import database
from species.read import read_object
from species.util import read_util


class EmissionLine:
    """
    Class for the analysis of emission lines.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 spec_name: str,
                 wavel_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        spec_name : str
            Name of the spectrum that is stored at the object data of ``object_name``.
        wavel_range : tuple(float, float), None
            Wavelength range (um) that is cropped from the spectrum. The full spectrum is used if
            the argument is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        self.object_name = object_name
        self.spec_name = spec_name

        self.object = read_object.ReadObject(object_name)

        self.spectrum = self.object.get_spectrum()[spec_name][0]

        if wavel_range is None:
            self.wavel_range = (self.spectrum[0, 0], self.spectrum[-1, 0])

        else:
            self.wavel_range = wavel_range

            indices = np.where((self.spectrum[:, 0] >= wavel_range[0]) &
                               (self.spectrum[:, 0] <= wavel_range[1]))[0]

            self.spectrum = self.spectrum[indices, ]

        self.continuum_flux = np.full(self.spectrum.shape[0], 0.)
        self.continuum_check = False

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    @typechecked
    def subtract_continuum(self,
                           poly_degree: int = 3,
                           plot_filename: str = 'continuum.pdf') -> None:
        """
        Method for fitting the continuum with a polynomial function of the following form:
        :math:`P = \\sum_{i=0}^{i=n}C_{i} * x^{i}`. The spectrum is first smoothed with a median
        filter and then fitted with a linear least squares algorithm.

        Parameters
        ----------
        poly_degree : int
            Degree of the polynomial series.
        plot_filename : str
            Filename for the plots with the continuum fit and the continuum-subtracted spectrum.

        Returns
        -------
        NoneType
            None
        """

        # Fit continuum

        print('Fitting continuum...', end='', flush=True)

        spec_extract = Spectrum1D(flux=self.spectrum[:, 1]*u.W,
                                  spectral_axis=self.spectrum[:, 0]*u.um,
                                  uncertainty=StdDevUncertainty(self.spectrum[:, 2]*u.W))

        g1_fit = fit_generic_continuum(spec_extract,
                                       median_window=3,
                                       model=Polynomial1D(poly_degree),
                                       fitter=LinearLSQFitter())

        continuum_fit = g1_fit(spec_extract.spectral_axis)

        print(' [DONE]')

        # Subtract continuum

        spec_cont_sub = spec_extract - continuum_fit

        self.continuum_flux = continuum_fit/u.W

        # Create plot

        print(f'Plotting continuum fit: {plot_filename}...', end='', flush=True)

        mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
        mpl.rcParams['font.family'] = 'serif'

        plt.rc('axes', edgecolor='black', linewidth=2)
        plt.rcParams['axes.axisbelow'] = False

        plt.figure(1, figsize=(6, 6))
        gs = mpl.gridspec.GridSpec(2, 1)
        gs.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])

        ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True, bottom=True,
                        left=True, right=True, labelbottom=False)

        ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True, bottom=True,
                        left=True, right=True, labelbottom=False)

        ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True, bottom=True,
                        left=True, right=True)

        ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True, bottom=True,
                        left=True, right=True)

        ax2.set_xlabel('Wavelength (µm)', fontsize=16)
        ax1.set_ylabel('Flux (W m$^{-2}$ µm$^{-1}$)', fontsize=16)
        ax2.set_ylabel('Flux (W m$^{-2}$ µm$^{-1}$)', fontsize=16)

        ax2.get_xaxis().set_label_coords(0.5, -0.1)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        ax2.get_yaxis().set_label_coords(-0.1, 0.5)

        ax1.plot(spec_extract.spectral_axis, spec_extract.flux, color='black', label=self.spec_name)
        ax1.plot(spec_extract.spectral_axis, continuum_fit, color='tab:blue', label='Continuum fit')

        ax2.plot(spec_cont_sub.spectral_axis, spec_cont_sub.flux,
                 color='black', label='Continuum subtracted')

        ax1.legend(loc='upper right', frameon=False, fontsize=12.)
        ax2.legend(loc='upper right', frameon=False, fontsize=12.)

        plt.savefig(plot_filename, bbox_inches='tight')
        plt.clf()
        plt.close()

        print(' [DONE]')

        # Overwrite original spectrum with continuum-subtracted spectrum
        self.spectrum[:, 1] = spec_cont_sub.flux

        self.continuum_check = True

    @typechecked
    def fit_gaussian(self,
                     tag: str,
                     min_num_live_points: float = 400,
                     bounds: Dict[str, Union[Tuple[float, float]]] = None,
                     output: str = 'ultranest/',
                     plot_filename: str = 'line_fit.pdf') -> None:
        """
        Method for fitting a Gaussian profile to an emission line and using ``UltraNest`` for
        sampling the posterior distributions and estimating the evidence.

        Parameters
        ----------
        tag : str
            Database tag where the posterior samples will be stored.
        min_num_live_points : int
            Minimum number of live points (see
            https://johannesbuchner.github.io/UltraNest/issues.html).
        bounds : dict(str, tuple(float, float)), None
            The boundaries that are used for the uniform priors. Conservative prior boundaries will
            be estimated from the spectrum if the argument is set to ``None`` or if any of the
            required parameters is missing in the ``bounds`` dictionary.
        output : str
            Path that is used for the output files from ``UltraNest``.
        plot_filename : str
            Filename for the plot with the best-fit line profile.

        Returns
        -------
        NoneType
            None
        """

        high_spec_res = 1e5

        @typechecked
        def gaussian_function(amplitude: float,
                              mean: float,
                              sigma: float,
                              wavel: np.ndarray):

            return amplitude * np.exp(-0.5*(wavel-mean)**2/sigma**2)

        # Model parameters

        modelpar = ['gauss_amplitude', 'gauss_mean', 'gauss_sigma']

        # Create a dictionary with the cube indices of the parameters

        cube_index = {}
        for i, item in enumerate(modelpar):
            cube_index[item] = i

        # Check if all prior boundaries are present

        if bounds is None:
            bounds = {}

        if 'gauss_amplitude' not in bounds:
            bounds['gauss_amplitude'] = (0., 2.*np.amax(self.spectrum[:, 1]))

        if 'gauss_mean' not in bounds:
            bounds['gauss_mean'] = (self.spectrum[0, 0], self.spectrum[-1, 0])

        if 'gauss_sigma' not in bounds:
            bounds['gauss_sigma'] = (0., self.spectrum[-1, 0]-self.spectrum[0, 0])

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI
            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        @typechecked
        def lnprior_ultranest(cube: np.ndarray) -> np.ndarray:
            """
            Function for transforming the unit cube into the parameter cube.

            Parameters
            ----------
            cube : np.ndarray
                Array with unit parameters.

            Returns
            -------
            np.ndarray
                Array with physical parameters.
            """

            params = cube.copy()

            for item in cube_index:
                # Uniform priors for all parameters
                params[cube_index[item]] = bounds[item][0] + \
                    (bounds[item][1]-bounds[item][0])*params[cube_index[item]]

            return params

        @typechecked
        def lnlike_ultranest(params: np.ndarray) -> np.float64:
            """
            Function for calculating the log-likelihood for the sampled parameter cube.

            Parameters
            ----------
            params : np.ndarray
                Cube with physical parameters.

            Returns
            -------
            float
                Log-likelihood.
            """

            data_flux = self.spectrum[:, 1]
            data_var = self.spectrum[:, 2]**2

            model_flux = gaussian_function(params[cube_index['gauss_amplitude']],
                                           params[cube_index['gauss_mean']],
                                           params[cube_index['gauss_sigma']],
                                           self.spectrum[:, 0])

            chi_sq = -0.5 * (data_flux-model_flux)**2 / data_var

            return np.nansum(chi_sq)

        sampler = ultranest.ReactiveNestedSampler(modelpar,
                                                  lnlike_ultranest,
                                                  transform=lnprior_ultranest,
                                                  resume='subfolder',
                                                  log_dir=output)

        result = sampler.run(show_status=True,
                             viz_callback=False,
                             min_num_live_points=min_num_live_points)

        # Log-evidence

        ln_z = result['logz']
        ln_z_error = result['logzerr']
        print(f'Log-evidence = {ln_z:.2f} +/- {ln_z_error:.2f}')

        # Best-fit parameters

        print('Best-fit parameters (mean +/- std):')

        for i, item in enumerate(modelpar):
            mean = np.mean(result['samples'][:, i])
            std = np.std(result['samples'][:, i])

            print(f'   - {item} = {mean:.2e} +/- {std:.2e}')

        # Maximum likelihood sample

        print('Maximum likelihood sample:')

        max_lnlike = result['maximum_likelihood']['logl']
        print(f'   - Log-likelihood = {max_lnlike:.2e}')

        for i, item in enumerate(result['maximum_likelihood']['point']):
            print(f'   - {modelpar[i]} = {item:.2e}')

        # Posterior samples

        samples = result['samples']

        # Best-fit model parameters

        model_param = {'gauss_amplitude': np.median(samples[:, 0]),
                       'gauss_mean': np.median(samples[:, 1]),
                       'gauss_sigma': np.median(samples[:, 2])}

        best_model = read_util.gaussian_spectrum(
            self.wavel_range, model_param, spec_res=high_spec_res)

        # Interpolate high-resolution continuum

        if self.continuum_check:
            cont_interp = interp1d(self.spectrum[:, 0], self.continuum_flux, bounds_error=False)
            cont_high_res = cont_interp(best_model.wavelength)

        else:
            cont_high_res = np.full(best_model.wavelength.shape[0], 0.)

        # Add FWHM velocity

        modelpar.append('gauss_fwhm')

        gauss_mean = samples[:, 1]  # (um)
        gauss_fwhm = 2. * np.sqrt(2.*np.log(2.)) * samples[:, 2]  # (um)

        vel_fwhm = 1e-3*constants.LIGHT*gauss_fwhm/gauss_mean  # (km s-1)
        vel_fwhm = vel_fwhm[..., np.newaxis]

        samples = np.append(samples, vel_fwhm, axis=1)

        # Add line flux and luminosity

        print('Calculating line fluxes...', end='', flush=True)

        distance = self.object.get_distance()[0]

        modelpar.append('line_flux')
        modelpar.append('line_luminosity')

        line_flux = np.zeros(samples.shape[0])
        line_lum = np.zeros(samples.shape[0])

        if self.continuum_check:
            modelpar.append('line_eq_width')
            eq_width = np.zeros(samples.shape[0])

        for i in range(samples.shape[0]):
            model_param = {'gauss_amplitude': samples[i, 0],
                           'gauss_mean': samples[i, 1],
                           'gauss_sigma': samples[i, 2]}

            model_box = read_util.gaussian_spectrum(
                self.wavel_range, model_param, spec_res=high_spec_res)

            line_flux[i] = np.trapz(model_box.flux, model_box.wavelength)  # (W m-2)

            line_lum[i] = 4. * np.pi * (distance*constants.PARSEC)**2 * line_flux[i]  # (W)
            line_lum[i] /= constants.L_SUN  # (Lsun)

            if self.continuum_check:
                # Normalize the spectrum to the continuum
                spec_norm = (model_box.flux+cont_high_res) / cont_high_res

                # Check if the flux is NaN (due to interpolation errors at the spectrum edge)
                indices = ~np.isnan(spec_norm)

                eq_width[i] = np.trapz(1.-spec_norm[indices], model_box.wavelength[indices])  # (um)
                eq_width[i] *= 1e4  # (A)

        line_flux = line_flux[..., np.newaxis]
        samples = np.append(samples, line_flux, axis=1)

        line_lum = line_lum[..., np.newaxis]
        samples = np.append(samples, line_lum, axis=1)

        if self.continuum_check:
            eq_width = eq_width[..., np.newaxis]
            samples = np.append(samples, eq_width, axis=1)

        print(' [DONE]')

        # Log-likelihood

        ln_prob = result['weighted_samples']['logl']

        # Log-evidence

        ln_z = result['logz']
        ln_z_error = result['logzerr']
        print(f'Log-evidence = {ln_z:.2f} +/- {ln_z_error:.2f}')

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI
            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only possible when using a single process

            species_db = database.Database()

            species_db.add_samples(sampler='ultranest',
                                   samples=samples,
                                   ln_prob=ln_prob,
                                   ln_evidence=(ln_z, ln_z_error),
                                   mean_accept=None,
                                   spectrum=('model', 'gaussian'),
                                   tag=tag,
                                   modelpar=modelpar,
                                   distance=distance,
                                   spec_labels=None)

        # Create plot

        print(f'Plotting best-fit line profile: {plot_filename}...', end='', flush=True)

        mpl.rcParams['font.serif'] = ['Bitstream Vera Serif']
        mpl.rcParams['font.family'] = 'serif'

        plt.rc('axes', edgecolor='black', linewidth=2)
        plt.rcParams['axes.axisbelow'] = False

        plt.figure(1, figsize=(6, 6))
        gs = mpl.gridspec.GridSpec(2, 1)
        gs.update(wspace=0, hspace=0.1, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])

        ax1.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True, bottom=True,
                        left=True, right=True, labelbottom=False)

        ax1.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True, bottom=True,
                        left=True, right=True, labelbottom=False)

        ax2.tick_params(axis='both', which='major', colors='black', labelcolor='black',
                        direction='in', width=1, length=5, labelsize=12, top=True, bottom=True,
                        left=True, right=True)

        ax2.tick_params(axis='both', which='minor', colors='black', labelcolor='black',
                        direction='in', width=1, length=3, labelsize=12, top=True, bottom=True,
                        left=True, right=True)

        ax2.set_xlabel('Wavelength (µm)', fontsize=16)
        ax1.set_ylabel('Flux (W m$^{-2}$ µm$^{-1}$)', fontsize=16)
        ax2.set_ylabel('Flux (W m$^{-2}$ µm$^{-1}$)', fontsize=16)

        ax2.get_xaxis().set_label_coords(0.5, -0.1)
        ax1.get_yaxis().set_label_coords(-0.1, 0.5)
        ax2.get_yaxis().set_label_coords(-0.1, 0.5)

        ax1.plot(self.spectrum[:, 0], self.spectrum[:, 1]+self.continuum_flux,
                 color='black', label=self.spec_name)

        ax1.plot(best_model.wavelength, best_model.flux+cont_high_res,
                 color='tab:blue', label='Best-fit model (continuum + line)')

        ax2.plot(self.spectrum[:, 0], self.spectrum[:, 1], color='black', label=self.spec_name)

        ax2.plot(best_model.wavelength, best_model.flux, color='tab:blue',
                 label='Best-fit line profile')

        ax1.legend(loc='upper left', frameon=False, fontsize=12.)
        ax2.legend(loc='upper left', frameon=False, fontsize=12.)

        plt.savefig(plot_filename, bbox_inches='tight')
        plt.clf()
        plt.close()

        print(' [DONE]')
