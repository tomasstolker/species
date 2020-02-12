"""
Module with reading functionalities for calibration spectra.
"""

import os
import configparser

import h5py
import spectres
import numpy as np

from scipy.optimize import curve_fit

from species.analysis import photometry
from species.core import box
from species.read import read_filter


class ReadCalibration:
    """
    Class for reading a calibration spectrum from the database.
    """

    def __init__(self,
                 tag,
                 filter_name=None):
        """
        Parameters
        ----------
        tag : str
            Database tag of the calibration spectrum.
        filter_name : str, None
            Filter ID that is used for the wavelength range. Full spectrum is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        self.tag = tag
        self.filter_name = filter_name

        if filter_name is None:
            self.wavel_range = None

        else:
            transmission = read_filter.ReadFilter(filter_name)
            self.wavel_range = transmission.wavelength_range()

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def resample_spectrum(self,
                          wavel_points,
                          apply_mask=False):
        """
        Function for resampling of a spectrum and uncertainties onto a new wavelength grid.

        Parameters
        ----------
        wavel_points : numpy.ndarray
            Wavelength points (micron).
        apply_mask : bool
            Exclude negative values and NaN values.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the resampled spectrum.
        """

        calibbox = self.get_spectrum()

        if apply_mask:
            indices = np.where(calibbox.flux > 0.)[0]

            calibbox.wavelength = calibbox.wavelength[indices]
            calibbox.flux = calibbox.flux[indices]
            calibbox.error = calibbox.error[indices]

        flux_new, error_new = spectres.spectres(new_spec_wavs=wavel_points,
                                                old_spec_wavs=calibbox.wavelength,
                                                spec_fluxes=calibbox.flux,
                                                spec_errs=calibbox.error)

        return box.create_box(boxtype='spectrum',
                              spectrum='calibration',
                              wavelength=wavel_points,
                              flux=flux_new,
                              error=error_new,
                              name=self.tag,
                              simbad=None,
                              sptype=None,
                              distance=None)

    def get_spectrum(self,
                     model_param=None,
                     apply_mask=False,
                     spec_res=None,
                     extrapolate=False,
                     min_wavelength=None):
        """
        Function for selecting the calibration spectrum.

        Parameters
        ----------
        model_param : dict, None
            Model parameters. Should contain the 'scaling' value. Not used if set to None.
        apply_mask : bool
            Exclude negative values and NaN values.
        spec_res : float, None
            Spectral resolution. Original wavelength points are used if set to None.
        extrapolate : bool
            Extrapolate to 6 micron by fitting a power law function.
        min_wavelength : float, None
            Minimum wavelength used for fitting the power law function. All data is used if set
            to None.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the spectrum.
        """

        with h5py.File(self.database, 'r') as h5_file:
            data = np.asarray(h5_file['spectra/calibration/'+self.tag])

            wavelength = np.asarray(data[0, ])
            flux = np.asarray(data[1, ])
            error = np.asarray(data[2, ])

        if apply_mask:
            indices = np.where(flux > 0.)[0]

            wavelength = wavelength[indices]
            flux = flux[indices]
            error = error[indices]

        if model_param is not None:
            flux = model_param['scaling']*flux

        if self.wavel_range is None:
            wl_index = np.ones(wavelength.size, dtype=bool)
        else:
            wl_index = (flux > 0.) & (wavelength > self.wavel_range[0]) & \
                       (wavelength < self.wavel_range[1])

        count = np.count_nonzero(wl_index)

        if count > 0:
            index = np.where(wl_index)[0]

            if index[0] > 0:
                wl_index[index[0] - 1] = True

            if index[-1] < len(wl_index)-1:
                wl_index[index[-1] + 1] = True

            wavelength = wavelength[wl_index]
            flux = flux[wl_index]
            error = error[wl_index]

        if extrapolate:
            def _power_law(wavelength, offset, scaling, power_index):
                return offset + scaling*wavelength**power_index

            if min_wavelength:
                indices = np.where(wavelength > min_wavelength)[0]
            else:
                indices = np.arange(0, wavelength.size, 1)

            popt, pcov = curve_fit(f=_power_law,
                                   xdata=wavelength[indices],
                                   ydata=flux[indices],
                                   p0=(0., np.mean(flux[indices]), -1.),
                                   sigma=error[indices])

            sigma = np.sqrt(np.diag(pcov))

            print(f'Fit result for f(x) = a + b*x^c:')
            print(f'a = {popt[0]} +/- {sigma[0]}')
            print(f'b = {popt[1]} +/- {sigma[1]}')
            print(f'c = {popt[2]} +/- {sigma[2]}')

            while wavelength[-1] <= 6.:
                wl_add = wavelength[-1] + wavelength[-1]/1000.

                wavelength = np.append(wavelength, wl_add)
                flux = np.append(flux, _power_law(wl_add, popt[0], popt[1], popt[2]))
                error = np.append(error, 0.)

        if spec_res is not None:
            wavelength_new = [wavelength[0]]

            while wavelength_new[-1] < wavelength[-1]:
                wavelength_new.append(wavelength_new[-1] + wavelength_new[-1]/spec_res)

            wavelength_new = np.asarray(wavelength_new[:-1])

            value_error = True

            while value_error:
                try:
                    flux_new, error_new = spectres.spectres(new_spec_wavs=wavelength_new,
                                                            old_spec_wavs=wavelength,
                                                            spec_fluxes=flux,
                                                            spec_errs=error)

                    value_error = False

                except ValueError:
                    wavelength_new = wavelength_new[1:-1]
                    value_error = True

            wavelength = wavelength_new
            flux = flux_new
            error = error_new

        return box.create_box(boxtype='spectrum',
                              spectrum='calibration',
                              wavelength=wavelength,
                              flux=flux,
                              error=error,
                              name=self.tag,
                              simbad=None,
                              sptype=None,
                              distance=None)

    def get_flux(self,
                 model_param=None):
        """
        Function for calculating the average flux density for the ``filter_name``.

        Parameters
        ----------
        model_param : dict, None
            Model parameters. Should contain the 'scaling' value. Not used if set to None.

        Returns
        -------
        float
            Average flux density (W m-2 micron-1).
        """

        specbox = self.get_spectrum(model_param=model_param)

        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(specbox.wavelength, specbox.flux)

    def get_magnitude(self,
                      model_param=None,
                      distance=None):
        """
        Function for calculating the apparent magnitude for the ``filter_name``.

        Parameters
        ----------
        model_param : dict, None
            Model parameters. Should contain the 'scaling' value. Not used if set to None.
        distance : float, None
            Distance to the calibration objects (pc).

        Returns
        -------
        float
            Apparent magnitude (mag).
        float, None
            Absolute magnitude (mag).
        """

        specbox = self.get_spectrum(model_param=model_param)

        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_magnitude(specbox.wavelength,
                                             specbox.flux,
                                             distance=distance)
