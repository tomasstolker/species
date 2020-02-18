"""
Utility functions for photometry.
"""

import math

import spectres
import numpy as np

from species.core import box
from species.read import read_model, read_calibration, read_filter, read_planck


def multi_photometry(datatype,
                     spectrum,
                     filters,
                     parameters):
    """
    Parameters
    ----------
    datatype : str
        Data type ('model' or 'calibration').
    spectrum : str
        Spectrum name (e.g., 'drift-phoenix').
    filters : tuple(str, )
        Filter IDs.
    parameters : dict
        Parameters and values for the spectrum

    Returns
    -------
    species.core.box.SynphotBox
        Box with synthetic photometry.
    """

    print('Calculating synthetic photometry...', end='', flush=True)

    flux = {}

    if datatype == 'model':
        for item in filters:
            if spectrum == 'planck':
                readmodel = read_planck.ReadPlanck(filter_name=item)
            else:
                readmodel = read_model.ReadModel(spectrum, filter_name=item)

            flux[item] = readmodel.get_flux(parameters)[0]

    elif datatype == 'calibration':
        for item in filters:
            readcalib = read_calibration.ReadCalibration(spectrum, filter_name=item)
            flux[item] = readcalib.get_flux(parameters)[0]

    print(' [DONE]')

    return box.create_box('synphot', name='synphot', flux=flux)


def apparent_to_absolute(app_mag,
                         distance):
    """
    Function for converting an apparent magnitude into an absolute magnitude. The uncertainty on
    the distance is propagated into the uncertainty on the absolute magnitude.

    Parameters
    ----------
    app_mag : tuple(float, float), tuple(numpy.ndarray, numpy.ndarray)
        Apparent magnitude and uncertainty (mag). The returned error on the absolute magnitude
        is set to None if the error on the apparent magnitude is set to None, for example
        ``app_mag=(15., None)``.
    distance : tuple(float, float), tuple(numpy.ndarray, numpy.ndarray)
        Distance and uncertainty (pc). The error is not propagated into the error on the absolute
        magnitude if set to None, for example ``distance=(20., None)``.

    Returns
    -------
    float, numpy.ndarray
        Absolute magnitude (mag).
    float, numpy.ndarray, None
        Uncertainty (mag).
    """

    abs_mag = app_mag[0] - 5.*np.log10(distance[0]) + 5.

    if app_mag[1] is not None and distance[1] is not None:
        dist_err = distance[1] * (5./(distance[0]*math.log(10.)))
        abs_err = math.sqrt(app_mag[1]**2 + dist_err**2)

    elif app_mag[1] is not None and distance[1] is None:
        abs_err = app_mag[1]

    else:
        abs_err = None

    return abs_mag, abs_err


def get_residuals(datatype,
                  spectrum,
                  parameters,
                  filters,
                  objectbox,
                  inc_phot=True,
                  inc_spec=False):
    """
    Parameters
    ----------
    datatype : str
        Data type ('model' or 'calibration').
    spectrum : str
        Name of the atmospheric model or calibration spectrum.
    parameters : dict
        Parameters and values for the spectrum
    filters : tuple(str, )
        Filter IDs. All available photometry of the object is used if set to None.
    objectbox : species.core.box.ObjectBox
        Box with the photometry and/or spectrum of an object.
    inc_phot : bool
        Include photometry.
    inc_spec : bool
        Include spectrum.

    Returns
    -------
    species.core.box.ResidualsBox
        Box with the photometry and/or spectrum residuals.
    """

    if filters is None:
        filters = objectbox.filters

    if inc_phot:
        model_phot = multi_photometry(datatype=datatype,
                                      spectrum=spectrum,
                                      filters=filters,
                                      parameters=parameters)

        res_phot = np.zeros((2, len(objectbox.flux)))

        for i, item in enumerate(filters):
            transmission = read_filter.ReadFilter(item)

            res_phot[0, i] = transmission.mean_wavelength()
            res_phot[1, i] = (objectbox.flux[item][0]-model_phot.flux[item])/objectbox.flux[item][1]

    else:
        res_phot = None

    print('Calculating residuals...', end='', flush=True)

    if inc_spec:
        res_spec = {}

        for key in objectbox.spectrum:
            wavel_range = (0.9*objectbox.spectrum[key][0][0, 0],
                           1.1*objectbox.spectrum[key][0][-1, 0])

            if spectrum == 'planck':
                readmodel = read_planck.ReadPlanck(wavel_range=wavel_range)
                model = readmodel.get_spectrum(model_param=parameters, spec_res=1000.)

            else:
                readmodel = read_model.ReadModel(spectrum, wavel_range=wavel_range)
                model = readmodel.get_model(parameters, spec_res=None)

            wl_new = objectbox.spectrum[key][0][:, 0]

            flux_new = spectres.spectres(new_spec_wavs=wl_new,
                                         old_spec_wavs=model.wavelength,
                                         spec_fluxes=model.flux,
                                         spec_errs=None)

            res_tmp = (objectbox.spectrum[key][0][:, 1]-flux_new)/objectbox.spectrum[key][0][:, 2]

            res_spec[key] = np.column_stack([wl_new, res_tmp])

    else:
        res_spec = None

    print(' [DONE]')

    print('Residuals [sigma]:')

    if res_phot is not None:
        for i, item in enumerate(filters):
            print(f'   - {item}: {res_phot[1, i]:.2f}')

    if res_spec is not None:
        for key in objectbox.spectrum:
            print(f'   - {key}: min: {np.amin(res_spec[key]):.2f}, max: {np.amax(res_spec[key]):.2f}')

    return box.create_box(boxtype='residuals',
                          name=objectbox.name,
                          photometry=res_phot,
                          spectrum=res_spec)
