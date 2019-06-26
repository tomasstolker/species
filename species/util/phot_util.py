"""
Utility functions for photometry.
"""

import sys

import spectres
import numpy as np

from species.core import box
from species.read import read_model, read_calibration, read_filter


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

    sys.stdout.write('Calculating synthetic photometry...')
    sys.stdout.flush()

    flux = {}

    if datatype == 'model':
        for item in filters:
            readmodel = read_model.ReadModel(spectrum, item)
            flux[item] = readmodel.get_photometry(parameters)

    elif datatype == 'calibration':
        for item in filters:
            readcalib = read_calibration.ReadCalibration(spectrum, item)
            flux[item] = readcalib.get_photometry(parameters)

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    return box.create_box('synphot', name='synphot', flux=flux)


def apparent_to_absolute(app_mag,
                         distance):
    """
    Parameters
    ----------
    app_mag : float or numpy.ndarray
        Apparent magnitude (mag).
    distance : float or numpy.ndarray
        Distance (pc).

    Returns
    -------
    float or numpy.ndarray
        Absolute magnitude (mag).
    """

    return app_mag - 5.*np.log10(distance) + 5.


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
        filters = objectbox.filter

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

    sys.stdout.write('Calculating residuals...')
    sys.stdout.flush()

    if inc_spec:
        wl_range = (0.9*objectbox.spectrum[0, 0], 1.1*objectbox.spectrum[-1, 0])

        readmodel = read_model.ReadModel(spectrum, wl_range)
        model = readmodel.get_model(parameters)

        wl_new = objectbox.spectrum[:, 0]

        flux_new = spectres.spectres(new_spec_wavs=wl_new,
                                     old_spec_wavs=model.wavelength,
                                     spec_fluxes=model.flux,
                                     spec_errs=None)

        res_spec = np.zeros((2, objectbox.spectrum.shape[0]))

        res_spec[0, :] = wl_new
        res_spec[1, :] = (objectbox.spectrum[:, 1]-flux_new)/objectbox.spectrum[:, 2]

    else:
        res_spec = None

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    return box.create_box(boxtype='residuals',
                          name=objectbox.name,
                          photometry=res_phot,
                          spectrum=res_spec)
