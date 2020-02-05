"""
Utility functions for data processing.
"""

import warnings

import numpy as np


def update_sptype(sptypes):
    """
    Function to update a list with spectral types to two characters (e.g., M8, L3, or T1).

    Parameters
    ----------
    sptypes : numpy.ndarray
        Input spectral types.

    Returns
    -------
    numpy.ndarray
        Updated spectral types.
    """

    sptype_list = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y']

    for i, spt_item in enumerate(sptypes):
        if spt_item == 'None':
            pass

        elif spt_item == 'null':
            sptypes[i] = 'None'

        else:
            for list_item in sptype_list:
                try:
                    sp_index = spt_item.index(list_item)
                    sptypes[i] = spt_item[sp_index:sp_index+2]

                except ValueError:
                    pass

    return sptypes


def update_filter(filter_in):
    """
    Function to update a filter ID from the Vizier Photometry viewer VOTable to the filter ID from
    the SVO Filter Profile Service.

    Parameters
    ----------
    filter_in : str
        Filter ID in the format of the Vizier Photometry viewer.

    Returns
    -------
    str
        Filter ID in the format of the SVO Filter Profile Service.
    """

    if filter_in[0:5] == b'2MASS':
        filter_out = str(b'2MASS/2MASS.'+filter_in[6:])

    elif filter_in[0:4] == b'WISE':
        filter_out = str(b'WISE/WISE.'+filter_in[5:])

    elif filter_in[0:10] == b'GAIA/GAIA2':
        filter_out = str(filter_in[0:9]+b'0'+filter_in[10:])

    else:
        filter_out = None

    return filter_out


def sort_data(teff,
              logg,
              feh,
              co,
              fsed,
              wavelength,
              flux):
    """
    Parameters
    ----------
    teff : numpy.ndarray
    logg : numpy.ndarray
    feh : numpy.ndarray, None
    co : numpy.ndarray, None
    fsed : numpy.ndarray, None
    wavelength : numpy.ndarray
    flux : numpy.ndarray

    Returns
    -------
    list(numpy.ndarray, )
    """

    teff_unique = np.unique(teff)
    logg_unique = np.unique(logg)

    if feh is None and co is None and fsed is None:
        spectrum = np.zeros((teff_unique.shape[0],
                             logg_unique.shape[0],
                             wavelength.shape[0]))

    elif feh is not None and co is None and fsed is None:
        feh_unique = np.unique(feh)

        spectrum = np.zeros((teff_unique.shape[0],
                             logg_unique.shape[0],
                             feh_unique.shape[0],
                             wavelength.shape[0]))

    elif feh is not None and co is not None and fsed is None:
        feh_unique = np.unique(feh)
        co_unique = np.unique(co)

        spectrum = np.zeros((teff_unique.shape[0],
                             logg_unique.shape[0],
                             feh_unique.shape[0],
                             co_unique.shape[0],
                             wavelength.shape[0]))

    elif feh is not None and co is None and fsed is not None:
        feh_unique = np.unique(feh)
        fsed_unique = np.unique(fsed)

        spectrum = np.zeros((teff_unique.shape[0],
                             logg_unique.shape[0],
                             feh_unique.shape[0],
                             fsed_unique.shape[0],
                             wavelength.shape[0]))

    else:
        feh_unique = np.unique(feh)
        co_unique = np.unique(co)
        fsed_unique = np.unique(fsed)

        spectrum = np.zeros((teff_unique.shape[0],
                             logg_unique.shape[0],
                             feh_unique.shape[0],
                             co_unique.shape[0],
                             fsed_unique.shape[0],
                             wavelength.shape[0]))

    for i in range(teff.shape[0]):
        index_teff = np.argwhere(teff_unique == teff[i])[0][0]
        index_logg = np.argwhere(logg_unique == logg[i])[0][0]

        if feh is None and co is None and fsed is None:
            spectrum[index_teff, index_logg, :] = flux[i]

        elif feh is not None and co is None and fsed is None:
            index_feh = np.argwhere(feh_unique == feh[i])[0][0]
            spectrum[index_teff, index_logg, index_feh, :] = flux[i]

        elif feh is not None and co is not None and fsed is None:
            index_feh = np.argwhere(feh_unique == feh[i])[0][0]
            index_co = np.argwhere(co_unique == co[i])[0][0]
            spectrum[index_teff, index_logg, index_feh, index_co, :] = flux[i]

            # for j, item in enumerate(flux[i]):
            #     spectrum[index_teff, index_logg, index_feh, index_co, j] = item

        elif feh is not None and co is None and fsed is not None:
            index_feh = np.argwhere(feh_unique == feh[i])[0][0]
            index_fsed = np.argwhere(fsed_unique == fsed[i])[0][0]
            spectrum[index_teff, index_logg, index_feh, index_fsed, :] = flux[i]

        else:
            index_feh = np.argwhere(feh_unique == feh[i])[0][0]
            index_co = np.argwhere(co_unique == co[i])[0][0]
            index_fsed = np.argwhere(fsed_unique == fsed[i])[0][0]
            spectrum[index_teff, index_logg, index_feh, index_co, index_fsed, :] = flux[i]

    sorted_data = [teff_unique, logg_unique]

    if feh is not None:
        sorted_data.append(feh_unique)

    if co is not None:
        sorted_data.append(co_unique)

    if fsed is not None:
        sorted_data.append(fsed_unique)

    sorted_data.append(wavelength)
    sorted_data.append(spectrum)

    return sorted_data


def write_data(model,
               parameters,
               database,
               data_sorted):
    """
    Function for writing the model spectra and parameters to the database.

    Parameters
    ----------
    model : str
        Atmosphere model.
    parameters : list(str, )
        Model parameters.
    database: h5py._hl.files.File
        Database.
    data_sorted : list(numpy.ndarray, )
        Sorted model data with the parameter values, wavelength points (micron), and flux
        densities (W m-2 micron-1).

    Returns
    -------
    NoneType
        None
    """

    if 'models/'+model in database:
        del database[f'models/{model}']

    dset = database.create_group(f'models/{model}')

    dset.attrs['nparam'] = len(parameters)

    for i, item in enumerate(parameters):
        dset.attrs[f'parameter{i}'] = item

        database.create_dataset(f'models/{model}/{item}',
                                data=data_sorted[i])

    database.create_dataset(f'models/{model}/wavelength',
                            data=data_sorted[len(parameters)])

    database.create_dataset(f'models/{model}/flux',
                            data=data_sorted[len(parameters)+1])


def add_missing(model,
                parameters,
                database):
    """
    Parameters
    ----------
    model : str
        Atmosphere model.
    parameters : list(str, )
        Model parameters.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    NoneType
        None
    """

    grid_shape = []
    for item in parameters:
        grid_shape.append(database[f'models/{model}/{item}'].shape[0])

    teff = np.asarray(database[f'models/{model}/teff'])
    flux = np.asarray(database[f'models/{model}/flux'])

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):

            if len(parameters) == 2:
                if np.count_nonzero(flux[i, j]) == 0:
                    try:
                        scaling = (teff[i+1]-teff[i])/(teff[i+1]-teff[i-1])
                        flux[i, j] = scaling*flux[i+1, j] + (1.-scaling)*flux[i-1, j]

                    except IndexError:
                        flux[i, j] = np.nan
                        warnings.warn(f'Interpolation is not possible at the edge of the '
                                      f'parameter grid. A NaN value is stored for Teff = '
                                      f'{teff[i]} K.')

            elif len(parameters) == 3:
                for k in range(grid_shape[2]):
                    if np.count_nonzero(flux[i, j, k]) == 0:
                        try:
                            scaling = (teff[i+1]-teff[i])/(teff[i+1]-teff[i-1])
                            flux[i, j, k] = scaling*flux[i+1, j, k] + (1.-scaling)*flux[i-1, j, k]

                        except IndexError:
                            flux[i, j, k] = np.nan
                            warnings.warn(f'Interpolation is not possible at the edge of the '
                                          f'parameter grid. A NaN value is stored for Teff = '
                                          f'{teff[i]} K.')

            elif len(parameters) == 4:
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        if np.count_nonzero(flux[i, j, k, m]) == 0:
                            try:
                                scaling = (teff[i+1]-teff[i])/(teff[i+1]-teff[i-1])
                                flux[i, j, k, m] = scaling*flux[i+1, j, k, m] + \
                                    (1.-scaling)*flux[i-1, j, k, m]

                            except IndexError:
                                flux[i, j, k, m] = np.nan
                                warnings.warn(f'Interpolation is not possible at the edge of the '
                                              f'parameter grid. A NaN value is stored for Teff = '
                                              f'{teff[i]} K.')

            elif len(parameters) == 5:
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        for n in range(grid_shape[4]):
                            if np.count_nonzero(flux[i, j, k, m, n]) == 0:
                                try:
                                    scaling = (teff[i+1]-teff[i])/(teff[i+1]-teff[i-1])
                                    flux[i, j, k, m, n] = scaling*flux[i+1, j, k, m, n] + \
                                        (1.-scaling)*flux[i-1, j, k, mm ]

                                except IndexError:
                                    flux[i, j, k, m, n] = np.nan
                                    warnings.warn(f'Interpolation is not possible at the edge of the '
                                                  f'parameter grid. A NaN value is stored for Teff = '
                                                  f'{teff[i]} K.')

            else:
                raise ValueError('The interpolation of missing data is not yet been implemented '
                                 'for 6 or more parameters.')

    del database[f'models/{model}/flux']

    database.create_dataset(f'models/{model}/flux',
                            data=flux)


def correlation_to_covariance(cor_matrix,
                              spec_sigma):
    """
    Parameters
    ----------
    cor_matrix : numpy.ndarray
        Correlation matrix of the spectrum.
    spec_sigma : numpy.ndarray
        Uncertainties (W m-2 micron-1).

    Returns
    -------
    numpy.ndarrays
        Covariance matrix of the spectrum.
    """

    cov_matrix = np.zeros(cor_matrix.shape)

    for i in range(cor_matrix.shape[0]):
        for j in range(cor_matrix.shape[1]):
            cov_matrix[i, j] = cor_matrix[i, j]*spec_sigma[i]*spec_sigma[j]

            if i == j:
                assert cor_matrix[i, j] == 1.

    return cov_matrix
