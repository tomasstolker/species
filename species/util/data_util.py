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
    Function to update afilter ID from the Vizier Photometry viewer VOTable to the filter ID from
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
    tuple(numpy.ndarray, )
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
        index_teff = np.argwhere(teff_unique == teff[i])[0]
        index_logg = np.argwhere(logg_unique == logg[i])[0]

        if feh is None and co is None and fsed is None:
            spectrum[index_teff, index_logg, :] = flux[i]

        elif feh is not None and co is None and fsed is None:
            index_feh = np.argwhere(feh_unique == feh[i])[0]
            spectrum[index_teff, index_logg, index_feh, :] = flux[i]

        elif feh is not None and co is not None and fsed is None:
            index_feh = np.argwhere(feh_unique == feh[i])[0]
            index_co = np.argwhere(co_unique == co[i])[0]
            spectrum[index_teff, index_logg, index_feh, index_co, :] = flux[i]

        elif feh is not None and co is None and fsed is not None:
            index_feh = np.argwhere(feh_unique == feh[i])[0]
            index_fsed = np.argwhere(fsed_unique == fsed[i])[0]
            spectrum[index_teff, index_logg, index_feh, index_fsed, :] = flux[i]

        else:
            index_feh = np.argwhere(feh_unique == feh[i])[0]
            index_co = np.argwhere(co_unique == co[i])[0]
            index_fsed = np.argwhere(fsed_unique == fsed[i])[0]
            spectrum[index_teff, index_logg, index_feh, index_co, index_fsed, :] = flux[i]

    if feh is None and co is None and fsed is None:
        sorted_data = (teff_unique, logg_unique, wavelength, spectrum)
    elif feh is not None and co is None and fsed is None:
        sorted_data = (teff_unique, logg_unique, feh_unique, wavelength, spectrum)
    elif feh is not None and co is not None and fsed is None:
        sorted_data = (teff_unique, logg_unique, feh_unique, co_unique, wavelength, spectrum)
    elif feh is not None and co is None and fsed is not None:
        sorted_data = (teff_unique, logg_unique, feh_unique, fsed_unique, wavelength, spectrum)
    else:
        sorted_data = (teff_unique, logg_unique, feh_unique, co_unique, fsed_unique, wavelength, spectrum)

    return sorted_data


def write_data(model,
               parameters,
               database,
               data_sorted):
    """
    Parameters
    ----------
    model : str
        Atmosphere model.
    parameters : tuple(str, )
        Model parameters.
    database: h5py._hl.files.File
        Database.
    data_sorted : tuple(numpy.ndarray, )

    Returns
    -------
    None
    """

    if 'models/'+model in database:
        del database['models/'+model]

    dset = database.create_group('models/'+model)

    dset.attrs['nparam'] = len(parameters)

    for i, item in enumerate(parameters):
        dset.attrs['parameter'+str(i)] = item

        database.create_dataset('models/'+model+'/'+item,
                                data=data_sorted[i],
                                dtype='f')

    database.create_dataset('models/'+model+'/wavelength',
                            data=data_sorted[len(parameters)],
                            dtype='f')

    database.create_dataset('models/'+model+'/flux',
                            data=data_sorted[len(parameters)+1],
                            dtype='f')


def add_missing(model,
                parameters,
                database):
    """
    Parameters
    ----------
    model : str
        Atmosphere model.
    parameters : tuple(str, )
        Model parameters.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    None
    """

    grid_shape = []
    for item in parameters:
        grid_shape.append(database['models/'+model+'/'+item].shape[0])

    teff = np.asarray(database['models/'+model+'/teff'])
    flux = np.asarray(database['models/'+model+'/flux'])

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
                                flux[i, j, k, m] = scaling*flux[i+1, j, k, m] + (1.-scaling)*flux[i-1, j, k, m]

                            except IndexError:
                                flux[i, j, k, m] = np.nan
                                warnings.warn(f'Interpolation is not possible at the edge of the '
                                              f'parameter grid. A NaN value is stored for Teff = '
                                              f'{teff[i]} K.')

    del database['models/'+model+'/flux']

    database.create_dataset('models/'+model+'/flux',
                            data=flux,
                            dtype='f')
