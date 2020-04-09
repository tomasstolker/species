"""
Utility functions for data processing.
"""

import warnings

import numpy as np

from scipy.interpolate import griddata


def update_sptype(sptypes):
    """
    Function to update a list with spectral types to two characters (e.g., M8, L3, or T1).

    Parameters
    ----------
    sptypes : np.ndarray
        Input spectral types.

    Returns
    -------
    np.ndarray
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


def sort_data(param_teff,
              param_logg,
              param_feh,
              param_co,
              param_fsed,
              wavelength,
              flux):
    """
    Parameters
    ----------
    param_teff : np.ndarray
        Array with the effective temperature (K) of each spectrum.
    param_logg : np.ndarray
        Array with the log10 surface gravity (cgs) of each spectrum.
    param_feh : np.ndarray, None
        Array with the metallicity of each spectrum. Not used if set to None.
    param_co : np.ndarray, None
        Array with the carbon-to-oxygen ratio of each spectrum. Not used if set to None.
    param_fsed : np.ndarray, None
        Array with the sedimentation parameter of each spectrum. Not used if set to None.
    wavelength : np.ndarray
        Array with the wavelengths (um).
    flux : np.ndarray
        Array with the spectra (n_spectra, n_wavelengths).

    Returns
    -------
    list(np.ndarray, )
        List with the unique values of the atmosphere parameters (each in a separate array), an
        array with the wavelengths, and a multidimensional array with the sorted spectra.
    """

    teff_unique = np.unique(param_teff)
    logg_unique = np.unique(param_logg)

    spec_shape = [teff_unique.shape[0], logg_unique.shape[0]]

    if param_feh is not None:
        feh_unique = np.unique(param_feh)
        spec_shape.append(feh_unique.shape[0])

    if param_co is not None:
        co_unique = np.unique(param_co)
        spec_shape.append(co_unique.shape[0])

    if param_fsed is not None:
        fsed_unique = np.unique(param_fsed)
        spec_shape.append(fsed_unique.shape[0])

    spec_shape.append(wavelength.shape[0])

    spectrum = np.zeros(spec_shape)

    for i in range(param_teff.shape[0]):
        # The parameter order: Teff, log(g), [Fe/H], C/O, f_sed
        # Not all parameters have to be included but the order matters

        index_teff = np.argwhere(teff_unique == param_teff[i])[0][0]
        index_logg = np.argwhere(logg_unique == param_logg[i])[0][0]

        spec_select = [index_teff, index_logg]

        if param_feh is not None:
            index_feh = np.argwhere(feh_unique == param_feh[i])[0][0]
            spec_select.append(index_feh)

        if param_co is not None:
            index_co = np.argwhere(co_unique == param_co[i])[0][0]
            spec_select.append(index_co)

        if param_fsed is not None:
            index_fsed = np.argwhere(fsed_unique == param_fsed[i])[0][0]
            spec_select.append(index_fsed)

        spec_select.append(...)

        spectrum[tuple(spec_select)] = flux[i]

    sorted_data = [teff_unique, logg_unique]

    if param_feh is not None:
        sorted_data.append(feh_unique)

    if param_co is not None:
        sorted_data.append(co_unique)

    if param_fsed is not None:
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
    data_sorted : list(np.ndarray, )
        Sorted model data with the parameter values, wavelength points (um), and flux
        densities (W m-2 um-1).

    Returns
    -------
    NoneType
        None
    """

    if f'models/{model}' in database:
        del database[f'models/{model}']

    dset = database.create_group(f'models/{model}')

    dset.attrs['n_param'] = len(parameters)

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
    Function for adding missing grid points by linearly interpolating the available grid points.

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

    print(f'Number of grid points per parameter:')

    grid_shape = []
    param_data = []

    for i, item in enumerate(parameters):
        grid_shape.append(database[f'models/{model}/{item}'].shape[0])
        param_data.append(np.asarray(database[f'models/{model}/{item}']))
        print(f'   - {item}: {grid_shape[i]}')

    teff = np.asarray(database[f'models/{model}/teff'])
    wavelength = np.asarray(database[f'models/{model}/wavelength'])
    flux = np.asarray(database[f'models/{model}/flux'])

    count_total = 0
    count_missing = 0

    print('Fixing missing grid points:')

    if len(parameters) == 2:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], []]
        new_points = [[], []]

        new_flux = np.zeros((grid_shape[0], grid_shape[1], wavelength.shape[0]))

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if np.count_nonzero(flux[i, j, ...]) == 0:
                    find_missing[i, j] = True

                else:
                    points[0].append(param_data[0][i])
                    points[1].append(param_data[1][j])

                    values.append(flux[i, j, ...])

                new_points[0].append(param_data[0][i])
                new_points[1].append(param_data[1][j])

                count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        test = griddata(points.T, values, new_points.T, method='linear')

        for item in test:
            if np.isnan(item[0]):
                count_missing += 1

        count_interp = np.sum(find_missing) - count_missing

        count = 0
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                new_flux[i, j, :] = test[count, :]
                count += 1

    elif len(parameters) == 3:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], [], []]
        new_points = [[], [], []]

        new_flux = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2],  wavelength.shape[0]))

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    if np.count_nonzero(flux[i, j, k, ...]) == 0:
                        find_missing[i, j, k] = True

                    else:
                        points[0].append(param_data[0][i])
                        points[1].append(param_data[1][j])
                        points[2].append(param_data[2][k])

                        values.append(flux[i, j, k, ...])

                    new_points[0].append(param_data[0][i])
                    new_points[1].append(param_data[1][j])
                    new_points[2].append(param_data[2][k])

                    count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        test = griddata(points.T, values, new_points.T, method='linear')

        for item in test:
            if np.isnan(item[0]):
                count_missing += 1

        count_interp = np.sum(find_missing) - count_missing

        count = 0
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    new_flux[i, j, k, :] = test[count, :]
                    count += 1

    elif len(parameters) == 4:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], [], [], []]
        new_points = [[], [], [], []]

        new_flux = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3], wavelength.shape[0]))

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        if np.count_nonzero(flux[i, j, k, m, ...]) == 0:
                            find_missing[i, j, k, m] = True

                        else:
                            points[0].append(param_data[0][i])
                            points[1].append(param_data[1][j])
                            points[2].append(param_data[2][k])
                            points[3].append(param_data[3][m])

                            values.append(flux[i, j, k, m, ...])

                        new_points[0].append(param_data[0][i])
                        new_points[1].append(param_data[1][j])
                        new_points[2].append(param_data[2][k])
                        new_points[3].append(param_data[3][m])

                        count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        test = griddata(points.T, values, new_points.T, method='linear')

        for item in test:
            if np.isnan(item[0]):
                count_missing += 1

        count_interp = np.sum(find_missing) - count_missing

        count = 0
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        new_flux[i, j, k, m, :] = test[count, :]
                        count += 1

    elif len(parameters) == 5:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], [], [], [], []]
        new_points = [[], [], [], [], []]

        new_flux = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3], grid_shape[4], wavelength.shape[0]))

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        for n in range(grid_shape[4]):
                            if np.count_nonzero(flux[i, j, k, m, n, ...]) == 0:
                                find_missing[i, j, k, m, n] = True

                            else:
                                points[0].append(param_data[0][i])
                                points[1].append(param_data[1][j])
                                points[2].append(param_data[2][k])
                                points[3].append(param_data[3][m])
                                points[4].append(param_data[4][n])

                                values.append(flux[i, j, k, m, n, ...])

                            new_points[0].append(param_data[0][i])
                            new_points[1].append(param_data[1][j])
                            new_points[2].append(param_data[2][k])
                            new_points[3].append(param_data[3][m])
                            new_points[4].append(param_data[4][n])

                            count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        test = griddata(points.T, values, new_points.T, method='linear')

        for item in test:
            if np.isnan(item[0]):
                count_missing += 1

        count_interp = np.sum(find_missing) - count_missing

        count = 0
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        for n in range(grid_shape[4]):
                            new_flux[i, j, k, m, n, :] = test[count, :]
                            count += 1

    else:
        raise ValueError('The add_missing function is currently not compatible with more than 5 '
                         'parameters.')

    # if len(parameters) == 4:
    #     check_constant = np.zeros(grid_shape, dtype=bool)
    #
    #     for z in range(5):
    #         for i in range(grid_shape[0]):
    #             for j in range(grid_shape[1]):
    #                 for k in range(grid_shape[2]):
    #                     for m in range(grid_shape[3]):
    #                         if z == 0:
    #                             count_total += 1
    #
    #                         index = (i, j, k, m, ...)
    #
    #                         if np.count_nonzero(flux[index]) == 0:
    #                             for dim_index in range(len(grid_shape)):
    #
    #                                 if index[dim_index] > 0 and index[dim_index] < grid_shape[dim_index]-1:
    #                                     index_low = [i, j, k, m, ...]
    #                                     index_up = [i, j, k, m, ...]
    #
    #                                     index_low[dim_index] = index_low[dim_index] - 1
    #                                     index_up[dim_index] = index_up[dim_index] + 1
    #
    #                                     index_low = tuple(index_low)
    #                                     index_up = tuple(index_up)
    #
    #                                     if np.count_nonzero(flux[index_low]) != 0 and np.count_nonzero(flux[index_up]) != 0:
    #                                         scaling = (param_data[dim_index][index[dim_index]] - param_data[dim_index][index_low[dim_index]]) / (param_data[dim_index][index_up[dim_index]] - param_data[dim_index][index_low[dim_index]])
    #                                         flux[index] = flux[index_low]*(1.-scaling) + flux[index_up]*scaling
    #                                         count_interp += 1
    #                                         break
    #
    #     for z in range(2):
    #         for i in range(grid_shape[0]):
    #             for j in range(grid_shape[1]):
    #                 for k in range(grid_shape[2]):
    #                     for m in range(grid_shape[3]):
    #                         index = (i, j, k, m, ...)
    #
    #                         if np.count_nonzero(flux[index]) == 0:
    #                             for dim_index in range(len(grid_shape)):
    #
    #                                 if index[dim_index] > 0:
    #                                     index_low = [i, j, k, m, ...]
    #                                     index_low[dim_index] = index_low[dim_index] - 1
    #                                     index_low = tuple(index_low)
    #
    #                                     if np.count_nonzero(flux[index_low]) != 0:
    #                                         if z == 0 and check_constant[index_low[:-1]]:
    #                                             continue
    #
    #                                         flux[index] = flux[index_low]
    #                                         count_same += 1
    #                                         check_constant[index] = True
    #                                         print(param_data[0][i], param_data[1][j], param_data[2][k], param_data[3][m])
    #                                         break
    #
    #                                 elif index[dim_index] < grid_shape[dim_index]-1:
    #                                     index_up = [i, j, k, m, ...]
    #                                     index_up[dim_index] = index_up[dim_index] + 1
    #                                     index_up = tuple(index_up)
    #
    #                                     if np.count_nonzero(flux[index_up]) != 0:
    #                                         if z == 0 and check_constant[index_up[:-1]]:
    #                                             continue
    #
    #                                         flux[index] = flux[index_up]
    #                                         count_same += 1
    #                                         check_constant[index] = True
    #                                         print(param_data[0][i], param_data[1][j], param_data[2][k], param_data[3][m])
    #                                         break
    #
    #                         if np.count_nonzero(flux[index]) == 0:
    #                             print(z)
    #                             if z == 1:
    #                                 count_missing += 1
    #
    #                                 warnings.warn(f'It is not possible to add the missing grid position '
    #                                               f'at ({param_data[0][i]}, {param_data[1][j]}, '
    #                                               f'{param_data[2][k]}, {param_data[3][m]}). '
    #                                               f'Storing a spectrum with only zeros instead.')
    #
    # else:
    #     raise ValueError(f'Interpolation of missing grid points is not implemented for '
    #                      f'{len(parameters)} parameters.')

    print(f'   - Number of stored grid points: {count_total}')
    print(f'   - Number of interpolated grid points: {count_interp}')
    print(f'   - Number of missing grid points: {count_missing}')

    del database[f'models/{model}/flux']

    database.create_dataset(f'models/{model}/flux',
                            data=flux)


def correlation_to_covariance(cor_matrix,
                              spec_sigma):
    """
    Parameters
    ----------
    cor_matrix : np.ndarray
        Correlation matrix of the spectrum.
    spec_sigma : np.ndarray
        Uncertainties (W m-2 um-1).

    Returns
    -------
    np.ndarrays
        Covariance matrix of the spectrum.
    """

    cov_matrix = np.zeros(cor_matrix.shape)

    for i in range(cor_matrix.shape[0]):
        for j in range(cor_matrix.shape[1]):
            cov_matrix[i, j] = cor_matrix[i, j]*spec_sigma[i]*spec_sigma[j]

            if i == j:
                assert cor_matrix[i, j] == 1.

    return cov_matrix
