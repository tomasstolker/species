"""
Utility functions.
"""

import numpy as np


def update_sptype(sptypes):
    """
    Function to update a list with spectral types to two characters (e.g., M8, L3, or T1).

    :param sptypes: Spectral types.
    :type sptypes: numpy.ndarray

    :return: Updated spectral types.
    :rtype: numpy.ndarray
    """

    mlty = ('M', 'L', 'T', 'Y')

    for i, spt in enumerate(sptypes):
        if spt == 'None':
            pass

        elif spt == 'null':
            sptypes[i] = 'None'

        else:
            for item in mlty:
                try:
                    sp_index = spt.index(item)
                    sptypes[i] = spt[sp_index:sp_index+2]

                except ValueError:
                    pass

    return sptypes


def update_filter(filter_in):
    """
    Function to update filter ID from the Vizier Photometry viewer VOTable to the filter ID from
    the SVO Filter Profile Service.

    :param filter_in: Filter ID in the format of the Vizier Photometry viewer
    :type filter_in: str

    :return: Filter ID in the format of the SVO Filter Profile Service.
    :rtype: str
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
              wavelength,
              flux):
    """
    :param teff:
    :type teff: numpy.ndarray
    :param logg:
    :type logg: numpy.ndarray
    :param feh:
    :type feh: numpy.ndarray
    :param wavelength:
    :type wavelength: numpy.ndarray
    :param flux:
    :type flux: numpy.ndarray

    :return:
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    teff_unique = np.unique(teff)
    logg_unique = np.unique(logg)
    feh_unique = np.unique(feh)

    spectrum = np.zeros((teff_unique.shape[0],
                         logg_unique.shape[0],
                         feh_unique.shape[0],
                         wavelength.shape[0]))

    for i in range(teff.shape[0]):
        index_teff = np.argwhere(teff_unique == teff[i])[0]
        index_logg = np.argwhere(logg_unique == logg[i])[0]
        index_feh = np.argwhere(feh_unique == feh[i])[0]

        spectrum[index_teff, index_logg, index_feh, :] = flux[i]

    return (teff_unique, logg_unique, feh_unique, wavelength, spectrum)


def write_data(model,
               database,
               data_sorted):
    """
    :param model: Atmosphere model.
    :type model: str
    :param database:
    :type database: h5py._hl.files.File

    :return: None
    """

    if 'models/'+model in database:
        del database['models/'+model]

    dset = database.create_group('models/'+model)

    dset.attrs['nparam'] = int(3)
    dset.attrs['parameter0'] = str('teff')
    dset.attrs['parameter1'] = str('logg')
    dset.attrs['parameter2'] = str('feh')

    database.create_dataset('models/'+model+'/teff',
                            data=data_sorted[0],
                            dtype='f')

    database.create_dataset('models/'+model+'/logg',
                            data=data_sorted[1],
                            dtype='f')

    database.create_dataset('models/'+model+'/feh',
                            data=data_sorted[2],
                            dtype='f')

    database.create_dataset('models/'+model+'/wavelength',
                            data=data_sorted[3],
                            dtype='f')

    database.create_dataset('models/'+model+'/flux',
                            data=data_sorted[4],
                            dtype='f')


def add_missing(model,
                database):
    """
    :param model: Atmosphere model.
    :type model: str
    :param database:
    :type database: h5py._hl.files.File

    :return: None
    """

    teff = np.asarray(database['models/'+model+'/teff'])
    logg = np.asarray(database['models/'+model+'/logg'])
    feh = np.asarray(database['models/'+model+'/feh'])
    flux = np.asarray(database['models/'+model+'/flux'])

    for i in range(teff.shape[0]):
        for j in range(logg.shape[0]):
            for k in range(feh.shape[0]):
                if np.count_nonzero(flux[i, j, k]) == 0:
                    scaling = (teff[i+1]-teff[i])/(teff[i+1]-teff[i-1])
                    flux[i, j, k] = scaling*flux[i+1, j, k] + (1.-scaling)*flux[i-1, j, k]

    del database['models/'+model+'/flux']

    database.create_dataset('models/'+model+'/flux',
                            data=flux,
                            dtype='f')
