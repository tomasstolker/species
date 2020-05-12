"""
Module with functionalities for reading and writing of data.
"""

import os
import json
import warnings
import configparser

from typing import Optional, Union, List, Tuple, Dict

import h5py
import tqdm
import emcee
import numpy as np

from typeguard import typechecked
from astropy.io import fits

# from petitRADTRANS import Radtrans
# from petitRADTRANS_ck_test_speed import nat_cst as nc
# from petitRADTRANS_ck_test_speed import Radtrans as RadtransScatter

from species.analysis import photometry
from species.core import box, constants
from species.data import drift_phoenix, btnextgen, vega, irtf, spex, vlm_plx, leggett, \
                         companions, filters, btsettl, ames_dusty, ames_cond, \
                         isochrones, petitcode, exo_rem, dust
from species.read import read_model, read_calibration, read_planck
from species.util import data_util
# from species.util import data_util, retrieval_util


class Database:
    """
    Class for fitting atmospheric model spectra to photometric data.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']
        self.input_path = config['species']['data_folder']

    def list_content(self):
        """
        Returns
        -------
        NoneType
            None
        """

        print('Database content:')

        def descend(h5_object,
                    seperator=''):
            """
            Parameters
            ----------
            h5_object : h5py._hl.files.File, h5py._hl.group.Group, h5py._hl.dataset.Dataset
            separator : str

            Returns
            -------
            NoneType
                None
            """

            if isinstance(h5_object, (h5py._hl.files.File, h5py._hl.group.Group)):
                for key in h5_object.keys():
                    print(seperator+'- '+key+': '+str(h5_object[key]))
                    descend(h5_object[key], seperator=seperator+'\t')

            elif isinstance(h5_object, h5py._hl.dataset.Dataset):
                for key in h5_object.attrs.keys():
                    print(seperator+'- '+key+': '+str(h5_object.attrs[key]))

        with h5py.File(self.database, 'r') as hdf_file:
            descend(hdf_file)

    @staticmethod
    def list_companions():
        """
        Returns
        -------
        NoneType
            None
        """

        for planet_name, planet_dict in companions.get_data().items():
            distance = planet_dict['distance']
            app_mag = planet_dict['app_mag']

            print(f'Object name = {planet_name}')
            print(f'Distance (pc) = {distance[0]} +/- {distance[1]}')

            for mag_name, mag_dict in app_mag.items():
                print(f'{mag_name} (mag) = {mag_dict[0]} +/- {mag_dict[1]}')

            print()

    def delete_data(self,
                    dataset):
        """
        Function for deleting a dataset from the HDF5 database.

        Parameters
        ----------
        dataset : str
            Dataset path in the HDF5 database.

        Returns
        -------
        NoneType
            None
        """

        with h5py.File(self.database, 'a') as hdf_file:
            if dataset in hdf_file:
                del hdf_file[dataset]
            else:
                warnings.warn(f'The dataset {dataset} is not found in {self.database}.')

    @typechecked
    def add_companion(self,
                      name: Union[Optional[str], Optional[List[str]]]) -> None:
        """
        Function for adding the magnitudes of directly imaged planets and brown dwarfs from
        :class:`~species.data.companions.get_data` to the database.

        Parameters
        ----------
        name : str, list(str, ), None
            Name or list with names of the directly imaged planets and brown dwarfs (e.g.
            ``'HR 8799 b'`` or ``['HR 8799 b', '51 Eri b', 'PZ Tel B']``). All the available
            companion data are added if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        if isinstance(name, str):
            name = list((name, ))

        data = companions.get_data()

        if name is None:
            name = data.keys()

        for item in name:
            self.add_object(object_name=item,
                            distance=data[item]['distance'],
                            app_mag=data[item]['app_mag'])

    @typechecked
    def add_dust(self) -> None:
        """
        Function for adding optical constants of MgSiO3 and Fe to the database. The optical
        constants have been compiled by Mollière et al. (2019) for petitRADTRANS from the
        following sources:

        - MgSiO3, crystalline
            - Scott & Duley (1996), ApJS, 105, 401
            - Jäger et al. (1998), A&A, 339, 904

        - MgSiO3, amorphous
            - Jäger et al. (2003), A&A, 408, 193

        - Fe, crystalline
            - Henning & Stognienko (1996), A&A, 311, 291

        - Fe, amorphous
            - Pollack et al. (1994), ApJ, 421, 615

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'dust' in h5_file:
            del h5_file['dust']

        h5_file.create_group('dust')

        dust.add_optical_constants(self.input_path, h5_file)

        h5_file.close()

    @typechecked
    def add_filter(self,
                   filter_name: str,
                   filename: Optional[str] = None) -> None:
        """
        Function for adding a filter profile to the database, either from the SVO Filter profile
        Service or input file.

        Parameters
        ----------
        filter_name : str
            Filter name from the SVO Filter Profile Service (e.g., 'Paranal/NACO.Lp').
        filename : str
            Filename of the filter profile. The first column should contain the wavelength
            (um) and the second column the transmission. The profile is downloaded from the SVO
            Filter Profile Service if set to None.

        Returns
        -------
        NoneType
            None
        """

        print(f'Adding filter: {filter_name}...', end='', flush=True)

        filter_split = filter_name.split('/')

        h5_file = h5py.File(self.database, 'a')

        if 'filters' not in h5_file:
            h5_file.create_group('filters')

        if f'filters/{filter_split[0]}' not in h5_file:
            h5_file.create_group(f'filters/{filter_split[0]}')

        if f'filters/{filter_name}' in h5_file:
            del h5_file[f'filters/{filter_name}']

        if filename is not None:
            data = np.loadtxt(filename)
            wavelength = data[:, 0]
            transmission = data[:, 1]

        else:
            wavelength, transmission = filters.download_filter(filter_name)

        if wavelength is not None and transmission is not None:
            h5_file.create_dataset(f'filters/{filter_name}',
                                   data=np.column_stack((wavelength, transmission)))

        h5_file.close()

        print(' [DONE]')

    def add_isochrones(self,
                       filename,
                       tag,
                       model='baraffe'):
        """
        Function for adding isochrones data to the database.

        Parameters
        ----------
        filename : str
            Filename with the isochrones data.
        tag : str
            Database tag name where the isochrone that will be stored.
        model : str
            Evolutionary model ('baraffe' or 'marleau'). For 'baraffe' models, the isochrone data
            can be downloaded from https://phoenix.ens-lyon.fr/Grids/. For 'marleau' models, the
            data can be requested from Gabriel Marleau.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'isochrones' not in h5_file:
            h5_file.create_group('isochrones')

        if 'isochrones/'+tag in h5_file:
            del h5_file[f'isochrones/{tag}']

        if model[0:7] == 'baraffe':
            isochrones.add_baraffe(h5_file, tag, filename)

        elif model[0:7] == 'marleau':
            isochrones.add_marleau(h5_file, tag, filename)

        h5_file.close()

    def add_model(self,
                  model,
                  wavel_range=None,
                  spec_res=None,
                  teff_range=None,
                  data_folder=None):
        """
        Parameters
        ----------
        model : str
            Model name ('ames-cond', 'ames-dusty', 'bt-settl', 'bt-nextgen', 'drift-phoenix',
            'petitcode-cool-clear', 'petitcode-cool-cloudy', 'petitcode-hot-clear',
            'petitcode-hot-cloudy', or 'exo-rem').
        wavel_range : tuple(float, float), None
            Wavelength range (um). Optional for the DRIFT-PHOENIX and petitCODE models. For
            these models, the original wavelength points are used if set to None.
            which case the argument can be set to None.
        spec_res : float, None
            Spectral resolution. Optional for the DRIFT-PHOENIX and petitCODE models, in which
            case the argument is only used if ``wavel_range`` is not None.
        teff_range : tuple(float, float), None
            Effective temperature range (K). Setting the value to None for will add all available
            temperatures.
        data_folder : str, None
            Folder with input data. Only required for the petitCODE hot models which are not
            publicly available.

        Returns
        -------
        NoneType
            None
        """

        proprietary = ['petitcode-hot-clear', 'petitcode-hot-cloudy']

        if model in proprietary and data_folder is None:
            raise ValueError(f'The {model} model is not publicly available and needs to '
                             f'be imported by setting the \'data_folder\' parameter.')

        if model in ['ames-cond', 'ames-dusty', 'bt-sett', 'bt-nextgen'] and wavel_range is None:
            raise ValueError('The \'wavel_range\' should be set for the \'{model}\' models to '
                             'resample the original spectra on a fixed wavelength grid.')

        if model in ['ames-cond', 'ames-dusty', 'bt-sett', 'bt-nextgen'] and spec_res is None:
            raise ValueError('The \'spec_res\' should be set for the \'{model}\' models to '
                             'resample the original spectra on a fixed wavelength grid.')

        if model in ['bt-settl', 'bt-nextgen'] and teff_range is None:
            warnings.warn('The temperature range is not restricted with the \'teff_range\' '
                          'parameter. Therefore, adding the BT-Settl or BT-NextGen spectra '
                          'will be very slow.')

        h5_file = h5py.File(self.database, 'a')

        if 'models' not in h5_file:
            h5_file.create_group('models')

        if model == 'ames-cond':
            ames_cond.add_ames_cond(self.input_path,
                                    h5_file,
                                    wavel_range,
                                    teff_range,
                                    spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'ames-dusty':
            ames_dusty.add_ames_dusty(self.input_path,
                                      h5_file,
                                      wavel_range,
                                      teff_range,
                                      spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'bt-settl':
            btsettl.add_btsettl(self.input_path,
                                h5_file,
                                wavel_range,
                                teff_range,
                                spec_res)

            data_util.add_missing(model, ['teff', 'logg'], h5_file)

        elif model == 'bt-nextgen':
            btnextgen.add_btnextgen(self.input_path,
                                    h5_file,
                                    wavel_range,
                                    teff_range,
                                    spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh'], h5_file)

        elif model == 'drift-phoenix':
            drift_phoenix.add_drift_phoenix(self.input_path,
                                            h5_file,
                                            wavel_range,
                                            teff_range,
                                            spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh'], h5_file)

        elif model == 'petitcode-cool-clear':
            petitcode.add_petitcode_cool_clear(self.input_path,
                                               h5_file,
                                               wavel_range,
                                               teff_range,
                                               spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh'], h5_file)

        elif model == 'petitcode-cool-cloudy':
            petitcode.add_petitcode_cool_cloudy(self.input_path,
                                                h5_file,
                                                wavel_range,
                                                teff_range,
                                                spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'fsed'], h5_file)

        elif model == 'petitcode-hot-clear':
            petitcode.add_petitcode_hot_clear(self.input_path,
                                              h5_file,
                                              data_folder,
                                              wavel_range,
                                              teff_range,
                                              spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'co'], h5_file)

        elif model == 'petitcode-hot-cloudy':
            petitcode.add_petitcode_hot_cloudy(self.input_path,
                                               h5_file,
                                               data_folder,
                                               wavel_range,
                                               teff_range,
                                               spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'co', 'fsed'], h5_file)

        elif model == 'exo-rem':
            exo_rem.add_exo_rem(self.input_path,
                                h5_file,
                                wavel_range,
                                teff_range,
                                spec_res)

            data_util.add_missing(model, ['teff', 'logg', 'feh', 'co'], h5_file)

        else:
            raise ValueError(f'The {model} atmospheric model is not available. Please choose from '
                             f'\'ames-cond\', \'ames-dusty\', \'bt-settl\', \'bt-nextgen\', '
                             f'\'drift-phoexnix\', \'petitcode-cool-clear\', '
                             f'\'petitcode-cool-cloudy\', \'petitcode-hot-clear\', '
                             f'\'petitcode-hot-cloudy\', \'exo-rem\'.')

        h5_file.close()

    @typechecked
    def add_object(self,
                   object_name: str,
                   distance: Optional[Tuple[float, float]] = None,
                   app_mag: Optional[Dict[str,
                                          Union[Tuple[float, float],
                                                List[Tuple[float, float]]]]] = None,
                   spectrum: Optional[Dict[str,
                                           Tuple[str,
                                                 Optional[str],
                                                 Optional[float]]]] = None) -> None:
        """
        Function for adding the photometric and/or spectroscopic data of an object to the database.

        Parameters
        ----------
        object_name: str
            Object name that will be used as label in the database.
        distance : tuple(float, float), None
            Distance and uncertainty (pc). Not stored if set to None.
        app_mag : dict, None
            Dictionary with the filter names, apparent magnitudes, and uncertainties. For example,
            ``{'Paranal/NACO.Lp': (15., 0.2), 'Paranal/NACO.Mp': (13., 0.3)}``. For the use of
            duplicate filter names, the magnitudes have to be provided in a list, for example
            ``{'Paranal/NACO.Lp': [(15., 0.2), (14.5, 0.5)], 'Paranal/NACO.Mp': (13., 0.3)}``.
            No photometric data is stored if set to ``None``.
        spectrum : dict, None
            Dictionary with the spectrum, optional covariance matrix, and spectral resolution for
            each instrument. The input data can either have a FITS or ASCII format. The spectra
            should have 3 columns with wavelength (um), flux (W m-2 um-1), and uncertainty
            (W m-2 um-1). The covariance matrix should be 2D with the same number of wavelength
            points as the spectrum. For example, ``{'SPHERE': ('spectrum.dat', 'covariance.fits',
            50.)}``. No covariance data is stored if set to None, for example, ``{'SPHERE':
            ('spectrum.dat', None, 50.)}``. The ``spectrum`` parameter is ignored if set to None.
            For GRAVITY data, the same FITS file can be provided as spectrum and covariance matrix.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if app_mag is not None:
            if 'spectra/calibration/vega' not in h5_file:
                self.add_spectrum('vega')

            for item in app_mag:
                if f'filters/{item}' not in h5_file:
                    self.add_filter(item)

        print(f'Adding object: {object_name}')

        if 'objects' not in h5_file:
            h5_file.create_group('objects')

        if f'objects/{object_name}' not in h5_file:
            h5_file.create_group(f'objects/{object_name}')

        if distance is not None:
            print(f'   - Distance (pc) = {distance[0]:.2f} +/- {distance[1]:.2f}')

            if f'objects/{object_name}/distance' in h5_file:
                del h5_file[f'objects/{object_name}/distance']

            h5_file.create_dataset(f'objects/{object_name}/distance',
                                   data=distance)  # (pc)

        flux = {}
        error = {}

        if app_mag is not None:
            for mag_item in app_mag:
                if isinstance(app_mag[mag_item], tuple):

                    try:
                        synphot = photometry.SyntheticPhotometry(mag_item)
                        flux[mag_item], error[mag_item] = synphot.magnitude_to_flux(
                            app_mag[mag_item][0], app_mag[mag_item][1])

                    except KeyError:
                        warnings.warn(f'Filter \'{mag_item}\' is not available on the SVO Filter '
                                      f'Profile Service so a flux calibration can not be done. '
                                      f'Please add the filter manually with the \'add_filter\' '
                                      f'function. For now, only the \'{mag_item}\' magnitude of '
                                      f'\'{object_name}\' is stored.')

                        # Write NaNs if the filter is not available
                        flux[mag_item], error[mag_item] = np.nan, np.nan

                elif isinstance(app_mag[mag_item], list):
                    flux_list = []
                    error_list = []

                    for i, dupl_item in enumerate(app_mag[mag_item]):

                        try:
                            synphot = photometry.SyntheticPhotometry(mag_item)
                            flux_dupl, error_dupl = synphot.magnitude_to_flux(
                                dupl_item[0], dupl_item[1])

                        except KeyError:
                            warnings.warn(f'Filter \'{mag_item}\' is not available on the SVO '
                                          f'Filter Profile Service so a flux calibration can not '
                                          f'be done. Please add the filter manually with the '
                                          f'\'add_filter\' function. For now, only the '
                                          f'\'{mag_item}\' magnitude of \'{object_name}\' is '
                                          f'stored.')

                            # Write NaNs if the filter is not available
                            flux_dupl, error_dupl = np.nan, np.nan

                        flux_list.append(flux_dupl)
                        error_list.append(error_dupl)

                    flux[mag_item] = flux_list
                    error[mag_item] = error_list

                else:
                    raise ValueError('The values in the dictionary with magnitudes should be '
                                     'tuples or a list with tuples (in case duplicate filter '
                                     'names are required).')

            for mag_item in app_mag:
                if f'objects/{object_name}/{mag_item}' in h5_file:
                    del h5_file[f'objects/{object_name}/{mag_item}']

                if isinstance(app_mag[mag_item], tuple):
                    n_phot = 1

                    print(f'   - {mag_item}:')

                    print(f'      - Apparent magnitude = {app_mag[mag_item][0]:.2f} +/- '
                          f'{app_mag[mag_item][1]:.2f}')

                    print(f'      - Flux (W m-2 um-1) = {flux[mag_item]:.2e} +/- '
                          f'{error[mag_item]:.2e}')

                    data = np.asarray([app_mag[mag_item][0],
                                       app_mag[mag_item][1],
                                       flux[mag_item],
                                       error[mag_item]])

                elif isinstance(app_mag[mag_item], list):
                    n_phot = len(app_mag[mag_item])
                    print(f'   - {mag_item} ({n_phot} values):')

                    mag_list = []
                    mag_err_list = []

                    for i, dupl_item in enumerate(app_mag[mag_item]):
                        print(f'      - Apparent magnitude = {app_mag[mag_item][i][0]:.2f} +/- '
                              f'{app_mag[mag_item][i][1]:.2f}')

                        print(f'      - Flux (W m-2 um-1) = {flux[mag_item][i]:.2e} +/- '
                              f'{error[mag_item][i]:.2e}')

                        mag_list.append(app_mag[mag_item][i][0])
                        mag_err_list.append(app_mag[mag_item][i][1])

                    data = np.asarray([mag_list,
                                       mag_err_list,
                                       flux[mag_item],
                                       error[mag_item]])

                # (mag), (mag), (W m-2 um-1), (W m-2 um-1)
                dset = h5_file.create_dataset(f'objects/{object_name}/{mag_item}',
                                              data=data)

                dset.attrs['n_phot'] = n_phot

        if spectrum is not None:
            read_spec = {}
            read_cov = {}

            if f'objects/{object_name}/spectrum' in h5_file:
                del h5_file[f'objects/{object_name}/spectrum']

            # Read spectra

            for key, value in spectrum.items():
                if value[0].endswith('.fits'):
                    with fits.open(value[0]) as hdulist:
                        if 'INSTRU' in hdulist[0].header and \
                                hdulist[0].header['INSTRU'] == 'GRAVITY':
                            # Read data from a FITS file with the GRAVITY format
                            print('   - GRAVITY spectrum:')

                            gravity_object = hdulist[0].header['OBJECT']
                            print(f'      - Object: {gravity_object}')

                            wavelength = hdulist[1].data['WAVELENGTH']  # (um)
                            flux = hdulist[1].data['FLUX']  # (W m-2 um-1)
                            covariance = hdulist[1].data['COVARIANCE']  # (W m-2 um-1)^2
                            error = np.sqrt(np.diag(covariance))  # (W m-2 um-1)

                            read_spec[key] = np.column_stack([wavelength, flux, error])

                        else:
                            # Otherwise try to read a 2D dataset with 3 columns
                            print('   - Spectrum:')

                            for i, hdu_item in enumerate(hdulist):
                                data = np.asarray(hdu_item.data)

                                if data.ndim == 2 and 3 in data.shape and key not in read_spec:
                                    read_spec[key] = data

                            if key not in read_spec:
                                raise ValueError(f'The spectrum data from {value[0]} can not be '
                                                 f'read. The data format should be 2D with 3 '
                                                 f'columns.')

                else:
                    try:
                        data = np.loadtxt(value[0])
                    except UnicodeDecodeError:
                        raise ValueError(f'The spectrum data from {value[0]} can not be read. '
                                         f'Please provide a FITS or ASCII file.')

                    if data.ndim != 2 or 3 not in data.shape:
                        raise ValueError(f'The spectrum data from {value[0]} can not be read. The '
                                         f'data format should be 2D with 3 columns.')

                    print('   - Spectrum:')
                    read_spec[key] = data

                wavelength = read_spec[key][:, 0]
                flux = read_spec[key][:, 1]
                error = read_spec[key][:, 2]

                print(f'      - Database tag: {key}')
                print(f'      - Filename: {value[0]}')
                print(f'      - Data shape: {read_spec[key].shape}')
                print(f'      - Wavelength range (um): {wavelength[0]:.2f} - {wavelength[-1]:.2f}')
                print(f'      - Mean flux (W m-2 um-1): {np.mean(flux):.2e}')
                print(f'      - Mean error (W m-2 um-1): {np.mean(error):.2e}')

            # Read covariance matrix

            for key, value in spectrum.items():
                if value[1] is None:
                    read_cov[key] = None

                elif value[1].endswith('.fits'):
                    with fits.open(value[1]) as hdulist:
                        if 'INSTRU' in hdulist[0].header and \
                                hdulist[0].header['INSTRU'] == 'GRAVITY':
                            # Read data from a FITS file with the GRAVITY format
                            print('   - GRAVITY covariance matrix:')

                            gravity_object = hdulist[0].header['OBJECT']
                            print(f'      - Object: {gravity_object}')

                            read_cov[key] = hdulist[1].data['COVARIANCE']  # (W m-2 um-1)^2

                        else:
                            # Otherwise try to read a square, 2D dataset
                            print('   - Covariance matrix:')

                            for i, hdu_item in enumerate(hdulist):
                                data = np.asarray(hdu_item.data)

                                corr_warn = f'The covariance matrix from {value[1]} contains ' \
                                            f'ones along the diagonal. Converting this ' \
                                            f'correlation  matrix into a covariance matrix.'

                                if data.ndim == 2 and data.shape[0] == data.shape[1]:
                                    if key not in read_cov:
                                        if data.shape[0] == read_spec[key].shape[0]:
                                            if np.all(np.diag(data) == 1.):
                                                warnings.warn(corr_warn)

                                                read_cov[key] = data_util.correlation_to_covariance(
                                                    data, read_spec[key][:, 2])

                                            else:
                                                read_cov[key] = data

                            if key not in read_cov:
                                raise ValueError(f'The covariance matrix from {value[1]} can not '
                                                 f'be read. The data format should be 2D with the '
                                                 f'same number of wavelength points as the '
                                                 f'spectrum.')

                else:
                    try:
                        data = np.loadtxt(value[1])
                    except UnicodeDecodeError:
                        raise ValueError(f'The covariance matrix from {value[1]} can not be read. '
                                         f'Please provide a FITS or ASCII file.')

                    if data.ndim != 2 or 3 not in data.shape:
                        raise ValueError(f'The covariance matrix from {value[1]} can not be read. '
                                         f'The data format should be 2D with the same number of '
                                         f'wavelength points as the spectrum.')

                    print('   - Covariance matrix:')

                    if np.all(np.diag(data) == 1.):
                        warnings.warn(f'The covariance matrix from {value[1]} contains ones on '
                                      f'the diagonal. Converting this correlation matrix into a '
                                      f'covariance matrix.')

                        read_cov[key] = data_util.correlation_to_covariance(
                            data, read_spec[key][:, 2])

                    else:
                        read_cov[key] = data

                if read_cov[key] is not None:
                    print(f'      - Database tag: {key}')
                    print(f'      - Filename: {value[1]}')
                    print(f'      - Data shape: {read_cov[key].shape}')

            print('   - Spectral resolution:')

            for key, value in spectrum.items():

                h5_file.create_dataset(f'objects/{object_name}/spectrum/{key}/spectrum',
                                       data=read_spec[key])

                if read_cov[key] is not None:
                    h5_file.create_dataset(f'objects/{object_name}/spectrum/{key}/covariance',
                                           data=read_cov[key])

                    h5_file.create_dataset(f'objects/{object_name}/spectrum/{key}/inv_covariance',
                                           data=np.linalg.inv(read_cov[key]))

                dset = h5_file[f'objects/{object_name}/spectrum/{key}']

                if value[2] is None:
                    print(f'      - {key}: None')
                    dset.attrs['specres'] = 0.

                else:
                    print(f'      - {key}: {value[2]:.2f}')
                    dset.attrs['specres'] = value[2]

        h5_file.close()

    def add_photometry(self,
                       phot_library):
        """
        Parameters
        ----------
        phot_library : str
            Photometric library ('vlm-plx' or 'leggett').

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'photometry' not in h5_file:
            h5_file.create_group('photometry')

        if 'photometry/'+phot_library in h5_file:
            del h5_file['photometry/'+phot_library]

        if phot_library[0:7] == 'vlm-plx':
            vlm_plx.add_vlm_plx(self.input_path, h5_file)

        elif phot_library[0:7] == 'leggett':
            leggett.add_leggett(self.input_path, h5_file)

        h5_file.close()

    def add_calibration(self,
                        tag,
                        filename=None,
                        data=None,
                        units=None,
                        scaling=None):
        """
        Function for adding a calibration spectrum to the database.

        Parameters
        ----------
        tag : str
            Tag name in the database.
        filename : str, None
            Filename with the calibration spectrum. The first column should contain the wavelength
            (um), the second column the flux density (W m-2 um-1), and the third column
            the error (W m-2 um-1). The `data` argument is used if set to None.
        data : numpy.ndarray, None
            Spectrum stored as 3D array with shape (n_wavelength, 3). The first column should
            contain the wavelength (um), the second column the flux density (W m-2 um-1),
            and the third column the error (W m-2 um-1).
        units : dict, None
            Dictionary with the wavelength and flux units. Default (um and W m-2 um-1) is
            used if set to None.
        scaling : tuple(float, float)
            Scaling for the wavelength and flux as (scaling_wavelength, scaling_flux). Not used if
            set to None.

        Returns
        -------
        NoneType
            None
        """

        if filename is None and data is None:
            raise ValueError('Either the \'filename\' or \'data\' argument should be provided.')

        if scaling is None:
            scaling = (1., 1.)

        h5_file = h5py.File(self.database, 'a')

        if 'spectra/calibration' not in h5_file:
            h5_file.create_group('spectra/calibration')

        if 'spectra/calibration/'+tag in h5_file:
            del h5_file['spectra/calibration/'+tag]

        if filename is not None:
            data = np.loadtxt(filename)

        if units is None:
            wavelength = scaling[0]*data[:, 0]  # (um)
            flux = scaling[1]*data[:, 1]  # (W m-2 um-1)

        else:
            if units['wavelength'] == 'um':
                wavelength = scaling[0]*data[:, 0]  # (um)

            if units['flux'] == 'w m-2 um-1':
                flux = scaling[1]*data[:, 1]  # (W m-2 um-1)
            elif units['flux'] == 'w m-2':
                if units['wavelength'] == 'um':
                    flux = scaling[1]*data[:, 1]/wavelength  # (W m-2 um-1)

        if data.shape[1] == 3:
            if units is None:
                error = scaling[1]*data[:, 2]  # (W m-2 um-1)

            else:
                if units['flux'] == 'w m-2 um-1':
                    error = scaling[1]*data[:, 2]  # (W m-2 um-1)
                elif units['flux'] == 'w m-2':
                    if units['wavelength'] == 'um':
                        error = scaling[1]*data[:, 2]/wavelength  # (W m-2 um-1)

        else:
            error = np.repeat(0., wavelength.size)

        print(f'Adding calibration spectrum: {tag}...', end='', flush=True)

        h5_file.create_dataset(f'spectra/calibration/{tag}',
                               data=np.vstack((wavelength, flux, error)))

        h5_file.close()

        print(' [DONE]')

    def add_spectrum(self,
                     spec_library,
                     sptypes=None):
        """
        Parameters
        ----------
        spec_library : str
            Spectral library ('irtf' or 'spex').
        sptypes : list(str, )
            Spectral types ('F', 'G', 'K', 'M', 'L', 'T'). Currently only implemented for 'irtf'.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'spectra' not in h5_file:
            h5_file.create_group('spectra')

        if 'spectra/'+spec_library in h5_file:
            del h5_file['spectra/'+spec_library]

        if spec_library[0:5] == 'vega':
            vega.add_vega(self.input_path, h5_file)

        elif spec_library[0:5] == 'irtf':
            irtf.add_irtf(self.input_path, h5_file, sptypes)

        elif spec_library[0:5] == 'spex':
            spex.add_spex(self.input_path, h5_file)

        h5_file.close()

    @typechecked
    def add_samples(self,
                    sampler: str,
                    samples: np.ndarray,
                    ln_prob: np.ndarray,
                    mean_accept: Optional[float],
                    spectrum: Tuple[str, str],
                    tag: str,
                    modelpar: List[str],
                    distance: float,
                    spec_labels: List[Optional[str]]):
        """
        Parameters
        ----------
        sampler : str
            Sampler ('emcee' or 'multinest').
        samples : np.ndarray
            Samples of the posterior.
        ln_prob : np.ndarray
            Log posterior for each sample.
        mean_accept : float, None
            Mean acceptance fraction. Set to ``None`` when ``sampler`` is 'mulitinest'.
        spectrum : tuple(str, str)
            Tuple with the spectrum type ('model' or 'calibration') and spectrum name (e.g.
            'drift-phoenix').
        tag : str
            Database tag.
        modelpar : list(str, )
            List with the model parameter names.
        distance : float
            Distance to the object (pc). Not used if set to ``None``.
        spec_labels : list(str, )
            List with the spectrum labels that are used for fitting an additional scaling
            parameter. Not used if an empty list is provided.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'results' not in h5_file:
            h5_file.create_group('results')

        if 'results/fit' not in h5_file:
            h5_file.create_group('results/fit')

        if f'results/fit/{tag}' in h5_file:
            del h5_file[f'results/fit/{tag}']

        dset = h5_file.create_dataset(f'results/fit/{tag}/samples', data=samples)
        h5_file.create_dataset(f'results/fit/{tag}/ln_prob', data=ln_prob)

        dset.attrs['type'] = str(spectrum[0])
        dset.attrs['spectrum'] = str(spectrum[1])
        dset.attrs['n_param'] = int(len(modelpar))
        dset.attrs['sampler'] = str(sampler)

        if mean_accept is not None:
            dset.attrs['mean_accept'] = float(mean_accept)
            print(f'Mean acceptance fraction: {mean_accept:.3f}')

        if distance is not None:
            dset.attrs['distance'] = float(distance)

        count_scaling = 0

        for i, item in enumerate(modelpar):
            dset.attrs[f'parameter{i}'] = str(item)

            if item in spec_labels:
                dset.attrs[f'scaling{count_scaling}'] = str(item)
                count_scaling += 1

        dset.attrs['n_scaling'] = int(count_scaling)

        try:
            int_auto = emcee.autocorr.integrated_time(samples)
            print(f'Integrated autocorrelation time = {int_auto}')

        except emcee.autocorr.AutocorrError:
            int_auto = None

            print('The chain is shorter than 50 times the integrated autocorrelation time. '
                  '[WARNING]')

        if int_auto is not None:
            for i, item in enumerate(int_auto):
                dset.attrs[f'autocorrelation{i}'] = float(item)

        h5_file.close()

    def get_probable_sample(self,
                            tag,
                            burnin):
        """
        Function for extracting the sample parameters with the highest posterior probability.

        Parameters
        ----------
        tag : str
            Database tag with the MCMC results.
        burnin : int
            Number of burnin steps.

        Returns
        -------
        dict
            Parameters and values for the sample with the maximum posterior probability.
        """

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']

        samples = np.asarray(dset)
        ln_prob = np.asarray(h5_file[f'results/fit/{tag}/ln_prob'])

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        if samples.ndim == 3:
            if burnin > samples.shape[1]:
                raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                 f'({samples.shape[1]}) that are made by the walkers.')

            samples = samples[:, burnin:, :]
            ln_prob = ln_prob[:, burnin:]

            samples = np.reshape(samples, (-1, n_param))
            ln_prob = np.reshape(ln_prob, -1)

        index_max = np.unravel_index(ln_prob.argmax(), ln_prob.shape)

        # max_prob = ln_prob[index_max]
        max_sample = samples[index_max]

        prob_sample = {}

        for i in range(n_param):
            par_key = dset.attrs[f'parameter{i}']
            par_value = max_sample[i]

            prob_sample[par_key] = par_value

        if dset.attrs.__contains__('distance'):
            prob_sample['distance'] = dset.attrs['distance']

        h5_file.close()

        return prob_sample

    def get_median_sample(self,
                          tag,
                          burnin=None):
        """
        Function for extracting the median parameter values from the MCMC samples.

        Parameters
        ----------
        tag : str
            Database tag with the MCMC results.
        burnin : int, None
            Number of burnin steps. No burnin is removed if set to None.

        Returns
        -------
        dict
            Parameters and values for the sample with the maximum posterior probability.
        """

        if burnin is None:
            burnin = 0

        with h5py.File(self.database, 'r') as h5_file:
            dset = h5_file[f'results/fit/{tag}/samples']

            if 'n_param' in dset.attrs:
                n_param = dset.attrs['n_param']
            elif 'nparam' in dset.attrs:
                n_param = dset.attrs['nparam']

            samples = np.asarray(dset)

            if samples.ndim == 3:
                if burnin > samples.shape[1]:
                    raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                     f'({samples.shape[1]}) that are made by the walkers.')

                if burnin is not None:
                    samples = samples[:, burnin:, :]

                samples = np.reshape(samples, (-1, n_param))

            median_sample = {}

            for i in range(n_param):
                par_key = dset.attrs[f'parameter{i}']
                par_value = np.percentile(samples[:, i], 50.)
                median_sample[par_key] = par_value

            if dset.attrs.__contains__('distance'):
                median_sample['distance'] = dset.attrs['distance']

        return median_sample

    def get_mcmc_spectra(self,
                         tag,
                         burnin,
                         random,
                         wavel_range,
                         spec_res=None):
        """
        Parameters
        ----------
        tag : str
            Database tag with the MCMC samples.
        burnin : int
            Number of burnin steps.
        random : int
            Number of random samples.
        wavel_range : tuple(float, float), str, None
            Wavelength range (um) or filter name. Full spectrum if set to None.
        spec_res : float
            Spectral resolution that is used for the smoothing with a Gaussian kernel. No smoothing
            is applied if set to None.

        Returns
        -------
        list(species.core.box.ModelBox, )
            Boxes with the randomly sampled spectra.
        """

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']

        spectrum_type = dset.attrs['type']
        spectrum_name = dset.attrs['spectrum']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        if 'n_scaling' in dset.attrs:
            n_scaling = dset.attrs['n_scaling']
        elif 'nscaling' in dset.attrs:
            n_scaling = dset.attrs['nscaling']
        else:
            n_scaling = 0

        if 'n_error' in dset.attrs:
            n_error = dset.attrs['n_error']
        else:
            n_error = 0

        scaling = []
        for i in range(n_scaling):
            scaling.append(dset.attrs[f'scaling{i}'])

        error = []
        for i in range(n_error):
            error.append(dset.attrs[f'error{i}'])

        if spec_res is not None and spectrum_type == 'calibration':
            warnings.warn('Smoothing of the spectral resolution is not implemented for calibration '
                          'spectra.')

        if dset.attrs.__contains__('distance'):
            distance = dset.attrs['distance']
        else:
            distance = None

        samples = np.asarray(dset)

        if samples.ndim == 2:
            ran_index = np.random.randint(samples.shape[0], size=random)
            samples = samples[ran_index, ]

        elif samples.ndim == 3:
            if burnin > samples.shape[1]:
                raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                 f'({samples.shape[1]}) that are made by the walkers.')

            samples = samples[:, burnin:, :]

            ran_walker = np.random.randint(samples.shape[0], size=random)
            ran_step = np.random.randint(samples.shape[1], size=random)
            samples = samples[ran_walker, ran_step, :]

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        if spectrum_type == 'model':
            if spectrum_name == 'planck':
                readmodel = read_planck.ReadPlanck(wavel_range)
            else:
                readmodel = read_model.ReadModel(spectrum_name, wavel_range=wavel_range)

        elif spectrum_type == 'calibration':
            readcalib = read_calibration.ReadCalibration(spectrum_name, filter_name=None)

        boxes = []

        for i in tqdm.tqdm(range(samples.shape[0]), desc='Getting MCMC spectra'):
            model_param = {}
            for j in range(samples.shape[1]):
                if param[j] not in scaling and param[j] not in error:
                    model_param[param[j]] = samples[i, j]

            if distance:
                model_param['distance'] = distance

            if spectrum_type == 'model':
                if spectrum_name == 'planck':
                    specbox = readmodel.get_spectrum(model_param, spec_res)
                else:
                    specbox = readmodel.get_model(model_param, spec_res=spec_res, smooth=True)

            elif spectrum_type == 'calibration':
                specbox = readcalib.get_spectrum(model_param)

            boxes.append(specbox)

        h5_file.close()

        return boxes

    def get_mcmc_photometry(self,
                            tag,
                            burnin,
                            filter_name):
        """
        Parameters
        ----------
        tag : str
            Database tag with the MCMC samples.
        burnin : int
            Number of burnin steps.
        filter_name : str
            Filter name for which the photometry is calculated.

        Returns
        -------
        numpy.ndarray
            Synthetic photometry (mag).
        """

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        spectrum_type = dset.attrs['type'].decode('utf-8')
        spectrum_name = dset.attrs['spectrum'].decode('utf-8')

        if dset.attrs.__contains__('distance'):
            distance = dset.attrs['distance']
        else:
            distance = None

        samples = np.asarray(dset)
        samples = samples[:, burnin:, :]
        samples = samples.reshape((samples.shape[0]*samples.shape[1], n_param))

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        h5_file.close()

        if spectrum_type == 'model':
            readmodel = read_model.ReadModel(spectrum_name, filter_name=filter_name)
        # elif spectrum_type == 'calibration':
        #     readcalib = read_calibration.ReadCalibration(spectrum_name, None)

        mcmc_phot = np.zeros((samples.shape[0], 1))

        for i in tqdm.tqdm(range(samples.shape[0]), desc='Getting MCMC photometry'):
            model_param = {}
            for j in range(n_param):
                model_param[param[j]] = samples[i, j]

            if distance is not None:
                model_param['distance'] = distance

            if spectrum_type == 'model':
                mcmc_phot[i, 0], _ = readmodel.get_magnitude(model_param)
            # elif spectrum_type == 'calibration':
            #     specbox = readcalib.get_spectrum(model_param)

        return mcmc_phot

    @typechecked
    def get_object(self,
                   object_name: str,
                   inc_phot: Union[bool, List[str]] = True,
                   inc_spec: Union[bool, List[str]] = True) -> box.ObjectBox:
        """
        Function for extracting the photometric and/or spectroscopic data of an object from the
        database. The spectroscopic data contains optionally the covariance matrix and its inverse.

        Parameters
        ----------
        object_name : str
            Object name in the database.
        inc_phot : bool, list(str)
            Include photometric data. If a boolean, either all (``True``) or none (``False``) of
            the data are selected. If a list, a subset of filter names (as stored in the database)
            can be provided.
        inc_spec : bool, list(str)
            Include spectroscopic data. If a boolean, either all (``True``) or none (``False``) of
            the data are selected. If a list, a subset of spectrum names (as stored in the database
            with :func:`~species.data.database.Database.add_object`) can be provided.

        Returns
        -------
        species.core.box.ObjectBox
            Box with the object's data.
        """

        print(f'Getting object: {object_name}...', end='', flush=True)

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'objects/{object_name}']

        distance = np.asarray(dset['distance'])

        if inc_phot:

            magnitude = {}
            flux = {}

            for observatory in dset.keys():
                if observatory not in ['distance', 'spectrum']:
                    for filter_name in dset[observatory]:
                        name = f'{observatory}/{filter_name}'

                        if isinstance(inc_phot, bool) or name in inc_phot:
                            magnitude[name] = dset[name][0:2]
                            flux[name] = dset[name][2:4]

            filters = list(magnitude.keys())

        else:

            magnitude = None
            flux = None
            filters = None

        if inc_spec and f'objects/{object_name}/spectrum' in h5_file:
            spectrum = {}

            for item in h5_file[f'objects/{object_name}/spectrum']:
                data_group = f'objects/{object_name}/spectrum/{item}'

                if isinstance(inc_spec, bool) or item in inc_spec:

                    if f'{data_group}/covariance' not in h5_file:
                        spectrum[item] = (np.asarray(h5_file[f'{data_group}/spectrum']),
                                          None,
                                          None,
                                          h5_file[f'{data_group}'].attrs['specres'])

                    else:
                        spectrum[item] = (np.asarray(h5_file[f'{data_group}/spectrum']),
                                          np.asarray(h5_file[f'{data_group}/covariance']),
                                          np.asarray(h5_file[f'{data_group}/inv_covariance']),
                                          h5_file[f'{data_group}'].attrs['specres'])

        else:
            spectrum = None

        h5_file.close()

        print(' [DONE]')

        return box.create_box('object',
                              name=object_name,
                              filters=filters,
                              magnitude=magnitude,
                              flux=flux,
                              distance=distance,
                              spectrum=spectrum)

    def get_samples(self,
                    tag,
                    burnin=None,
                    random=None):
        """
        Parameters
        ----------
        tag: str
            Database tag with the samples.
        burnin : int, None
            Number of burnin samples to exclude. All samples are selected if set to None.
        random : int, None
            Number of random samples to select. All samples (with the burnin excluded) are
            selected if set to None.

        Returns
        -------
        species.core.box.SamplesBox
            Box with the MCMC samples.
        """

        if burnin is None:
            burnin = 0

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/fit/{tag}/samples']

        spectrum = dset.attrs['spectrum']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        samples = np.asarray(dset)

        if samples.ndim == 3:
            if burnin > samples.shape[1]:
                raise ValueError(f'The \'burnin\' value is larger than the number of steps '
                                 f'({samples.shape[1]}) that are made by the walkers.')

            samples = samples[:, burnin:, :]

            if random:
                ran_walker = np.random.randint(samples.shape[0], size=random)
                ran_step = np.random.randint(samples.shape[1], size=random)
                samples = samples[ran_walker, ran_step, :]

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        h5_file.close()

        if samples.ndim == 3:
            prob_sample = self.get_probable_sample(tag, burnin)
        else:
            prob_sample = None

        median_sample = self.get_median_sample(tag, burnin)

        return box.create_box('samples',
                              spectrum=spectrum,
                              parameters=param,
                              samples=samples,
                              prob_sample=prob_sample,
                              median_sample=median_sample)

    # def add_retrieval(self,
    #                   tag,
    #                   output_name):
    #     """
    #     Parameters
    #     ----------
    #     tag : str
    #         Database tag.
    #     output_name : str
    #         Output name that was used for the output files by MultiNest.
    #
    #     Returns
    #     -------
    #     NoneType
    #         None
    #     """
    #
    #     print('Storing samples in the database...', end='', flush=True)
    #
    #     with open(f'{output_name}_params.json') as json_file:
    #         parameters = json.load(json_file)
    #
    #     with open(f'{output_name}_radtrans.json') as json_file:
    #         radtrans = json.load(json_file)
    #
    #     samples = np.loadtxt(f'{output_name}_post_equal_weights.dat')
    #
    #     with h5py.File(self.database, 'a') as h5_file:
    #
    #         if 'results' not in h5_file:
    #             h5_file.create_group('results')
    #
    #         if 'results/fit' not in h5_file:
    #             h5_file.create_group('results/fit')
    #
    #         if f'results/fit/{tag}' in h5_file:
    #             del h5_file[f'results/fit/{tag}']
    #
    #         # remove the column with the log-likelihood value
    #         samples = samples[:, :-1]
    #
    #         if samples.shape[1] != len(parameters):
    #             raise ValueError('The number of parameters is not equal to the parameter size '
    #                              'of the samples array.')
    #
    #         dset = h5_file.create_dataset(f'results/fit/{tag}/samples', data=samples)
    #
    #         dset.attrs['type'] = 'model'
    #         dset.attrs['spectrum'] = 'petitradtrans'
    #         dset.attrs['n_param'] = len(parameters)
    #         dset.attrs['distance'] = radtrans['distance']
    #
    #         count_scale = 0
    #         count_error = 0
    #
    #         for i, item in enumerate(parameters):
    #             dset.attrs[f'parameter{i}'] = item
    #
    #         for i, item in enumerate(parameters):
    #             if item[0:6] == 'scaling_':
    #                 dset.attrs[f'scaling{count_scale}'] = item
    #                 count_scale += 1
    #
    #         for i, item in enumerate(parameters):
    #             if item[0:6] == 'error_':
    #                 dset.attrs[f'error{count_error}'] = item
    #                 count_error += 1
    #
    #         dset.attrs['n_scaling'] = count_scale
    #         dset.attrs['n_error'] = count_error
    #
    #         for i, item in enumerate(radtrans['line_species']):
    #             dset.attrs[f'line_species{i}'] = item
    #
    #         for i, item in enumerate(radtrans['cloud_species']):
    #             dset.attrs[f'cloud_species{i}'] = item
    #
    #         dset.attrs['n_line_species'] = len(radtrans['line_species'])
    #         dset.attrs['n_cloud_species'] = len(radtrans['cloud_species'])
    #
    #         dset.attrs['scattering'] = radtrans['scattering']
    #         dset.attrs['quenching'] = radtrans['quenching']
    #         dset.attrs['pt_profile'] = radtrans['pt_profile']
    #
    #     print(' [DONE]')
    #
    # def get_retrieval_spectra(self,
    #                           tag,
    #                           random,
    #                           wavel_range,
    #                           spec_res=None):
    #     """
    #     Parameters
    #     ----------
    #     tag : str
    #         Database tag with the MCMC samples.
    #     random : int
    #         Number of randomly selected samples.
    #     wavel_range : tuple(float, float) or str
    #         Wavelength range (um) or filter name.
    #     spec_res : float
    #         Spectral resolution that is used for the smoothing with a Gaussian kernel. No smoothing
    #         is applied if set to None.
    #
    #     Returns
    #     -------
    #     list(species.core.box.ModelBox, )
    #         Boxes with the randomly sampled spectra.
    #     """
    #
    #     config_file = os.path.join(os.getcwd(), 'species_config.ini')
    #
    #     config = configparser.ConfigParser()
    #     config.read_file(open(config_file))
    #
    #     database_path = config['species']['database']
    #
    #     h5_file = h5py.File(database_path, 'r')
    #     dset = h5_file[f'results/fit/{tag}/samples']
    #
    #     spectrum_type = dset.attrs['type']
    #     spectrum_name = dset.attrs['spectrum']
    #
    #     if 'n_param' in dset.attrs:
    #         n_param = dset.attrs['n_param']
    #     elif 'nparam' in dset.attrs:
    #         n_param = dset.attrs['nparam']
    #
    #     n_line_species = dset.attrs['n_line_species']
    #     n_cloud_species = dset.attrs['n_cloud_species']
    #
    #     scattering = dset.attrs['scattering']
    #     quenching = dset.attrs['quenching']
    #     pt_profile = dset.attrs['pt_profile']
    #
    #     if dset.attrs.__contains__('distance'):
    #         distance = dset.attrs['distance']
    #     else:
    #         distance = None
    #
    #     samples = np.asarray(dset)
    #
    #     random_indices = np.random.randint(samples.shape[0], size=random)
    #     samples = samples[random_indices, :]
    #
    #     parameters = []
    #     for i in range(n_param):
    #         parameters.append(dset.attrs[f'parameter{i}'])
    #
    #     parameters = np.asarray(parameters)
    #
    #     line_species = []
    #     for i in range(n_line_species):
    #         line_species.append(dset.attrs[f'line_species{i}'])
    #
    #     line_species = np.asarray(line_species)
    #
    #     cloud_species = []
    #     for i in range(n_cloud_species):
    #         cloud_species.append(dset.attrs[f'cloud_species{i}'])
    #
    #     cloud_species = np.asarray(cloud_species)
    #
    #     # create mock p-t profile
    #
    #     temp_params = {}
    #     temp_params['log_delta'] = -6.
    #     temp_params['log_gamma'] = 1.
    #     temp_params['t_int'] = 750.
    #     temp_params['t_equ'] = 0.
    #     temp_params['log_p_trans'] = -3.
    #     temp_params['alpha'] = 0.
    #
    #     pressure, _ = nc.make_press_temp(temp_params)
    #
    #     logg_index = np.argwhere(parameters == 'logg')[0]
    #     radius_index = np.argwhere(parameters == 'radius')[0]
    #     feh_index = np.argwhere(parameters == 'feh')[0]
    #     co_index = np.argwhere(parameters == 'co')[0]
    #
    #     if quenching:
    #         log_p_quench_index = np.argwhere(parameters == 'log_p_quench')[0]
    #
    #     if pt_profile == 'molliere':
    #         tint_index = np.argwhere(parameters == 'tint')[0]
    #         t1_index = np.argwhere(parameters == 't1')[0]
    #         t2_index = np.argwhere(parameters == 't2')[0]
    #         t3_index = np.argwhere(parameters == 't3')[0]
    #         alpha_index = np.argwhere(parameters == 'alpha')[0]
    #         log_delta_index = np.argwhere(parameters == 'log_delta')[0]
    #
    #     elif pt_profile == 'line':
    #         temp_index = []
    #         for i in range(15):
    #             temp_index.append(np.argwhere(parameters == f't{i}')[0])
    #
    #         knot_press = np.logspace(np.log10(pressure[0]), np.log10(pressure[-1]), 15)
    #
    #     if scattering:
    #         rt_object = RadtransScatter(line_species=line_species,
    #                                     rayleigh_species=['H2', 'He'],
    #                                     cloud_species=cloud_species,
    #                                     continuum_opacities=['H2-H2', 'H2-He'],
    #                                     wlen_bords_micron=wavel_range,
    #                                     mode='c-k',
    #                                     test_ck_shuffle_comp=scattering,
    #                                     do_scat_emis=scattering)
    #
    #     else:
    #         rt_object = Radtrans(line_species=line_species,
    #                              rayleigh_species=['H2', 'He'],
    #                              cloud_species=cloud_species,
    #                              continuum_opacities=['H2-H2', 'H2-He'],
    #                              wlen_bords_micron=wavel_range,
    #                              mode='c-k')
    #
    #     # create RT arrays of appropriate lengths by using every three pressure points
    #     rt_object.setup_opa_structure(pressure[::3])
    #
    #     boxes = []
    #
    #     for i, item in tqdm.tqdm(enumerate(samples), desc='Getting MCMC spectra'):
    #
    #         if pt_profile == 'molliere':
    #             temp, _, _ = retrieval_util.pt_ret_model(
    #                 np.array([item[t1_index][0], item[t2_index][0], item[t3_index][0]]),
    #                 10.**item[log_delta_index][0], item[alpha_index][0], item[tint_index][0], pressure,
    #                 item[feh_index][0], item[co_index][0])
    #
    #         elif pt_profile == 'line':
    #             knot_temp = []
    #             for i in range(15):
    #                 knot_temp.append(item[temp_index[i]][0])
    #
    #             temp = retrieval_util.pt_spline_interp(knot_press, knot_temp, pressure)
    #
    #         if quenching:
    #             log_p_quench = item[log_p_quench_index][0]
    #         else:
    #             log_p_quench = -10.
    #
    #         wavelength, flux = retrieval_util.calc_spectrum_clear(
    #             rt_object, pressure, temp, item[logg_index][0], item[co_index][0],
    #             item[feh_index][0], log_p_quench, half=True)
    #
    #         flux *= (item[radius_index]*constants.R_JUP/(distance*constants.PARSEC))**2.
    #
    #         if spec_res is not None:
    #             # convolve with a Gaussian line spread function
    #             flux = retrieval_util.convolve(wavelength, flux, spec_res)
    #
    #         model_box = box.create_box(boxtype='model',
    #                                    model='petitradtrans',
    #                                    wavelength=wavelength,
    #                                    flux=flux,
    #                                    parameters=None,
    #                                    quantity='flux')
    #
    #         boxes.append(model_box)
    #
    #     h5_file.close()
    #
    #     return boxes
