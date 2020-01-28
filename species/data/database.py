"""
Module with functionalities for reading and writing of data.
"""

import os
import warnings
import configparser

import h5py
import tqdm
import emcee
import numpy as np

from species.analysis import photometry
from species.core import box
from species.data import drift_phoenix, btnextgen, vega, irtf, spex, vlm_plx, leggett, \
                         companions, filters, btsettl, ames_dusty, ames_cond, \
                         isochrones, petitcode
from species.read import read_model, read_calibration, read_planck
from species.util import data_util


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
            print(f'Distance [pc] = {distance[0]} +/- {distance[1]}')

            for mag_name, mag_dict in app_mag.items():
                print(f'{mag_name} [mag] = {mag_dict[0]} +/- {mag_dict[1]}')

            print()

    def add_companion(self,
                      name=None):
        """
        Parameters
        ----------
        name : list(str, )
            Companion name. All companions are added if set to None.

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

    def add_filter(self,
                   filter_name,
                   filename=None):
        """
        Parameters
        ----------
        filter_name : str
            Filter name from the SVO Filter Profile Service (e.g., 'Paranal/NACO.Lp').
        filename : str
            Filename with the filter profile. The first column should contain the wavelength
            (micron) and the second column the transmission (no units). The profile is downloaded
            from the SVO Filter Profile Service if set to None.

        Returns
        -------
        NoneType
            None
        """

        filter_split = filter_name.split('/')

        h5_file = h5py.File(self.database, 'a')

        if 'filters' not in h5_file:
            h5_file.create_group('filters')

        if 'filters/'+filter_split[0] not in h5_file:
            h5_file.create_group('filters/'+filter_split[0])

        if 'filters/'+filter_name in h5_file:
            del h5_file['filters/'+filter_name]

        print(f'Adding filter: {filter_name}...', end='')

        if filename:
            data = np.loadtxt(filename)
            wavelength = data[:, 0]
            transmission = data[:, 1]

        else:
            wavelength, transmission = filters.download_filter(filter_name)

        h5_file.create_dataset('filters/'+filter_name,
                               data=np.vstack((wavelength, transmission)),
                               dtype='f')

        print(' [DONE]')

        h5_file.close()

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
            del h5_file['isochrones/'+tag]

        if model[0:7] == 'baraffe':
            isochrones.add_baraffe(h5_file, tag, filename)

        elif model[0:7] == 'marleau':
            isochrones.add_marleau(h5_file, tag, filename)

        h5_file.close()

    def add_model(self,
                  model,
                  wavel_range=None,
                  teff_range=None,
                  spec_res=1000.,
                  data_folder=None):
        """
        Parameters
        ----------
        model : str
            Model name ('ames-cond', 'ames-dusty', 'bt-settl', 'bt-nextgen', 'drift-phoenix',
            'petitcode-cool-clear', 'petitcode-cool-cloudy', 'petitcode-hot-clear', or
            'petitcode-hot-cloudy').
        wavel_range : tuple(float, float), None
            Wavelength range (micron).
        teff_range : tuple(float, float), None
            Effective temperature range (K).
        spec_res : float
            Spectral resolution.
        data_folder : str, None
            Folder with input data. Only required for the petitCODE hot models which are not
            publically available.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'models' not in h5_file:
            h5_file.create_group('models')

        if model[0:13] == 'drift-phoenix':
            drift_phoenix.add_drift_phoenix(self.input_path, h5_file)
            data_util.add_missing(model, ('teff', 'logg', 'feh'), h5_file)

        elif model[0:8] == 'bt-settl':
            btsettl.add_btsettl(self.input_path, h5_file, wavel_range, teff_range, spec_res)
            data_util.add_missing(model, ('teff', 'logg'), h5_file)

        elif model[0:10] == 'bt-nextgen':
            btnextgen.add_btnextgen(self.input_path, h5_file, wavel_range, teff_range, spec_res)
            data_util.add_missing(model, ('teff', 'logg', 'feh'), h5_file)

        elif model[0:10] == 'ames-dusty':
            ames_dusty.add_ames_dusty(self.input_path, h5_file, wavel_range, teff_range, spec_res)
            data_util.add_missing(model, ('teff', 'logg'), h5_file)

        elif model[0:9] == 'ames-cond':
            ames_cond.add_ames_cond(self.input_path, h5_file, wavel_range, teff_range, spec_res)
            data_util.add_missing(model, ('teff', 'logg'), h5_file)

        elif model[0:9] == 'ames-cond':
            ames_cond.add_ames_cond(self.input_path, h5_file, wavel_range, teff_range, spec_res)
            data_util.add_missing(model, ('teff', 'logg'), h5_file)

        elif model[0:20] == 'petitcode-cool-clear':
            petitcode.add_petitcode_cool_clear(self.input_path, h5_file, wavel_range, teff_range, spec_res)
            data_util.add_missing(model, ('teff', 'logg', 'feh'), h5_file)

        elif model[0:21] == 'petitcode-cool-cloudy':
            petitcode.add_petitcode_cool_cloudy(self.input_path, h5_file, wavel_range, teff_range, spec_res)
            data_util.add_missing(model, ('teff', 'logg', 'feh', 'fsed'), h5_file)

        elif model[0:19] == 'petitcode-hot-clear':
            petitcode.add_petitcode_hot_clear(self.input_path, h5_file, wavel_range, teff_range,
                                              spec_res, data_folder)
            data_util.add_missing(model, ('teff', 'logg', 'feh', 'co'), h5_file)

        elif model[0:20] == 'petitcode-hot-cloudy':
            petitcode.add_petitcode_hot_cloudy(self.input_path, h5_file, wavel_range, teff_range,
                                               spec_res, data_folder)
            data_util.add_missing(model, ('teff', 'logg', 'feh', 'co', 'fsed'), h5_file)

        h5_file.close()

    def add_object(self,
                   object_name,
                   distance=None,
                   app_mag=None,
                   spectrum=None,
                   instrument=None):
        """
        Parameters
        ----------
        object_name: str
            Object name.
        distance : tuple(float, float), None
            Distance and uncertainty (pc). Not written if set to None.
        app_mag : dict
            Apparent magnitudes. Not written if set to None.
        spectrum : str
            Spectrum filename. The first three columns should contain the wavelength (micron),
            flux density (W m-2 micron-1), and the error (W m-2 micron-1). Not written if set
            to None.
        instrument : str
            Instrument that was used for the spectrum (currently only 'gpi' possible). Not
            used if set to None.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'objects' not in h5_file:
            h5_file.create_group('objects')

        if 'objects/'+object_name not in h5_file:
            h5_file.create_group('objects/'+object_name)

        if distance is not None:
            if 'objects/'+object_name+'/distance' in h5_file:
                del h5_file['objects/'+object_name+'/distance']

            h5_file.create_dataset('objects/'+object_name+'/distance',
                                   data=distance,
                                   dtype='f')  # [pc]

        if app_mag is not None:
            flux = {}
            error = {}

            for item in app_mag:
                try:
                    synphot = photometry.SyntheticPhotometry(item)
                    flux[item], error[item] = synphot.magnitude_to_flux(app_mag[item][0],
                                                                        app_mag[item][1])

                except ValueError:
                    # Write NaNs if the filter is not available
                    flux[item], error[item] = np.nan, np.nan

            for item in app_mag:
                if 'objects/'+object_name+'/'+item in h5_file:
                    del h5_file['objects/'+object_name+'/'+item]

                data = np.asarray([app_mag[item][0],
                                   app_mag[item][1],
                                   flux[item],
                                   error[item]])

                # [mag], [mag], [W m-2 micron-1], [W m-2 micron-1]
                h5_file.create_dataset('objects/'+object_name+'/'+item,
                                       data=data,
                                       dtype='f')

        print(f'Adding object: {object_name}...', end='')

        if spectrum is not None:

            if 'objects/'+object_name+'/spectrum' in h5_file:
                del h5_file['objects/'+object_name+'/spectrum']

            data = np.loadtxt(spectrum)

            dset = h5_file.create_dataset('objects/'+object_name+'/spectrum',
                                          data=data[:, 0:3],
                                          dtype='f')

            dset.attrs['instrument'] = str(instrument)

        print(' [DONE]')

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
            (micron), the second column the flux density (W m-2 micron-1), and the third column
            the error (W m-2 micron-1). The `data` argument is used if set to None.
        data : numpy.ndarray, None
            Spectrum stored as 3D array with shape (n_wavelength, 3). The first column should
            contain the wavelength (micron), the second column the flux density (W m-2 micron-1),
            and the third column the error (W m-2 micron-1).
        units : dict, None
            Dictionary with the wavelength and flux units. Default (micron and W m-2 micron-1) is
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
            wavelength = scaling[0]*data[:, 0]  # [micron]
            flux = scaling[1]*data[:, 1]  # [W m-2 micron-1]

        else:
            if units['wavelength'] == 'micron':
                wavelength = scaling[0]*data[:, 0]  # [micron]

            if units['flux'] == 'w m-2 micron-1':
                flux = scaling[1]*data[:, 1]  # [W m-2 micron-1]
            elif units['flux'] == 'w m-2':
                if units['wavelength'] == 'micron':
                    flux = scaling[1]*data[:, 1]/wavelength  # [W m-2 micron-1]

        if data.shape[1] == 3:
            if units is None:
                error = scaling[1]*data[:, 2]  # [W m-2 micron-1]

            else:
                if units['flux'] == 'w m-2 micron-1':
                    error = scaling[1]*data[:, 2]  # [W m-2 micron-1]
                elif units['flux'] == 'w m-2':
                    if units['wavelength'] == 'micron':
                        error = scaling[1]*data[:, 2]/wavelength  # [W m-2 micron-1]

        else:
            error = np.repeat(0., wavelength.size)

        print(f'Adding calibration spectrum: {tag}...', end='')

        h5_file.create_dataset('spectra/calibration/'+tag,
                               data=np.vstack((wavelength, flux, error)),
                               dtype='f')

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

    def add_samples(self,
                    sampler,
                    spectrum,
                    tag,
                    modelpar,
                    distance=None):
        """
        Parameters
        ----------
        sampler : emcee.ensemble.EnsembleSampler
            Ensemble sampler.
        spectrum : tuple(str, str)
            Tuple with the spectrum type ('model' or 'calibration') and spectrum name (e.g.
            'drift-phoenix').
        tag : str
            Database tag.
        modelpar : list(str, )
            List with the model parameter names.
        distance : float
            Distance to the object (pc). Not used if set to None.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'results' not in h5_file:
            h5_file.create_group('results')

        if 'results/mcmc' not in h5_file:
            h5_file.create_group('results/mcmc')

        if 'results/mcmc/'+tag in h5_file:
            del h5_file['results/mcmc/'+tag]

        dset = h5_file.create_dataset('results/mcmc/'+tag+'/samples',
                                      data=sampler.chain,
                                      dtype='f')

        h5_file.create_dataset('results/mcmc/'+tag+'/probability',
                               data=np.exp(sampler.lnprobability),
                               dtype='f')

        dset.attrs['type'] = str(spectrum[0])
        dset.attrs['spectrum'] = str(spectrum[1])
        dset.attrs['nparam'] = int(len(modelpar))

        if distance:
            dset.attrs['distance'] = float(distance)

        for i, item in enumerate(modelpar):
            dset.attrs['parameter'+str(i)] = str(item)

        mean_accep = np.mean(sampler.acceptance_fraction)
        dset.attrs['acceptance'] = float(mean_accep)
        print(f'Mean acceptance fraction: {mean_accep:.3f}')

        try:
            int_auto = emcee.autocorr.integrated_time(sampler.flatchain)
            print(f'Integrated autocorrelation time = {int_auto}')

        except emcee.autocorr.AutocorrError:
            int_auto = None

            print('The chain is shorter than 50 times the integrated autocorrelation time. '
                  '[WARNING]')

        if int_auto is not None:
            for i, item in enumerate(int_auto):
                dset.attrs['autocorrelation'+str(i)] = float(item)

        h5_file.close()

    def get_probable_sample(self,
                            tag,
                            burnin):
        """
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

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag+'/samples']

        samples = np.asarray(dset)
        samples = samples[:, burnin:, :]

        probability = np.asarray(h5_file['results/mcmc/'+tag+'/probability'])
        probability = probability[:, burnin:]

        nparam = dset.attrs['nparam']

        index_max = np.unravel_index(probability.argmax(), probability.shape)

        # max_prob = probability[index_max]
        max_sample = samples[index_max]

        # print(f'Maximum probability: {max_prob:.2f}')
        # print(f'Most probable sample:', end='')

        prob_sample = {}

        for i in range(nparam):
            par_key = dset.attrs['parameter'+str(i)]
            par_value = max_sample[i]

            prob_sample[par_key] = par_value
            # print(f' {par_key}={par_value:.2f}', end=')

        if dset.attrs.__contains__('distance'):
            prob_sample['distance'] = dset.attrs['distance']

        # print()

        h5_file.close()

        return prob_sample

    def get_median_sample(self,
                          tag,
                          burnin):
        """
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

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag+'/samples']

        nparam = dset.attrs['nparam']

        samples = np.asarray(dset)
        samples = samples[:, burnin:, :]
        samples = np.reshape(samples, (-1, nparam))

        # print(f'Median sample:', end='')

        median_sample = {}

        for i in range(nparam):
            par_key = dset.attrs['parameter'+str(i)]
            par_value = np.percentile(samples[:, i], 50.)

            median_sample[par_key] = par_value
            # print(f' {par_key}={par_value:.2f}', end='')

        if dset.attrs.__contains__('distance'):
            median_sample['distance'] = dset.attrs['distance']

        # print()

        h5_file.close()

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
        wavel_range : tuple(float, float) or str
            Wavelength range (micron) or filter name. Full spectrum if set to None.
        spec_res : float
            Spectral resolution, achieved by smoothing with a Gaussian kernel. The original
            wavelength points are used if set to None.

        Returns
        -------
        list(species.core.box.ModelBox, )
            Boxes with the randomly sampled spectra.
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/mcmc/{tag}/samples']

        nparam = dset.attrs['nparam']
        spectrum_type = dset.attrs['type']
        spectrum_name = dset.attrs['spectrum']

        if spec_res is not None and spectrum_type == 'calibration':
            warnings.warn('Smoothing of the spectral resolution is not implemented for calibration '
                          'spectra.')

        if dset.attrs.__contains__('distance'):
            distance = dset.attrs['distance']
        else:
            distance = None

        samples = np.asarray(dset)
        samples = samples[:, burnin:, :]

        ran_walker = np.random.randint(samples.shape[0], size=random)
        ran_step = np.random.randint(samples.shape[1], size=random)
        samples = samples[ran_walker, ran_step, :]

        param = []
        for i in range(nparam):
            param.append(str(dset.attrs[f'parameter{i}']))

        if spectrum_type == 'model':
            if spectrum_name == 'planck':
                readmodel = read_planck.ReadPlanck(wavel_range)
            else:
                readmodel = read_model.ReadModel(spectrum_name, wavel_range)

        elif spectrum_type == 'calibration':
            readcalib = read_calibration.ReadCalibration(spectrum_name, filter_name=None)

        boxes = []

        for i in tqdm.tqdm(range(samples.shape[0]), desc='Getting MCMC spectra'):
            model_param = {}
            for j in range(samples.shape[1]):
                model_param[param[j]] = samples[i, j]

            if distance:
                model_param['distance'] = distance

            if spectrum_type == 'model':
                if spectrum_name == 'planck':
                    specbox = readmodel.get_spectrum(model_param, spec_res)
                else:
                    specbox = readmodel.get_model(model_param, spec_res)

            elif spectrum_type == 'calibration':
                specbox = readcalib.get_spectrum(model_param)

            box.type = 'mcmc'

            boxes.append(specbox)

        h5_file.close()

        return list(boxes)

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

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file[f'results/mcmc/{tag}/samples']

        nparam = dset.attrs['nparam']
        spectrum_type = dset.attrs['type']
        spectrum_name = dset.attrs['spectrum']

        if dset.attrs.__contains__('distance'):
            distance = dset.attrs['distance']
        else:
            distance = None

        samples = np.asarray(dset)
        samples = samples[:, burnin:, :]
        samples = samples.reshape((samples.shape[0]*samples.shape[1], nparam))

        param = []
        for i in range(nparam):
            param.append(str(dset.attrs['parameter'+str(i)]))

        h5_file.close()

        if spectrum_type == 'model':
            readmodel = read_model.ReadModel(spectrum_name, filter_name)
        # elif spectrum_type == 'calibration':
        #     readcalib = read_calibration.ReadCalibration(spectrum_name, None)

        mcmc_phot = np.zeros((samples.shape[0], 1))

        for i in tqdm.tqdm(range(samples.shape[0]), desc='Getting MCMC photometry'):
            model_param = {}
            for j in range(nparam):
                model_param[param[j]] = samples[i, j]

            if distance:
                model_param['distance'] = distance

            if spectrum_type == 'model':
                mcmc_phot[i, 0], _ = readmodel.get_magnitude(model_param)
            # elif spectrum_type == 'calibration':
            #     specbox = readcalib.get_spectrum(model_param)

        return mcmc_phot

    def get_object(self,
                   object_name,
                   filters=None,
                   inc_phot=True,
                   inc_spec=True):
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        filters : list(str, )
            Filter names for which the photometry is selected. All available photometry of the object
            is selected if set to None.
        inc_phot : bool
            Include photometry in the box.
        inc_spec : bool
            Include spectrum in the box.

        Returns
        -------
        species.core.box.ObjectBox
            Box with the object's data.
        """

        print(f'Getting object: {object_name}...', end='')

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['objects/'+object_name]

        distance = np.asarray(dset['distance'])[0]

        if inc_phot:

            magnitude = {}
            flux = {}

            if filters:
                for item in filters:
                    data = dset[item]

                    magnitude[item] = np.asarray(data[0:2])
                    flux[item] = np.asarray(data[2:4])

            else:
                for key in dset.keys():
                    if key not in ('distance', 'spectrum'):
                        for item in dset[key]:
                            name = key+'/'+item

                            magnitude[name] = np.asarray(dset[name][0:2])
                            flux[name] = np.asarray(dset[name][2:4])

            filters = list(magnitude.keys())

        else:

            magnitude = None
            flux = None
            filters = None

        if inc_spec and 'objects/'+object_name+'/spectrum' in h5_file:
            spectrum = np.asarray(h5_file['objects/'+object_name+'/spectrum'])
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
        dset = h5_file['results/mcmc/'+tag+'/samples']

        spectrum = dset.attrs['spectrum']
        nparam = dset.attrs['nparam']

        samples = np.asarray(dset)
        samples = samples[:, burnin:, :]

        if random:
            ran_walker = np.random.randint(samples.shape[0], size=random)
            ran_step = np.random.randint(samples.shape[1], size=random)
            samples = samples[ran_walker, ran_step, :]

        param = []
        for i in range(nparam):
            param.append(dset.attrs['parameter'+str(i)])

        h5_file.close()

        prob_sample = self.get_probable_sample(tag, burnin)
        median_sample = self.get_median_sample(tag, burnin)

        return box.create_box('samples',
                              spectrum=spectrum,
                              parameters=param,
                              samples=samples,
                              prob_sample=prob_sample,
                              median_sample=median_sample)
