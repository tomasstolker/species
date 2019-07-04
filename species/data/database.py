"""
Database module.
"""

import os
import sys
import math
import warnings
import configparser

import h5py
import emcee
import progress.bar
import numpy as np

from species.analysis import photometry
from species.core import box, constants
from species.data import drift_phoenix, btnextgen, vega, irtf, spex, vlm_plx, leggett, \
                         companions, filters, mamajek, btsettl, ames_dusty, ames_cond, \
                         isochrones
from species.read import read_model, read_calibration
from species.util import data_util


class Database:
    """
    Text.
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
        self.input_path = config['species']['input']

    def list_items(self):
        """
        Returns
        -------
        NoneType
            None
        """

        sys.stdout.write('Database content:\n')

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
                    sys.stdout.write(seperator+'- '+key+': '+str(h5_object[key])+'\n')
                    descend(h5_object[key], seperator=seperator+'\t')

            elif isinstance(h5_object, h5py._hl.dataset.Dataset):
                for key in h5_object.attrs.keys():
                    sys.stdout.write(seperator+'- '+key+': '+str(h5_object.attrs[key])+'\n')

        h5_file = h5py.File(self.database, 'r')
        descend(h5_file)
        h5_file.close()

        sys.stdout.flush()

    def list_companions(self):
        """
        Returns
        -------
        NoneType
            None
        """

        comp_phot = companions.get_data()

        sys.stdout.write('Database: '+self.database+'\n')
        sys.stdout.write('Directly imaged companions: ')
        sys.stdout.write(str(list(comp_phot.keys()))+'\n')
        sys.stdout.flush()

    def add_companion(self,
                      name=None):
        """
        Parameters
        ----------
        name : tuple(str, )
            Companion name. All companions are added if set to None.

        Returns
        -------
        NoneType
            None
        """

        if isinstance(name, str):
            name = tuple((name, ))

        data = companions.get_data()

        if name is None:
            name = data.keys()

        for item in name:
            self.add_object(object_name=item,
                            distance=data[item]['distance'],
                            app_mag=data[item]['app_mag'])

    def add_filter(self,
                   filter_id,
                   filename=None):
        """
        Parameters
        ----------
        filter_id : str
            Filter ID from the SVO Filter Profile Service (e.g., 'Paranal/NACO.Lp').
        filename : str
            Filename with the filter profile. The first column should contain the wavelength
            (micron) and the second column the transmission (no units). The profile is downloaded
            from the SVO Filter Profile Service if set to None.

        Returns
        -------
        NoneType
            None
        """

        filter_split = filter_id.split('/')

        h5_file = h5py.File(self.database, 'a')

        if 'filters' not in h5_file:
            h5_file.create_group('filters')

        if 'filters/'+filter_split[0] not in h5_file:
            h5_file.create_group('filters/'+filter_split[0])

        if 'filters/'+filter_id in h5_file:
            del h5_file['filters/'+filter_id]

        sys.stdout.write('Adding filter: '+filter_id+'...')
        sys.stdout.flush()

        if filename:
            data = np.loadtxt(filename)
            wavelength = data[:, 0]
            transmission = data[:, 1]

        else:
            wavelength, transmission = filters.download_filter(filter_id)

        h5_file.create_dataset('filters/'+filter_id,
                               data=np.vstack((wavelength, transmission)),
                               dtype='f')

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

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
            Tag name in the database.
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
                  wavelength=None,
                  teff=None,
                  specres=None):
        """
        Parameters
        ----------
        model : str
            Model name.
        wavelength : tuple(float, float)
            Wavelength (micron) range.
        teff : tuple(float, float), None
            Effective temperature (K) range.
        specres : float
            Spectral resolution.

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
            btsettl.add_btsettl(self.input_path, h5_file, wavelength, teff, specres)
            data_util.add_missing(model, ('teff', 'logg'), h5_file)

        elif model[0:10] == 'bt-nextgen':
            btnextgen.add_btnextgen(self.input_path, h5_file, wavelength, teff, specres)
            data_util.add_missing(model, ('teff', 'logg', 'feh'), h5_file)

        elif model[0:10] == 'ames-dusty':
            ames_dusty.add_ames_dusty(self.input_path, h5_file, wavelength, teff, specres)
            data_util.add_missing(model, ('teff', 'logg'), h5_file)

        elif model[0:9] == 'ames-cond':
            ames_cond.add_ames_cond(self.input_path, h5_file, wavelength, teff, specres)
            data_util.add_missing(model, ('teff', 'logg'), h5_file)

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
        distance : float
            Distance (pc). Not written if set to None.
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

        if distance:
            if 'objects/'+object_name+'/distance' in h5_file:
                del h5_file['objects/'+object_name+'/distance']

            h5_file.create_dataset('objects/'+object_name+'/distance',
                                   data=distance,
                                   dtype='f')  # [pc]

        if app_mag:
            flux = {}
            error = {}

            for item in app_mag:
                synphot = photometry.SyntheticPhotometry(item)
                flux[item], error[item] = synphot.magnitude_to_flux(app_mag[item][0],
                                                                    app_mag[item][1])

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

        sys.stdout.write('Adding object: '+object_name+'...')
        sys.stdout.flush()

        if spectrum:

            if 'objects/'+object_name+'/spectrum' in h5_file:
                del h5_file['objects/'+object_name+'/spectrum']

            data = np.loadtxt(spectrum)

            dset = h5_file.create_dataset('objects/'+object_name+'/spectrum',
                                          data=data[:, 0:3],
                                          dtype='f')

            dset.attrs['instrument'] = str(instrument)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        h5_file.close()

    def add_photometry(self,
                       library):
        """
        Parameters
        ----------
        library : str
            Photometry library.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'photometry' not in h5_file:
            h5_file.create_group('photometry')

        if 'photometry/'+library in h5_file:
            del h5_file['photometry/'+library]

        if library[0:7] == 'vlm-plx':
            vlm_plx.add_vlm_plx(self.input_path, h5_file)

        elif library[0:7] == 'leggett':
            leggett.add_leggett(self.input_path, h5_file)

        elif library[0:7] == 'mamajek':
            mamajek.add_mamajek(self.input_path, h5_file)

        h5_file.close()

    def add_calibration(self,
                        filename,
                        tag,
                        units=None,
                        scaling=None):
        """
        Function for adding a calibration spectrum to the database.

        Parameters
        ----------
        filename : str
            Filename with the calibration spectrum. The first column should contain the wavelength
            (micron), the second column the flux density (W m-2 micron-1), and the third column
            the error (W m-2 micron-1).
        tag : str
            Tag name in the database.
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

        if scaling is None:
            scaling = (1., 1.)

        h5_file = h5py.File(self.database, 'a')

        if 'spectra/calibration' not in h5_file:
            h5_file.create_group('spectra/calibration')

        if 'spectra/calibration/'+tag in h5_file:
            del h5_file['spectra/calibration/'+tag]

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

        sys.stdout.write('Adding calibration spectrum: '+tag+'...')
        sys.stdout.flush()

        h5_file.create_dataset('spectra/calibration/'+tag,
                               data=np.vstack((wavelength, flux, error)),
                               dtype='f')

        h5_file.close()

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    def add_spectrum(self,
                     spectrum,
                     sptypes=None):
        """
        Parameters
        ----------
        spectrum : str
            Spectral library.
        sptypes : tuple(str, )
            Spectral types ('F', 'G', 'K', 'M', 'L', 'T'). Currently only implemented for IRTF.

        Returns
        -------
        NoneType
            None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'spectra' not in h5_file:
            h5_file.create_group('spectra')

        if 'spectra/'+spectrum in h5_file:
            del h5_file['spectra/'+spectrum]

        if spectrum[0:5] == 'vega':
            vega.add_vega(self.input_path, h5_file)

        elif spectrum[0:5] == 'irtf':
            irtf.add_irtf(self.input_path, h5_file, sptypes)

        elif spectrum[0:5] == 'spex':
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

        index_max = np.unravel_index(sampler.lnprobability.argmax(),
                                     sampler.lnprobability.shape)

        max_prob = math.exp(sampler.lnprobability[index_max])
        best_sample = sampler.chain[index_max]

        if 'results' not in h5_file:
            h5_file.create_group('results')

        if 'results/mcmc' not in h5_file:
            h5_file.create_group('results/mcmc')

        if 'results/mcmc/'+tag in h5_file:
            del h5_file['results/mcmc/'+tag]

        dset = h5_file.create_dataset('results/mcmc/'+tag,
                                      data=sampler.chain,
                                      dtype='f')

        dset.attrs['type'] = str(spectrum[0])
        dset.attrs['spectrum'] = str(spectrum[1])
        dset.attrs['nparam'] = int(len(modelpar))

        if distance:
            dset.attrs['distance'] = float(distance)

        for i, item in enumerate(modelpar):
            dset.attrs['parameter'+str(i)] = str(item)

        dset.attrs['max_prob'] = max_prob

        for i, item in enumerate(modelpar):
            dset.attrs['best_sample'+str(i)] = best_sample[i]

        sys.stdout.write(f'Maximum probability: {max_prob:.2f}\n')
        sys.stdout.write(f'Parameter values:')
        for i, item in enumerate(modelpar):
            sys.stdout.write(f' {item}={best_sample[i]:.2f}')
        sys.stdout.write('\n')
        sys.stdout.flush()

        mean_accep = np.mean(sampler.acceptance_fraction)
        dset.attrs['acceptance'] = float(mean_accep)

        sys.stdout.write('Mean acceptance fraction: {0:.3f}'.format(mean_accep)+'\n')
        sys.stdout.flush()

        try:
            int_auto = emcee.autocorr.integrated_time(sampler.flatchain)

            sys.stdout.write('Integrated autocorrelation time = '+str(int_auto)+'\n')
            sys.stdout.flush()

        except emcee.autocorr.AutocorrError:
            int_auto = None

        if int_auto is not None:
            for i, item in enumerate(int_auto):
                dset.attrs['autocorrelation'+str(i)] = float(item)

        h5_file.close()

    def get_best_sample(self,
                        tag):
        """
        Parameters
        ----------
        tag : str
            Database tag with the MCMC results.

        Returns
        -------
        dict
            Parameters and values for the sample with the maximum posterior probability.
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag]

        nparam = dset.attrs['nparam']

        best_sample = {}

        for i in range(nparam):
            par_key = dset.attrs['parameter'+str(i)]
            par_value = dset.attrs['best_sample'+str(i)]

            best_sample[par_key] = par_value

        if dset.attrs.__contains__('distance'):
            best_sample['distance'] = dset.attrs['distance']

        h5_file.close()

        return best_sample

    def get_mcmc_spectra(self,
                         tag,
                         burnin,
                         random,
                         wavelength,
                         specres=None):
        """
        Parameters
        ----------
        tag : str
            Database tag with the MCMC samples.
        burnin : int
            Number of burnin steps.
        random : int
            Number of random samples.
        wavelength : tuple(float, float) or str
            Wavelength range (micron) or filter name. Full spectrum if set to None.
        specres : float
            Spectral resolution, achieved by smoothing with a Gaussian kernel. The original
            wavelength points are used if set to None.

        Returns
        -------
        tuple(species.core.box.ModelBox, )
            Boxes with the randomly sampled spectra.
        """

        sys.stdout.write('Getting MCMC spectra...')
        sys.stdout.flush()

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag]

        nparam = dset.attrs['nparam']
        spectrum_type = dset.attrs['type']
        spectrum_name = dset.attrs['spectrum']

        if specres is not None and spectrum_type == 'calibration':
            warnings.warn("Smoothing of the spectral resolution is not implemented for calibration "
                          "spectra.")

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
            param.append(str(dset.attrs['parameter'+str(i)]))

        if spectrum_type == 'model':
            readmodel = read_model.ReadModel(spectrum_name, wavelength)
        elif spectrum_type == 'calibration':
            readcalib = read_calibration.ReadCalibration(spectrum_name, None)

        boxes = []

        progbar = progress.bar.Bar('\rGetting MCMC spectra...',
                                   max=samples.shape[0],
                                   suffix='%(percent)d%%')

        for i in range(samples.shape[0]):
            model_par = {}
            for j in range(samples.shape[1]):
                model_par[param[j]] = samples[i, j]

            if distance:
                model_par['distance'] = distance

            if spectrum_type == 'model':
                specbox = readmodel.get_model(model_par, specres)
            elif spectrum_type == 'calibration':
                specbox = readcalib.get_spectrum(model_par)

            box.type = 'mcmc'

            boxes.append(specbox)

            progbar.next()

        progbar.finish()

        h5_file.close()

        return tuple(boxes)

    def get_mcmc_photometry(self,
                            tag,
                            burnin,
                            filter_id):
        """
        Parameters
        ----------
        tag : str
            Database tag with the MCMC samples.
        burnin : int
            Number of burnin steps.
        filter_id : str
            Filter ID for which the photometry is calculated.

        Returns
        -------
        numpy.ndarray
            Synthetic photometry (mag).
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag]

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
            readmodel = read_model.ReadModel(spectrum_name, filter_id)
        # elif spectrum_type == 'calibration':
        #     readcalib = read_calibration.ReadCalibration(spectrum_name, None)

        mcmc_phot = np.zeros((samples.shape[0], 1))

        progbar = progress.bar.Bar('Getting MCMC photometry...',
                                   max=samples.shape[0],
                                   suffix='%(percent)d%%')

        for i in range(samples.shape[0]):
            model_par = {}
            for j in range(nparam):
                model_par[param[j]] = samples[i, j]

            if distance:
                model_par['distance'] = distance

            if spectrum_type == 'model':
                mcmc_phot[i, 0], _ = readmodel.get_magnitude(model_par)
            # elif spectrum_type == 'calibration':
            #     specbox = readcalib.get_spectrum(model_par)

            progbar.next()

        progbar.finish()

        return mcmc_phot

    def get_object(self,
                   object_name,
                   filter_id=None,
                   inc_phot=True,
                   inc_spec=True):
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        filter_id : tuple(str, )
            Filter IDs for which the photometry is selected. All available photometry of the object
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

        sys.stdout.write('Getting object: '+object_name+'...')
        sys.stdout.flush()

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['objects/'+object_name]

        distance = np.asarray(dset['distance'])

        if inc_phot:

            magnitude = {}
            flux = {}

            if filter_id:
                for item in filter_id:
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

            filterids = tuple(magnitude.keys())

        else:

            magnitude = None
            flux = None
            filterids = None

        if inc_spec and 'objects/'+object_name+'/spectrum' in h5_file:
            spectrum = np.asarray(h5_file['objects/'+object_name+'/spectrum'])
        else:
            spectrum = None

        h5_file.close()

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        return box.create_box('object',
                              name=object_name,
                              filter=filterids,
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
        burnin : int
            Number of burnin samples to exclude. All samples are selected if set to None.
        random : int
            Number of random samples to select. All samples (with the burnin excluded) are
            selected if set to None.

        Returns
        -------
        species.core.box.SamplesBox
            Box with the MCMC samples.
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag]

        spectrum = dset.attrs['spectrum']
        nparam = dset.attrs['nparam']

        samples = np.asarray(dset)

        if burnin:
            samples = samples[:, burnin:, :]

        if random:
            ran_walker = np.random.randint(samples.shape[0], size=random)
            ran_step = np.random.randint(samples.shape[1], size=random)
            samples = samples[ran_walker, ran_step, :]

        param = []
        best_sample = []
        for i in range(nparam):
            param.append(dset.attrs['parameter'+str(i)])
            best_sample.append(dset.attrs['best_sample'+str(i)])

        h5_file.close()

        return box.create_box('samples',
                              spectrum=spectrum,
                              parameters=param,
                              samples=samples,
                              best_sample=best_sample)
