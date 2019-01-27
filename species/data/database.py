"""
Database module.
"""

import os
import sys
import warnings
import configparser

import h5py
import requests
import numpy as np

from .. analysis import photometry
from .. core import box
from .. read import read_model
from . import drift_phoenix
from . import vega
from . import irtf
from . import spex
from . import vlm_plx


warnings.simplefilter('ignore', UserWarning)


class Database:
    """
    Text.
    """

    def __init__(self):
        """
        :return: None
        """

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']
        self.input_path = config['species']['input']

    def list_items(self):
        """
        return: None
        """

        print("Database content:")

        def descend(h5_object,
                    seperator=''):
            """
            :param h5_object:
            :type h5_object: h5py._hl.files.File, h5py._hl.group.Group, h5py._hl.dataset.Dataset
            :param separator:
            :type separator: str

            :return: None
            """

            if isinstance(h5_object, (h5py._hl.files.File, h5py._hl.group.Group)):
                for key in h5_object.keys():
                    print(seperator, '-', key, ':', h5_object[key])
                    descend(h5_object[key], seperator=seperator+'\t')

            elif isinstance(h5_object, h5py._hl.dataset.Dataset):
                for key in h5_object.attrs.keys():
                    print(seperator, '-', key, ':', h5_object.attrs[key])

        h5_file = h5py.File(self.database, 'r')
        descend(h5_file)

    def add_filter(self,
                   filter_id,
                   filename=None):
        """
        :param filter_id: Filter ID from the SVO Filter Profile Service (e.g., 'Paranal/NACO.Lp').
        :type filter_id: str
        :param filename: Filename with the filter profile. The first column should contain the
                         wavelength (micron) and the second column the transmission (no units).
                         The profile is downloaded from the SVO Filter Profile Service if set to
                         None.
        :type filename: str

        :return: None
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

        url = 'http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii&id='+filter_id

        if filename:
            data = np.loadtxt(filename)
            wavelength = data[:, 0]
            transmission = data[:, 1]

        else:
            session = requests.Session()
            response = session.get(url)
            data = response.content

            wavelength = []
            transmission = []
            for line in data.splitlines():
                if not line.startswith(b'#'):
                    split = line.split(b' ')

                    wavelength.append(float(split[0])*1e-4) # [micron]
                    transmission.append(float(split[1]))

            wavelength = np.array(wavelength)
            transmission = np.array(transmission)

        h5_file.create_dataset('filters/'+filter_id,
                               data=np.vstack((wavelength, transmission)),
                               dtype='f')

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        h5_file.close()

    def add_model(self,
                  model):
        """
        :param model: Model name.
        :type model: str

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'models' not in h5_file:
            h5_file.create_group('models')

        if model[0:13] == 'drift-phoenix':
            drift_phoenix.add_drift_phoenix(self.input_path, h5_file)
            drift_phoenix.add_missing(h5_file)

        # elif model[0:9] == 'petitcode':
        #     petitcode.add_petitcode(self.input_path, h5_file)

        h5_file.close()

    def add_object(self,
                   object_name,
                   distance,
                   app_mag):
        """
        :param object_name: Object name.
        :type object_name: str
        :param distance: Distance (pc).
        :type distance: float
        :param app_mag: Apparent magnitudes.
        :type app_mag: dict

        :return: None
        """

        flux = {}
        error = {}

        for item in app_mag:
            mag = app_mag[item]

            synphot = photometry.SyntheticPhotometry(item)
            flux[item], error[item] = synphot.magnitude_to_flux(mag[0], mag[1])

        sys.stdout.write('Adding object: '+object_name+'...')
        sys.stdout.flush()

        h5_file = h5py.File(self.database, 'a')

        if 'objects' not in h5_file:
            h5_file.create_group('objects')

        if 'objects/'+object_name not in h5_file:
            h5_file.create_group('objects/'+object_name)

        if 'objects/'+object_name+'/distance' in h5_file:
            del h5_file['objects/'+object_name+'/distance']

        h5_file.create_dataset('objects/'+object_name+'/distance',
                               data=distance,
                               dtype='f') # [pc]

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

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        h5_file.close()

    def add_photometry(self,
                       library):
        """
        :param library: Photometry library.
        :type library: str

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'photometry' not in h5_file:
            h5_file.create_group('photometry')

        if 'photometry/'+library in h5_file:
            del h5_file['photometry/'+library]

        if library[0:7] == 'vlm-plx':
            vlm_plx.add_vlm_plx(self.input_path, h5_file)

        h5_file.close()

    def add_spectrum(self,
                     spectrum):
        """
        :param spectrum: Spectral library.
        :type spectrum: str

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'spectra' not in h5_file:
            h5_file.create_group('spectra')

        if 'spectra/'+spectrum in h5_file:
            del h5_file['spectra/'+spectrum]

        if spectrum[0:5] == 'vega':
            vega.add_vega(self.input_path, h5_file)

        elif spectrum[0:5] == 'irtf':
            irtf.add_irtf(self.input_path, h5_file)

        elif spectrum[0:5] == 'spex':
            spex.add_spex(self.input_path, h5_file)

        h5_file.close()

    def get_chisquare(self,
                      tag):
        """
        :param tag:
        :type tag: str

        :return:
        :rtype: species.core.box.SamplesBox
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/chisquare/'+tag]

        nparam = dset.attrs['nparam']
        model = dset.attrs['model']

        param = {}
        for i in range(nparam):
            par_key = dset.attrs['parameter'+str(i)]
            par_value = dset.attrs[par_key]

            param[par_key] = par_value

        h5_file.close()

        return model, param

    def get_mcmc_spectra(self,
                         tag,
                         burnin,
                         random,
                         wavelength,
                         coverage):

        """
        :param tag:
        :type tag: str
        :param burnin:
        :type burnin: int
        :param random:
        :type random: int
        :param coverage:
        :type coverage: tuple
        :param coverage:
        :type coverage: tuple

        :return:
        :rtype: tuple(species.core.box.ModelBox, )
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag]

        model = dset.attrs['model']
        distance = dset.attrs['distance']
        nparam = dset.attrs['nparam']

        samples = np.asarray(dset)
        samples = samples[:, burnin:, :]

        ran_walker = np.random.randint(samples.shape[0], size=random)
        ran_step = np.random.randint(samples.shape[1], size=random)
        samples = samples[ran_walker, ran_step, :]

        param = []
        for i in range(nparam):
            param.append(str(dset.attrs['parameter'+str(i)]))

        readmodel = read_model.ReadModel(model, wavelength)
        boxes = []

        for i in range(samples.shape[0]):
            model_par = {}
            for j in range(samples.shape[1]):
                model_par[param[j]] = samples[i, j]

            model_par['distance'] = distance

            modelbox = readmodel.get_model(model_par, coverage)
            modelbox.type = 'mcmc'

            boxes.append(modelbox)

        h5_file.close()

        return tuple(boxes)

    def get_object(self,
                   object_name,
                   filters):
        """
        :param object_name:
        :type object_name: str
        :param filters:
        :type filters: tuple(str, )

        :return:
        :rtype: species.core.box.ObjectBox
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['objects/'+object_name]

        distance = np.asarray(dset['distance'])

        magnitude = {}
        flux = {}

        for item in filters:
            data = dset[item]

            magnitude[item] = np.asarray(data[0:2])
            flux[item] = np.asarray(data[2:4])

        h5_file.close()

        return box.create_box('object',
                              name=object_name,
                              magnitude=magnitude,
                              flux=flux,
                              distance=distance)

    def get_samples(self,
                    tag,
                    burnin=None,
                    random=None):
        """
        :param tag:
        :type tag: str
        :param burnin:
        :type burnin: int
        :param random:
        :type random: int

        :return:
        :rtype: species.core.box.SamplesBox
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag]

        model = dset.attrs['model']
        nparam = dset.attrs['nparam']

        samples = np.asarray(dset)

        if burnin:
            samples = samples[:, burnin:, :]

        if random:
            ran_walker = np.random.randint(samples.shape[0], size=random)
            ran_step = np.random.randint(samples.shape[1], size=random)
            samples = samples[ran_walker, ran_step, :]

        param = []
        chisquare = []
        for i in range(nparam):
            param.append(dset.attrs['parameter'+str(i)])
            chisquare.append(dset.attrs['chisquare'+str(i)])

        h5_file.close()

        return box.create_box('samples',
                              model=model,
                              parameters=param,
                              samples=samples,
                              chisquare=chisquare)
