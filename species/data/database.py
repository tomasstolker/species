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

from .. core import box
from . import drift_phoenix
# from . import petitcode
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

    def add_photometry(self,
                       photometry):
        """
        :param photometry: Photometry library.
        :type photometry: str

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        if 'photometry' not in h5_file:
            h5_file.create_group('photometry')

        if 'photometry/'+photometry in h5_file:
            del h5_file['photometry/'+photometry]

        if photometry[0:7] == 'vlm-plx':
            vlm_plx.add_vlm_plx(self.input_path, h5_file)

        h5_file.close()

    def add_filter(self,
                   filter_id):
        """
        :param filter_id: Filter ID from the SVO Filter Profile Service (e.g., 'Paranal/NACO.Lp').
        :type filter_id: str

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

        h5_file = h5py.File(self.database, 'a')

        sys.stdout.write('Adding object: '+object_name+'...')
        sys.stdout.flush()

        if 'objects' not in h5_file:
            h5_file.create_group('objects')

        if 'objects/'+object_name not in h5_file:
            h5_file.create_group('objects/'+object_name)

        if 'objects/'+object_name+'/distance' in h5_file:
            del h5_file['objects/'+object_name+'/distance']

        h5_file.create_dataset('objects/'+object_name+'/distance',
                               data=distance,
                               dtype='f') # [pc]

        for _, item in enumerate(app_mag):
            if 'objects/'+object_name+'/'+item in h5_file:
                del h5_file['objects/'+object_name+'/'+item]

            h5_file.create_dataset('objects/'+object_name+'/'+item,
                                   data=app_mag[item],
                                   dtype='f') # [mag], [mag]

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        h5_file.close()

    def get_samples(self,
                    tag):
        """
        :param tag:
        :type tag: str

        :return:
        :rtype: species.core.box.SamplesBox
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['results/mcmc/'+tag]
        samples = np.asarray(dset)

        model = dset.attrs['model']

        if model == 'drift-phoenix':
            nparam = 4

        param = []
        for i in range(nparam):
            param.append(dset.attrs['parameter'+str(i+1)])

        h5_file.close()

        return box.create_box('samples', model=model, parameters=param, samples=samples)
