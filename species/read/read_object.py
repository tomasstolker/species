'''
Read module.
'''

import os
import configparser

import h5py
import numpy as np

from species.analysis import photometry


class ReadObject:
    '''
    Text
    '''

    def __init__(self,
                 object_name):
        '''
        :param object_name: Object name.
        :type object_name: str

        :return: None
        '''

        self.object_name = object_name

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_photometry(self,
                       filter_name):
        '''
        :param filter_name: Filter name.
        :type filter_name: str

        :return: Apparent magnitude (mag), magnitude error (error),
                 apparent flux (W m-2 micron-1), flux error (W m-2 micron-1).
        :rtype: numpy.ndarray
        '''

        h5_file = h5py.File(self.database, 'r')
        obj_phot = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])
        h5_file.close()

        return obj_phot

    def get_distance(self):
        '''
        :return: Distance (pc).
        :rtype: float
        '''

        h5_file = h5py.File(self.database, 'r')
        obj_distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])
        h5_file.close()

        return float(obj_distance)

    def get_absmag(self,
                   filter_name):
        '''
        :param filter_name: Filter name.
        :type filter_name: str

        :return: Absolute magnitude (mag), magnitude error (error).
        :rtype: float, float
        '''

        h5_file = h5py.File(self.database, 'r')
        obj_distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])
        obj_phot = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])
        h5_file.close()

        abs_mag = photometry.apparent_to_absolute(obj_phot[0], obj_distance)

        return abs_mag, obj_phot[1]
