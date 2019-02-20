"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np


class ReadResult:
    """
    Text
    """

    def __init__(self,
                 res_type,
                 tag):
        """
        :param res_type: Result type.
        :type res_type: str
        :param tag: Database tag.
        :type tag: str

        :return: None
        """

        self.res_type = res_type
        self.tag = tag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_chisquare(self, fix=None):
        """
        :param fix:
        :type fix: dict

        :return: Grid points and reduced chi-square values.
        :rtype: numpy.ndarray, numpy.ndarray
        """

        h5_file = h5py.File(self.database, 'r')

        dset = h5_file['results/chisquare/'+self.tag]
        nparam = int(dset.attrs['nparam'])

        points = {}
        for i in range(nparam):
            param = dset.attrs['parameter'+str(i)]
            values = np.asarray(h5_file['results/chisquare/'+self.tag+'/'+param])
            points[param] = values

        chisquare = np.asarray(h5_file['results/chisquare/'+self.tag+'/chisquare'])

        h5_file.close()

        if fix:
            if nparam == 4:
                indices = [slice(0, chisquare.shape[0]),
                           slice(0, chisquare.shape[1]),
                           slice(0, chisquare.shape[2]),
                           slice(0, chisquare.shape[3])]

            for item in fix:
                param_index = list(points.keys()).index(item)
                value_index = min(enumerate(points[item]), key=lambda x: abs(x[1]-fix[item]))

                indices[param_index] = value_index[0]

                del points[item]

            chisquare = chisquare[tuple(indices)]

        return points, chisquare
