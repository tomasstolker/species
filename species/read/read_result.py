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
        Parameters
        ----------
        res_type : str
            Result type.
        tag : str
            Database tag.

        Returns
        -------
        NoneType
            None
        """

        self.res_type = res_type
        self.tag = tag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_chisquare(self, fix=None):
        """
        Parameters
        ----------
        fix : dict
            Some text.

        Returns
        -------
        numpy.ndarray
            Grid points.
        numpy.ndarray
            Reduced chi-square values.
        """

        with h5py.File(self.database, 'r') as h5_file:
            dset = h5_file['results/chisquare/'+self.tag]
            nparam = int(dset.attrs['nparam'])

            points = {}
            for i in range(nparam):
                param = dset.attrs['parameter'+str(i)]
                values = np.asarray(h5_file['results/chisquare/'+self.tag+'/'+param])
                points[param] = values

            chisquare = np.asarray(h5_file['results/chisquare/'+self.tag+'/chisquare'])

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
