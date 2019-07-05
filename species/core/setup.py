"""
Setup module.
"""

import os
import sys
import configparser

import h5py

import species


class SpeciesInit:
    """
    Text.
    """

    def __init__(self,
                 config_path):
        """
        Parameters
        ----------
        config_path : str
            Location of the configuration file named *species_config.ini*.

        Returns
        -------
        NoneType
            None
        """

        sys.stdout.write('Initiating species v'+species.__version__+'...')
        sys.stdout.flush()

        self.config_path = config_path

        config_file = os.path.join(self.config_path, 'species_config.ini')

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        if not os.path.isfile(config_file):

            sys.stdout.write('Creating species_config.ini...')
            sys.stdout.flush()

            with open(config_file, 'w') as file_obj:
                file_obj.write('[species]\n')
                file_obj.write('database = species_database.hdf5\n')
                file_obj.write('config = species_config.ini\n')
                file_obj.write('input = ./')

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        database_file = config['species']['database']
        input_path = config['species']['input']

        if not os.path.exists(input_path):
            sys.stdout.write('Creating input folder...')
            sys.stdout.flush()

            os.makedirs(input_path)

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

        if not os.path.isfile(database_file):
            sys.stdout.write('Creating species_database.hdf5...')
            sys.stdout.flush()

            h5_file = h5py.File(database_file, 'w')
            h5_file.close()

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()
