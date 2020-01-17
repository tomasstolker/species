"""
Module for setting up species in the working folder.
"""

import os
import sys
import configparser

import h5py

import species


class SpeciesInit:
    """
    Class for initiating species by creating the database and configuration file in case they are
    not present in the working folder, and creating the data folder for storage of input data.
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
                file_obj.write('data_folder = ./data/\n')

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        database_file = config['species']['database']
        data_folder = config['species']['data_folder']

        if not os.path.exists(data_folder):
            sys.stdout.write('Creating data folder...')
            sys.stdout.flush()

            os.makedirs(data_folder)

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

        if not os.path.isfile(database_file):
            sys.stdout.write('Creating species_database.hdf5...')
            sys.stdout.flush()

            h5_file = h5py.File(database_file, 'w')
            h5_file.close()

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()
