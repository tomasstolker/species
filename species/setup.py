"""
Setup module.
"""

import os
import sys
import configparser

import h5py


class SpeciesInit(object):
    """
    Text.
    """

    def __init__(self,
                 database_path,
                 input_path):
        """
        :param database_path: Database location.
        :type database_path: str
        :param input_path: Input data location.
        :type input_path: str

        :return: None
        """

        self.database_path = database_path
        self.input_path = os.path.abspath(input_path)

        database_file = os.path.join(self.database_path, "species_database.hdf5")
        database_file = os.path.abspath(database_file)

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        if not os.path.exists(self.input_path):
            sys.stdout.write("Creating folder for input data... ")
            sys.stdout.flush()

            os.makedirs(self.input_path)

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

        if not os.path.isfile(database_file):
            sys.stdout.write("Creating species_database.hdf5... ")
            sys.stdout.flush()

            h5_file = h5py.File(database_file, 'w')
            h5_file.close()

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

        if not os.path.isfile(config_file):
            sys.stdout.write("Creating species_config.ini... ")
            sys.stdout.flush()

            config = configparser.ConfigParser()

            config['species'] = {'database': database_file,
                                 'config':config_file,
                                 'input':self.input_path}

            with open(config_file, 'w') as config_ini:
                config.write(config_ini)

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()
