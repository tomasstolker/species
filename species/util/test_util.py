"""
Text
"""

import os

def create_config():
    """
    Function for creating a configuration file in the test folder.
    """

    config_file = os.path.join(os.getcwd(), 'species_config.ini')
    database_file = os.path.join(os.getcwd(), 'species_database.hdf5')
    data_folder = os.path.join(os.getcwd(), 'data/')

    config = open(config_file, 'w')
    config.write('[species]\n')
    config.write('database = '+database_file+'\n')
    config.write('config = '+config_file+'\n')
    config.write('input = '+data_folder)
    config.close()
