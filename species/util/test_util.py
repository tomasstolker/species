"""
Text
"""

import os

def create_config(test_path):
    """
    Function for creating a configuration file in the test folder.

    Parameters
    ----------
    test_path : str
        Folder where the unit tests are located.

    Returns
    -------
    NoneType
        None
    """

    config_file = os.path.join(test_path, 'species_config.ini')
    database_file = os.path.join(test_path, 'species_database.hdf5')
    data_folder = os.path.join(test_path, 'data/')

    config = open(config_file, 'w')
    config.write('[species]\n')
    config.write('database = '+database_file+'\n')
    config.write('config = '+config_file+'\n')
    config.write('input = '+data_folder)
    config.close()
