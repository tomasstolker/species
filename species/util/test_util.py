"""
Utility functions for running the unit tests.
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

    with open(config_file, 'w') as config:
        config.write('[species]\n')
        config.write('database = '+database_file+'\n')
        config.write('data_folder = '+data_folder)
