"""
Text
"""

import os
import sys

from urllib.request import urlretrieve

import h5py
import numpy as np
import pandas as pd

from species.util import data_util


def add_mamajek(input_path,
                database):
    """
    Function for adding "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence".
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    NoneType
        None
    """

    data_file = os.path.join(input_path, 'EEM_dwarf_UBVIJHK_colors_Teff.txt')

    url = 'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt'

    if not os.path.isfile(data_file):
        sys.stdout.write('Downloading Stellar Colors and Effective Temperatures (53 kB)...')
        sys.stdout.flush()

        urlretrieve(url, data_file)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

    sys.stdout.write('Adding Stellar Colors and Effective Temperatures...')
    sys.stdout.flush()

    group = 'photometry/mamajek'

    database.create_group(group)

    dataframe = pd.read_csv(data_file,
                            delimiter=r"\s+",
                            nrows=124,
                            header=22,
                            na_values=['...', '....', '.....', ',..', '19.52:'],
                            dtype={'SpT': str, 'Teff': float, 'logT': float, 'BCv': float,
                                   'U-B': float, 'Mv': float, 'logL': float, 'B-V': float,
                                   'Bt-Vt': float, 'V-Rc': float, 'V-Ic': float, 'V-Ks': float,
                                   'J-H': float, 'H-Ks': float, 'Ks-W1': float, 'W1-W2': float,
                                   'W1-W3': float, 'W1-W4': float, 'Msun': float, 'logAge': str,
                                   'b-y': float, 'M_J': float, 'M_Ks': float, 'Mbol': float,
                                   'i-z': float, 'z-Y': float, 'R_Rsun': float, 'SpT2': str,
                                   'G-V': float, 'Bp-Rp': float, 'M_G': float, 'G-Rp': float})

    dataframe.columns = dataframe.columns.str.replace('#', '')

    sptype = np.asarray(dataframe['SpT'])
    sptype = data_util.update_sptype(sptype)

    flag = np.repeat('star', sptype.size)
    distance = np.repeat(10., sptype.size)  # [pc]

    v_mag = dataframe['Mv']  # [mag]
    b_mag = dataframe['B-V']+dataframe['Mv']  # [mag]
    u_mag = dataframe['U-B']+b_mag  # [mag]

    j_mag = dataframe['M_J']  # [mag]
    ks_mag = dataframe['M_Ks']  # [mag]
    h_mag = dataframe['H-Ks']+ks_mag  # [mag]

    w1_mag = -dataframe['Ks-W1']+ks_mag  # [mag]
    w2_mag = -dataframe['W1-W2']+w1_mag  # [mag]
    w3_mag = -dataframe['W1-W3']+w1_mag  # [mag]
    w4_mag = -dataframe['W1-W4']+w1_mag  # [mag]

    dtype = h5py.special_dtype(vlen=bytes)

    dset = database.create_dataset(group+'/sptype', (np.size(sptype), ), dtype=dtype)
    dset[...] = sptype

    dset = database.create_dataset(group+'/flag', (np.size(flag), ), dtype=dtype)
    dset[...] = flag

    database.create_dataset(group+'/distance', data=distance, dtype='f')
    database.create_dataset(group+'/Teff', data=dataframe['Teff'], dtype='f')
    database.create_dataset(group+'/U', data=u_mag, dtype='f')
    database.create_dataset(group+'/B', data=b_mag, dtype='f')
    database.create_dataset(group+'/V', data=v_mag, dtype='f')
    database.create_dataset(group+'/J', data=j_mag, dtype='f')
    database.create_dataset(group+'/H', data=h_mag, dtype='f')
    database.create_dataset(group+'/Ks', data=ks_mag, dtype='f')
    database.create_dataset(group+'/W1', data=w1_mag, dtype='f')
    database.create_dataset(group+'/W2', data=w2_mag, dtype='f')
    database.create_dataset(group+'/W3', data=w3_mag, dtype='f')
    database.create_dataset(group+'/W4', data=w4_mag, dtype='f')

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    database.close()
