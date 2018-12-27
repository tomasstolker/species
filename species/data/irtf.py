"""
Text
"""

import os
import sys
import tarfile

from urllib.request import urlretrieve

import numpy as np

from astropy.io import fits

from . import queries
from . import util


def add_irtf(input_path, database):
    """
    Function to add the IRTF Spectral Library to the database.

    :param input_path:
    :type input_path: str
    :param database:
    :type database: h5py._hl.files.File

    :return: None
    """

    data_file = [os.path.join(input_path, 'M_fits_091201.tar'),
                 os.path.join(input_path, 'L_fits_091201.tar'),
                 os.path.join(input_path, 'T_fits_091201.tar')]

    data_folder = [os.path.join(input_path, 'M_fits_091201'),
                   os.path.join(input_path, 'L_fits_091201'),
                   os.path.join(input_path, 'T_fits_091201')]

    data_type = ['M dwarfs (7.5 MB)',
                 'L dwarfs (850 kB)',
                 'T dwarfs (100 kB)']

    url_root = 'http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/Data/'

    url = [url_root+'M_fits_091201.tar',
           url_root+'L_fits_091201.tar',
           url_root+'T_fits_091201.tar']

    for i, item in enumerate(data_file):
        if not os.path.isfile(item):
            sys.stdout.write('Downloading IRTF Spectral Library - '+data_type[i]+'...')
            sys.stdout.flush()

            urlretrieve(url[i], item)

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

    sys.stdout.write('Unpacking IRTF Spectral Library...')
    sys.stdout.flush()

    for i, item in enumerate(data_file):
        tar = tarfile.open(item)
        tar.extractall(path=data_folder[i])
        tar.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    database.create_group('spectra/irtf')

    for i, item in enumerate(data_folder):
        for root, _, files in os.walk(item):

            for _, filename in enumerate(files):
                if filename[-9:] != '_ext.fits':
                    fitsfile = os.path.join(root, filename)
                    spdata, header = fits.getdata(fitsfile, header=True)

                    name = header['OBJECT']
                    sptype = header['SPTYPE']

                    if name[-2:] == 'AB':
                        name = name[:-2]
                    elif name[-3:] == 'ABC':
                        name = name[:-3]

                    sys.stdout.write('\rAdding IRTF Spectral Library... '+'{:<40}'.format(name))
                    sys.stdout.flush()

                    simbad_id, distance = queries.get_distance(name) # [pc]

                    sptype = util.update_sptype(np.array([sptype]))[0]

                    dset = database.create_dataset('spectra/irtf/'+name,
                                                   data=spdata,
                                                   dtype='f')

                    dset.attrs['name'] = str(name)
                    dset.attrs['sptype'] = str(sptype)
                    dset.attrs['simbad'] = str(simbad_id)
                    dset.attrs['distance'] = distance

    sys.stdout.write('\rAdding IRTF Spectral Library... '+'{:<40}'.format('[DONE]')+'\n')
    sys.stdout.flush()

    database.close()
