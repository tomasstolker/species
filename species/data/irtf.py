"""
Module for adding IRTF spectra tot the database.
"""

import os
import sys
import tarfile

from urllib.request import urlretrieve

import numpy as np

from astropy.io import fits

from species.data import queries
from species.util import data_util


def add_irtf(input_path,
             database,
             sptypes):
    """
    Function to add the IRTF Spectral Library to the database.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        Database.
    sptypes : tuple(str, )
        Spectral types ('F', 'G', 'K', 'M', 'L', 'T').

    Returns
    -------
    NoneType
        None
    """

    datadir = os.path.join(input_path, 'irtf')

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    data_file = {'F': os.path.join(input_path, 'irtf/F_fits_091201.tar'),
                 'G': os.path.join(input_path, 'irtf/G_fits_091201.tar'),
                 'K': os.path.join(input_path, 'irtf/K_fits_091201.tar'),
                 'M': os.path.join(input_path, 'irtf/M_fits_091201.tar'),
                 'L': os.path.join(input_path, 'irtf/L_fits_091201.tar'),
                 'T': os.path.join(input_path, 'irtf/T_fits_091201.tar')}

    data_folder = {'F': os.path.join(input_path, 'irtf/F_fits_091201'),
                   'G': os.path.join(input_path, 'irtf/G_fits_091201'),
                   'K': os.path.join(input_path, 'irtf/K_fits_091201'),
                   'M': os.path.join(input_path, 'irtf/M_fits_091201'),
                   'L': os.path.join(input_path, 'irtf/L_fits_091201'),
                   'T': os.path.join(input_path, 'irtf/T_fits_091201')}

    data_type = {'F': 'F stars (4.4 MB)',
                 'G': 'G stars (5.6 MB)',
                 'K': 'K stars (5.5 MB)',
                 'M': 'M stars (7.5 MB)',
                 'L': 'L dwarfs (850 kB)',
                 'T': 'T dwarfs (100 kB)'}

    url_root = 'http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/Data/'

    url = {'F': url_root+'F_fits_091201.tar',
           'G': url_root+'G_fits_091201.tar',
           'K': url_root+'K_fits_091201.tar',
           'M': url_root+'M_fits_091201.tar',
           'L': url_root+'L_fits_091201.tar',
           'T': url_root+'T_fits_091201.tar'}

    for item in sptypes:
        if not os.path.isfile(data_file[item]):
            sys.stdout.write('Downloading IRTF Spectral Library - '+data_type[item]+'...')
            sys.stdout.flush()

            urlretrieve(url[item], data_file[item])

            sys.stdout.write(' [DONE]\n')
            sys.stdout.flush()

    sys.stdout.write('Unpacking IRTF Spectral Library...')
    sys.stdout.flush()

    for item in sptypes:
        tar = tarfile.open(data_file[item])
        tar.extractall(path=datadir)
        tar.close()

    sys.stdout.write(' [DONE]\n')
    sys.stdout.flush()

    database.create_group('spectra/irtf')

    for item in sptypes:
        for root, _, files in os.walk(data_folder[item]):

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

                    spt_split = sptype.split()

                    if item in ('L', 'T') or spt_split[1][0] == 'V':
                        sys.stdout.write('\rAdding IRTF Spectral Library... '+'{:<40}'.format(name))
                        sys.stdout.flush()

                        simbad_id, distance = queries.get_distance(name)  # [pc]

                        sptype = data_util.update_sptype(np.array([sptype]))[0]

                        dset = database.create_dataset('spectra/irtf/'+name,
                                                       data=spdata,
                                                       dtype='f')

                        dset.attrs['name'] = str(name).encode()
                        dset.attrs['sptype'] = str(sptype).encode()
                        dset.attrs['simbad'] = str(simbad_id).encode()
                        dset.attrs['distance'] = distance

    sys.stdout.write('\rAdding IRTF Spectral Library... '+'{:<40}'.format('[DONE]')+'\n')
    sys.stdout.flush()

    database.close()
