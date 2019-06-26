"""
Text
"""

import os
import sys
import warnings

from urllib.request import urlretrieve

import numpy as np

from astropy.io.votable import parse_single_table
from astropy.utils.exceptions import AstropyWarning

from species.analysis import photometry
from species.data import queries
from species.read import read_filter
from species.util import data_util


warnings.simplefilter('ignore', category=AstropyWarning)


def add_spex(input_path, database):
    """
    Function for adding the SpeX Prism Spectral Library to the database.

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

    database.create_group('spectra/spex')

    data_path = os.path.join(input_path, 'spex')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    url_all = 'http://svo2.cab.inta-csic.es/vocats/v2/spex/' \
              'cs.php?RA=180.000000&DEC=0.000000&SR=180.000000&VERB=2'

    xml_file = os.path.join(data_path, 'spex.xml')

    urlretrieve(url_all, xml_file)

    table = parse_single_table(xml_file)
    name = table.array['name']
    twomass = table.array['name2m']
    url = table.array['access_url']

    os.remove(xml_file)

    for i, item in enumerate(url):
        xml_file = os.path.join(data_path, twomass[i].decode('utf-8')+'.xml')
        urlretrieve(item.decode('utf-8'), xml_file)

        table = parse_single_table(xml_file)
        name = table.array['ID']
        name = name[0].decode('utf-8')
        url = table.array['access_url']

        sys.stdout.write('\rDownloading SpeX Prism Spectral Library... '+'{:<40}'.format(name))
        sys.stdout.flush()

        os.remove(xml_file)

        xml_file = os.path.join(data_path, name+'.xml')
        urlretrieve(url[0].decode('utf-8'), xml_file)

    sys.stdout.write('\rDownloading SpeX Prism Spectral Library... '+'{:<40}'.format('[DONE]')+'\n')
    sys.stdout.flush()

    h_twomass = photometry.SyntheticPhotometry('2MASS/2MASS.H')

    transmission = read_filter.ReadFilter('2MASS/2MASS.H')
    transmission.get_filter()

    # 2MASS H band zero point for 0 mag (Cogen et al. 2003)
    h_zp = 1.133e-9  # [W m-2 micron-1]

    for votable in os.listdir(data_path):
        if votable.endswith('.xml'):
            xml_file = os.path.join(data_path, votable)

            table = parse_single_table(xml_file)

            wavelength = table.array['wavelength']  # [Angstrom]
            flux = table.array['flux']  # Normalized units

            wavelength = np.array(wavelength*1e-4)  # [micron]
            flux = np.array(flux)

            # 2MASS magnitudes
            j_mag = table.get_field_by_id('jmag').value
            h_mag = table.get_field_by_id('hmag').value
            ks_mag = table.get_field_by_id('ksmag').value

            if j_mag == b'':
                j_mag = np.nan
            else:
                j_mag = float(j_mag)

            if h_mag == b'':
                h_mag = np.nan
            else:
                h_mag = float(h_mag)

            if ks_mag == b'':
                ks_mag = np.nan
            else:
                ks_mag = float(ks_mag)

            name = table.get_field_by_id('name').value
            name = name.decode('utf-8')
            twomass_id = table.get_field_by_id('name2m').value

            sys.stdout.write('\rAdding SpeX Prism Spectral Library... '+'{:<40}'.format(name))
            sys.stdout.flush()

            try:
                sptype = table.get_field_by_id('nirspty').value
                sptype = sptype.decode('utf-8')

            except KeyError:
                try:
                    sptype = table.get_field_by_id('optspty').value
                    sptype = sptype.decode('utf-8')

                except KeyError:
                    sptype = 'None'

            sptype = data_util.update_sptype(np.array([sptype]))[0]

            h_flux, _ = h_twomass.magnitude_to_flux(h_mag, None, h_zp)
            phot = h_twomass.spectrum_to_photometry(wavelength, flux)  # Normalized units

            flux *= h_flux/phot  # [W m-2 micron-1]

            spdata = np.vstack((wavelength, flux))

            simbad_id, distance = queries.get_distance('2MASS '+twomass_id.decode('utf-8'))  # [pc]

            dset = database.create_dataset('spectra/spex/'+name, data=spdata)

            dset.attrs['name'] = str(name)
            dset.attrs['sptype'] = str(sptype)
            dset.attrs['simbad'] = str(simbad_id)
            dset.attrs['2MASS/2MASS.J'] = j_mag
            dset.attrs['2MASS/2MASS.H'] = h_mag
            dset.attrs['2MASS/2MASS.Ks'] = ks_mag
            dset.attrs['distance'] = distance  # [pc]

    sys.stdout.write('\rAdding SpeX Prism Spectral Library... '+'{:<40}'.format('[DONE]')+'\n')
    sys.stdout.flush()

    database.close()
