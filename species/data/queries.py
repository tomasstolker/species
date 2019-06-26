"""
Text
"""

import os
import sys
import time

import numpy as np

from numpy import ma
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


class NoStdStreams:
    """
    Text
    """

    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


with NoStdStreams():
    from astroquery.gaia import Gaia


def get_simbad(name):
    """
    Function for getting the Simbad identifier of an object.

    Parameters
    ----------
    name : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Simbad name.
    """

    simbad_id = []

    for item in name:
        time.sleep(0.15)

        simbad = Simbad.query_object(item)

        if simbad is None:
            sim_id = None

        else:
            sim_id = simbad['MAIN_ID'][0]
            # sim_id = sim_id.decode('utf-8')

        simbad_id.append(sim_id)

    return np.asarray(simbad_id, dtype=np.str_)


def get_distance(target):
    """
    Parameters
    ----------
    target : str
        Target name.

    Returns
    -------
    str
        SIMBAD name.
    float
        Distance (pc).
    """

    # Liu et al. (2016)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/833/96

    # Dupuy & Liu (2012)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJS/201/19

    # Faherty et al. (2012)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/752/56

    # Weinberger et al. (2016)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/AJ/152/24

    # Theissen (2018)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/862/173

    # Kirkpatrick et al. (2012)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/753/156

    # Monet et al. (1992)
    # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/AJ/103/638

    catalogues = ('J/ApJ/833/96'
                  'J/ApJS/201/19',
                  'J/ApJ/752/56',
                  'J/AJ/152/24',
                  'J/ApJ/862/173',
                  'J/ApJ/753/156',
                  'J/AJ/103/638')

    if target[-2:] == 'AB':
        target = target[:-2]

    elif target[-3:] == 'ABC':
        target = target[:-3]

    if target[0:7] == 'DENIS-P':
        target = target[:5]+target[7:]

        if target[-2] == '.':
            target = target[:-2]

    if target[0:5] == 'DENIS' and target[6:7] != 'J':
        target = target[:5]+' J'+target[6:]

    simbad_id = 'None'
    parallax = ma.masked

    time.sleep(0.15)

    simbad = Simbad()
    simbad.add_votable_fields('parallax')

    simbad = simbad.query_object(target)

    if simbad is not None:
        simbad_id = simbad['MAIN_ID'][0]
        simbad_id = simbad_id.decode('utf-8')
        parallax = simbad['PLX_VALUE'][0]

    if ma.is_masked(parallax):
        for _, item in enumerate(catalogues):
            result = Vizier.query_object(target, catalog=item)

            if result.keys():
                try:
                    parallax = result[0]['plx'][0]  # [mas]
                except KeyError:
                    pass

                if ma.is_masked(parallax):
                    try:
                        parallax = result[0]['Plx'][0]  # [mas]
                    except KeyError:
                        pass

            else:
                continue

            if not ma.is_masked(parallax):
                break

    if ma.is_masked(parallax):

        if simbad is not None:
            coord_ra = simbad['RA'][0]
            coord_dec = simbad['DEC'][0]

            coord = SkyCoord(ra=coord_ra, dec=coord_dec, unit=(u.hourangle, u.deg), frame='icrs')
            result = Gaia.query_object(coordinate=coord, width=1.*u.arcsec, height=1.*u.arcsec)

            if result:
                parallax = result['parallax'][0]  # [mas]

    if ma.is_masked(parallax) or parallax < 0.:
        distance = np.nan
    else:
        distance = 1./(parallax*1e-3)  # [pc]

    return simbad_id, distance
