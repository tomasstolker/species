"""
Parallax module.
"""

import os
import sys

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

class NoStdStreams(object):
    """
    Text
    """
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

with NoStdStreams():
    from astroquery.gaia import Gaia


def get_distance(target):
    """
    :param target: Target name.
    :type target: str

    :return: Distance (pc).
    :rtype: float
    """

    if target[-2:] == "AB":
        target = target[:-2]

    elif target[-3:] == "ABC":
        target = target[:-3]

    if target[0:7] == "DENIS-P":
        target = target[:5]+target[7:]

        if target[-2] == ".":
            target = target[:-2]

    if target[0:5] == "DENIS" and target[6:7] != "J":
        target = target[:5]+" J"+target[6:]

    parallax = None

    catalogues = ('J/ApJS/201/19',
                  'J/ApJ/752/56',
                  'J/AJ/152/24',
                  'J/ApJ/862/173',
                  'J/ApJ/753/156',
                  'J/AJ/103/638')

    for _, item in enumerate(catalogues):
        result = Vizier.query_object(target, catalog=item)

        if not result._dict:
            continue

        else:
            if item == 'J/ApJS/201/19':
                # Dupuy & Liu (2012)
                # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJS/201/19
                parallax = result[0]['plx'][0] # [mas]

            elif item == 'J/ApJ/752/56':
                # Faherty et al. (2012)
                # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/752/56
                parallax = result[0]['Plx'][0] # [mas]

            elif item == 'J/AJ/152/24':
                # Weinberger et al. (2016)
                # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/AJ/152/24
                parallax = result[0]['Plx'][0] # [mas]

            elif item == 'J/ApJ/862/173':
                # Theissen (2018)
                # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/862/173
                parallax = result[0]['Plx'][0] # [mas]

            elif item == 'J/ApJ/753/156':
                # Kirkpatrick et al. (2012)
                # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/ApJ/753/156
                parallax = result[0]['Plx'][0] # [mas]

            elif item == 'J/AJ/103/638':
                # Monet et al. (1992)
                # http://cdsarc.u-strasbg.fr/viz-bin/cat/J/AJ/103/638
                parallax = result[0]['plx'][0] # [mas]

            break

    if parallax is None:
        simbad = Simbad.query_object(target)

        if simbad is not None:
            coord_ra = simbad['RA'][0]
            coord_dec = simbad['DEC'][0]

            coord = SkyCoord(ra=coord_ra, dec=coord_dec, unit=(u.hourangle, u.deg), frame='icrs')

            width = u.Quantity(0.01, u.deg)
            height = u.Quantity(0.01, u.deg)

            result = Gaia.query_object(coordinate=coord, width=width, height=height)

            if result:
                parallax = result['parallax'][0] # [mas]

    if parallax is None:
        distance = np.nan

    else:
        if not isinstance(parallax, np.ma.core.MaskedConstant) and parallax > 0.:
            distance = 1./(parallax*1e-3) # [pc]
        else:
            distance = np.nan

    return distance
