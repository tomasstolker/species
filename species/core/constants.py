"""
Physical constants in the International System of Units (SI).
"""

from typing import Final
from astropy import constants

PLANCK: Final = constants.h.value  # (m2 kg s-1)
LIGHT: Final = constants.c.value  # (m s-1)
BOLTZMANN: Final = constants.k_B.value  # (J K-1)
GRAVITY: Final = constants.G.value  # (m3 kg−1 s−2)
PARSEC: Final = constants.pc.value  # (m)
AU: Final = constants.au.value  # (m)
R_JUP: Final = constants.R_jup.value  # (m)
M_JUP: Final = constants.M_jup.value  # (kg)
L_SUN: Final = constants.L_sun.value  # (W)
R_SUN: Final = constants.R_sun.value  # (m)
M_SUN: Final = constants.M_sun.value  # (kg)
R_EARTH: Final = constants.R_earth.value  # (m)
M_EARTH: Final = constants.M_earth.value  # (kg)
SIGMA_SB: Final = constants.sigma_sb.value  # (W m−2 K−4)
ATOMIC_MASS: Final = constants.u.value  # (kg)
RYDBERG: Final = constants.Ryd.value  # (m-1)
