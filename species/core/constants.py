"""
Physical constants in the International System of Units (SI).
"""

from beartype import typing
from astropy import constants

PLANCK: typing.Final = constants.h.value  # (m2 kg s-1)
LIGHT: typing.Final = constants.c.value  # (m s-1)
BOLTZMANN: typing.Final = constants.k_B.value  # (J K-1)
GRAVITY: typing.Final = constants.G.value  # (m3 kg−1 s−2)
PARSEC: typing.Final = constants.pc.value  # (m)
AU: typing.Final = constants.au.value  # (m)
R_JUP: typing.Final = constants.R_jup.value  # (m)
M_JUP: typing.Final = constants.M_jup.value  # (kg)
L_SUN: typing.Final = constants.L_sun.value  # (W)
R_SUN: typing.Final = constants.R_sun.value  # (m)
M_SUN: typing.Final = constants.M_sun.value  # (kg)
R_EARTH: typing.Final = constants.R_earth.value  # (m)
M_EARTH: typing.Final = constants.M_earth.value  # (kg)
SIGMA_SB: typing.Final = constants.sigma_sb.value  # (W m−2 K−4)
ATOMIC_MASS: typing.Final = constants.u.value  # (kg)
RYDBERG: typing.Final = constants.Ryd.value  # (m-1)
