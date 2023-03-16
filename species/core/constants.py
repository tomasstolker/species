"""
Physical constants in the International System of Units (SI).
"""

from astropy import constants

PLANCK = constants.h.value  # (m2 kg s-1)
LIGHT = constants.c.value  # (m s-1)
BOLTZMANN = constants.k_B.value  # (J K-1)
GRAVITY = constants.G.value  # (m3 kg−1 s−2)
PARSEC = constants.pc.value  # (m)
AU = constants.au.value  # (m)
R_JUP = constants.R_jup.value  # (m)
M_JUP = constants.M_jup.value  # (kg)
L_SUN = constants.L_sun.value  # (W)
R_SUN = constants.R_sun.value  # (m)
M_SUN = constants.M_sun.value  # (kg)
R_EARTH = constants.R_earth.value  # (m)
M_EARTH = constants.M_earth.value  # (kg)
SIGMA_SB = constants.sigma_sb.value  # (W m−2 K−4)
ATOMIC_MASS = constants.u.value  # (kg)
RYDBERG = constants.Ryd.value  # (m-1)
