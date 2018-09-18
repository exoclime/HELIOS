# ==============================================================================
# Module with physical constants
# Copyright (C) 2018 Matej Malik
#
# All values are in cgs units.
# ==============================================================================
# This file is part of HELIOS.
#
#     HELIOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     HELIOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You find a copy of the GNU General Public License in the main
#     HELIOS directory under <license.txt>. If not, see
#     <http://www.gnu.org/licenses/>.
# ==============================================================================

import math
from astropy import constants as const


C = const.c.cgs.value                   # speed of light in cm / s
K_B = const.k_B.cgs.value               # Boltzmann constant in erg / K
H = const.h.cgs.value                   # Planck constant in erg s
R_UNIV = const.R.cgs.value              # universal gas constant in erg / mol / K
SIGMA_SB = const.sigma_sb.cgs.value     # Stefan-Boltzmann constant in erg / cm2 / K
AU = const.au.cgs.value                 # astronomical unit in cm
AMU = const.u.cgs.value                 # atomic mass unit in g
M_E = const.m_e.cgs.value               # mass of electron in g
Q_E = const.e.esu.value                 # charge of electron in Franklin, cgs-unit of charge
R_SUN = const.R_sun.cgs.value           # solar radius in cm
R_JUP = const.R_jup.cgs.value           # radius Jupiter in cm, old value -- 6.9911e9
GAMMA = 0.5772156649                    # Euler-Mascheroni constant

if __name__ == "__main__":
    print("This module provides physical constants. Nothing more, nothing less.")