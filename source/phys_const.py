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

from astropy import constants as const
import math


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
M_SUN = const.M_sun.cgs.value           # solar mass
R_JUP = const.R_jup.cgs.value           # radius Jupiter in cm, old value -- 6.9911e9
M_JUP = const.M_jup.cgs.value           # mass Jupiter
R_EARTH = const.R_earth.cgs.value       # radius Earth in cm, old value
M_EARTH = const.M_earth.cgs.value       # mass Earth
G = const.G.cgs.value                   # gravitational constant

# molecular weights (mus)
M_H2 = 2.01588
M_H2O = 18.0153
M_CO2 = 44.01
M_CO = 28.01
M_CH4 = 16.04
M_O2 = 31.9988
M_NO = 30.01
M_SO2 = 64.066
M_NH3 = 17.031
M_OH = 17.00734
M_HCN = 27.0253
M_C2H2 = 26.04
M_PH3 = 33.99758
M_H2S = 34.081
M_SO3 = 80.066
M_VO = 66.9409
M_TIO = 63.866
M_ALO = 42.98
M_SIO = 44.08
M_CAO = 56.0774
M_SIH = 29.09344
M_CAH = 41.085899
M_PO = 46.97316
M_MGH = 25.3129
M_NAH = 23.99771
M_ALH = 27.9889
M_CRH = 53.0040

# atomic weights (mus)
M_H = 1.007825
M_HE = 4.0026
M_C = 12.0096
M_N = 14.007
M_O = 15.999
M_F = 18.9984
M_NE = 20.1797
M_NA = 22.989769
M_MG = 24.305
M_AL = 26.9815385
M_SI = 28.085
M_P = 30.973761998
M_S = 32.06
M_CL = 35.45
M_AR = 39.948
M_k = 39.0983
M_CA = 40.078
M_TI = 47.867
M_V = 50.9415
M_CR = 51.9961
M_MN = 54.938044
M_FE = 55.845
M_COB = 58.933194
M_NI = 58.6934
M_CU = 63.546
M_ZN = 65.38



if __name__ == "__main__":
    print("This module provides physical constants. Nothing more, nothing less.")