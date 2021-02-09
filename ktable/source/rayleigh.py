# =============================================================================
# Module for computing Rayleigh scattering opacities
# Copyright (C) 2018 Matej Malik
# =============================================================================
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
# =============================================================================

import numpy as np
from source import phys_const as pc

class Rayleigh_scat(object):
    """ class providing the Rayleigh scattering calculation"""
    
    def __init__(self):
        self.King_h2 = 1
        self.n_ref_h2 = 2.65163e19
        self.King_he = 1
        self.King_co = 1
        self.n_ref_he = 2.546899e19
        self.King_h2o = (6 + 3 * 3e-4) / (6 - 7 * 3e-4)  # converted from depolarisation factor
        self.n_ref_co2 = 2.546899e19
        self.n_ref_n2 = 2.546899e19
        self.n_ref_o2 = 2.68678e19
        self.n_ref_co = 2.546899e19

    @staticmethod
    def index_h2(lam):
        """ calculates the refractive index of molecular hydrogen """

        result = 13.58e-5 * (1 + 7.52e-11 * lam**-2) + 1

        return result

    @staticmethod
    def index_he(lam):
        """ calculates the refractive index of helium """

        result = 1e-8 * (2283 + 1.8102e13/(1.5342e10 - lam**-2)) + 1

        return result

    @staticmethod
    def index_n2(lam):
        """ calculates the refractive index of N2 """

        if lam**-1 <= 21360:

            result = 1e-8 * (6498.2 + 307.4335e12/(14.4e9 - lam**-2)) + 1

        elif lam**-1 > 21360:

            result = 1e-8 * (5677.465 + 318.81874e12/(14.4e9 - lam**-2)) + 1

        return result

    @staticmethod
    def index_o2(lam):
        """ calculates the refractive index of O2 """

        result = 1e-8 * (20564.8 + 2.480899e13/(4.09e9 - lam**-2)) + 1

        return result

    @staticmethod
    def index_co(lam):
        """ calculates the refractive index of CO """

        result = 1e-8 * (22851 + 0.456e14 / (71427**2 - lam ** -2)) + 1

        return result

    @staticmethod
    def index_h2o(lam, press, temp, f_h2o):
        """ calculates the refractive index of H2O """

        dens = f_h2o * press * pc.M_H2O * pc.AMU / (pc.K_B * temp)  # mass density of water

        Lamda = lam / 0.589e-4
        delta = dens / 1.0
        theta = temp / 273.15

        Lamda_UV = 0.229202
        Lamda_IR = 5.432937

        a0 = 0.244257733
        a1 = 0.974634476e-2
        a2 = -0.373234996e-2
        a3 = 0.268678472e-3
        a4 = 0.158920570e-2
        a5 = 0.245934259e-2
        a6 = 0.900704920
        a7 = -0.166626219e-1

        A = delta * (a0 + a1*delta + a2*theta + a3*Lamda**2*theta + a4*Lamda**-2 + a5 / (Lamda**2 - Lamda_UV**2) + a6 / (Lamda**2 - Lamda_IR**2) + a7*delta**2)

        # because refractive index is complex number
        A = complex(A)

        result = ((2 * A + 1)/(1 - A))**0.5

        return result

    @staticmethod
    def index_co2(lam):
        """ calculates the refractive index of CO2 """

        bracket = 5799.25 / (128908.9**2 - lam**-2) + 120.05 / (89223.8**2 - lam**-2) + 5.3334 / (75037.5**2 - lam**-2) + 4.3244 / (67837.7**2 - lam**-2) + 0.1218145e-6 / (2418.136**2 - lam**-2)

        result = bracket * 1.1427e3 + 1

        return result

    ### reference densities (unless constant) ###
    @staticmethod
    def n_ref_h2o(press, temp, f_h2o):
        """ calculates the reference number density of water (it is the actual number density) """

        n_ref = f_h2o * press / (pc.K_B * temp)

        return n_ref

    ### King factors (unless constant) ###
    @staticmethod
    def King_co2(lam):
        """ calculates the King factor of CO2 """

        result = 1.1364 + 25.3e-12 * lam**-2

        return result

    @staticmethod
    def King_n2(lam):
        """ calculates the King factor of N2 """

        result = 1.034 + 3.17e-12 * lam**-1

        return result

    @staticmethod
    def King_o2(lam):
        """ calculates the King factor of O2 """

        result = 1.09 + 1.385e-11 * lam**-2 + 1.448e-20 * lam**-4

        return result

    ### cross-sections ###
    @staticmethod
    def cross_sect(lamda, index, n_ref, King, lamda_limit):
        """ calculates the scattering cross sections """

        if lamda <= lamda_limit:

            result = 24.0 * np.pi ** 3 / (n_ref ** 2 * lamda ** 4) * ((index ** 2 - 1.0) / (index ** 2 + 2.0)).real ** 2 * King
        else:
            result = 0

        return result

    @staticmethod
    def cross_sect_h(lamda):

        # coefficient
        cp = [1.26563, 3.73828125, 8.813930935, 19.15379502, 39.92303232, 81.10881152, 161.9089166, 319.0231631, 622.2679809, 1203.891509]

        # Thomson cross-section
        sigma_T = 0.665e-24

        # Lyman limit
        lamda_l = 91.2e-7

        sum_term = sum([cp[i] * (lamda_l / lamda)**(2*i) for i in range(10)])

        sigma = sigma_T * (lamda_l / lamda)**4 * sum_term

        return sigma

    @staticmethod
    def success():
        """ prints success message """
        print("\nCalculation of the Rayleigh scattering cross sections --- Successful!")

if __name__ == "__main__":
    print("This module is for the computation of Rayleigh scattering cross sections.")
