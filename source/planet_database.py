# ==============================================================================
# Module with a database of planetary system parameters used in HELIOS
# Copyright (C) 2020 - 2022 Matej Malik
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

from source import phys_const as pc


class Planet(object):

    def __init__(self, R_p, g_p, a, T_star, R_star, g_star, metal_star, R_p_unit):

        self.R_p = R_p
        if R_p_unit == "R_Earth":
            self.R_p *= pc.R_EARTH / pc.R_JUP

        self.g_p = g_p
        self.a = a
        self.T_star = T_star
        self.R_star = R_star
        self.g_star = g_star
        self.metal_star = metal_star


planet_lib = {}

# Units are [R_p]=R_Earth or R_Jup, [g]=cm s^-2 or [g]=log(cm s^-2), [a]=AU, [R_star]=R_Sun, [T_star]=K

planet_lib["GJ_1214b"] = Planet(R_p=2.85, R_p_unit="R_Earth",
                                g_p=760,
                                a=0.01411,
                                T_star=3026,
                                R_star=0.216,
                                g_star=4.944,
                                metal_star=0.39
                                )  # references: Harpsoe et al. (2013)

planet_lib["HD_209458b"] = Planet(R_p=1.380, R_p_unit="R_Jupiter",
                               g_p=930,
                               a=0.04747,
                               T_star=6117,
                               R_star=1.162,
                               g_star=4.368,
                               metal_star=0.02
                               )  # references: Southworth (2010)

if __name__ == "__main__":
    print("This module stores information about planetary systems. No guarantee that anything here is remotely correct.")