# =============================================================================
# Module for removing condensate species
# Copyright (C) 2016 Matej Malik
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

from scipy import interpolate as ip
import numpy as npy

class Condense(object):
    """ class providing the continuous opacities """

    @staticmethod
    def read_stability_curve(file):

        press = []
        temp = []

        with open(file) as f:
            next(f)
            for line in f:
                if line:
                    column = line.split()
                    press.append(float(column[0]) * 1e6)
                    temp.append(float(column[1]))

        return temp, press

    def calc_stability_curve(self, cond_path, species):
        """
        returns the function in P,T for the stability curve
        """

        temp, press = self.read_stability_curve(cond_path + species + ".dat")

        log_p = [npy.log10(p) for p in press]

        interp_func = ip.interp1d(log_p, temp, kind='linear', fill_value='extrapolate')

        return interp_func


if __name__ == "__main__":
    print("This module removes species from the atmosphere. It's basically the bouncer.")