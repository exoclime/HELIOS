# =============================================================================
# Module for computing continuous opacities
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


class ContiClass(object):
    """ class providing the continuous opacities """

    @staticmethod
    def bf_cross_sect_h_min(lamda, temp, press, f_e):
        """ calculates the bound-free cross-section for H- according to John 1988 """

        # convert cm to micron
        lamda *= 1e4

        if lamda < 0.125 or lamda > 1.6419:

            return 0

        else:

            lamda_0 = 1.6419

            c = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]

            f = sum([c[i] * (1/lamda - 1/lamda_0)**(i/2) for i in range(6)])

            sigma_lamda = 1e-18 * lamda**3 * (1/lamda - 1/lamda_0) ** 1.5 * f

            # alpha value in the John 1988 paper is wrong. It should be alpha = c * h / k = 1.439e4 micron K
            alpha = 1.439e4

            k_bf = 0.75 * temp**(-5/2) * np.exp(alpha/(lamda_0*temp)) * (1 - np.exp(-alpha/(lamda_0*temp))) * sigma_lamda

            sigma = k_bf * f_e * press

            return sigma

    @staticmethod
    def ff_cross_sect_h_min(lamda, temp, press, f_e):
        """ calculates the free-free cross-section for H- according to John 1988 """

        lamda *= 1e4

        if lamda < 0.1823:

            return 0

        elif lamda < 0.3645:

            j = 0

        else:

            j = 1

        # first row lower wavelength regime, 2nd row higher wavelength regime
        a = [[518.1021, 473.2636, -482.2089, 115.5291, 0, 0],[0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830]]
        b = [[-734.8666, 1443.4137, -737.1616, 169.6374, 0, 0],[0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170]]
        c = [[1021.1775, -1977.3395, 1096.8827, -245.6490, 0, 0],[0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8640]]
        d = [[-479.0721, 922.3575, -521.1341, 114.2430, 0, 0],[0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880]]
        e = [[93.1373, -178.9275, 101.7963, -21.9972, 0, 0],[0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880]]
        f = [[-6.4285, 12.3600, -7.0571, 1.5097, 0, 0],[0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850]]

        sum_term = sum([(5040/temp)**((i+2)/2) * (lamda**2 * a[j][i] + b[j][i] + c[j][i] / lamda + d[j][i] / lamda**2 + e[j][i] / lamda**3 + f[j][i] / lamda**4) for i in range(6)])

        k_ff = 1e-29 * sum_term

        sigma = k_ff * f_e * press

        return sigma

    @staticmethod
    def success():
        """ prints success message """
        print("\nCalculation of the continuous opacities --- Successful!")

if __name__ == "__main__":
    print("This module is for the computation of continuous and hence analytical opacities. So far Mister H- is a little bit lonely here.")
