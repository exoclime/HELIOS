# =============================================================================
# Module for computing continuous opacities
# Copyright (C) 2018 - 2022 Matej Malik
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
from scipy import interpolate


class ContiClass(object):
    """ class providing the continuous opacities """

    @staticmethod
    def h_min_bf_cross_sect(lamda):
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

            ################ OUTDATED BLOCK -- taking H- abundance directly from FastChem now ##################
            #
            # alpha value in the John 1988 paper is wrong. It should be alpha = c * h / k = 1.439e4 micron K
            # alpha = 1.439e4
            #
            # k_bf = 0.75 * temp**(-5/2) * np.exp(alpha/(lamda_0*temp)) * (1 - np.exp(-alpha/(lamda_0*temp))) * sigma_lamda
            #
            # sigma = k_bf * f_e * press
            #
            # return sigma
            #
            #################

            return sigma_lamda  # returns cross-section per H- atom (i.e., needs to be multiplied with H- VMR later)

    @staticmethod
    def h_min_ff_cross_sect(lamda, temp, press):
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

        sigma = k_ff * press

        return sigma  # returns cross-section per electron and per H atom (i.e., needs to be multiplied with e- VMR and H VMR later)

    @staticmethod
    def include_he_min_opacity():

        # data from John 1994
        lamda_0 = [0.5063, 0.5695, 0.6509, 0.7594, 0.9113, 1.1391, 1.5188, 1.8225, 2.2782, 3.0376, 3.6451, 4.5564, 6.0751, 9.1127, 11.3909, 15.1878]
        lamda_plus = [30, 50, 80, 120, 160, 200]
        lamda_0_plus = lamda_0 + lamda_plus

        theta_0 = [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6, 100.8]  # last entry added manually because we want 50 K as bottom limit
        temp_0 = [5040 / t for t in theta_0]  # convert to Kelvin
        temp_0.sort()

        k_ff = [0.121, 0.145, 0.178, 0.227, 0.305, 0.444, 0.737, 1.030, 1.574, 2.765, 3.979, 6.234, 11.147, 25.268, 39.598, 70.580,
                0.100, 0.120, 0.148, 0.190, 0.258, 0.380, 0.643, 0.910, 1.405, 2.490, 3.592, 5.632, 10.059, 22.747, 35.606, 63.395,
                0.078, 0.094, 0.117, 0.152, 0.210, 0.316, 0.547, 0.782, 1.218, 2.167, 3.126, 4.897, 8.728, 19.685, 30.782, 54.757,
                0.072, 0.087, 0.109, 0.143, 0.198, 0.300, 0.522, 0.747, 1.165, 2.073, 2.990, 4.681, 8.338, 18.795, 29.384, 52.262,
                0.066, 0.081, 0.102, 0.133, 0.186, 0.283, 0.495, 0.710, 1.108, 1.971, 2.842, 4.448, 7.918, 17.838, 27.882, 49.583,
                0.061, 0.074, 0.094, 0.124, 0.173, 0.266, 0.466, 0.670, 1.045, 1.860, 2.681, 4.193, 7.460, 16.798, 26.252, 46.678,
                0.055, 0.067, 0.086, 0.114, 0.160, 0.247, 0.435, 0.625, 0.977, 1.737, 2.502, 3.910, 6.955, 15.653, 24.461, 43.488,
                0.049, 0.061, 0.077, 0.103, 0.147, 0.227, 0.400, 0.576, 0.899, 1.597, 2.299, 3.593, 6.387, 14.372, 22.456, 39.921,
                0.043, 0.053, 0.069, 0.092, 0.131, 0.204, 0.360, 0.518, 0.808, 1.435, 2.065, 3.226, 5.733, 12.897, 20.151, 35.882,
                0.036, 0.045, 0.059, 0.079, 0.113, 0.176, 0.311, 0.447, 0.698, 1.239, 1.783, 2.784, 4.947, 11.128, 17.386, 30.907,
                0.033, 0.041, 0.053, 0.072, 0.102, 0.159, 0.282, 0.405, 0.632, 1.121, 1.614, 2.520, 4.479, 10.074, 15.739, 27.979]

        upper_limit = [0.307, 0.275, 0.238, 0.227, 0.215, 0.202, 0.189, 0.173, 0.155, 0.134, 0.121]

        k_ff_plus = []

        for t in range(len(temp_0)):

            if t == 0:
                t_index = 0
            else:
                t_index = t - 1

            for x in range(len(lamda_0_plus)):

                if x < len(lamda_0):

                    k_ff_plus.append(k_ff[x + len(lamda_0) * t_index])

                else:

                    k_ff_plus.append(upper_limit[t_index] * lamda_0_plus[x] ** 2)

        k_ff_plus = [k * 1e-26 for k in k_ff_plus]  # correct order of magnitude

        log10_k_ff_plus = [np.log10(k) for k in k_ff_plus]
        log10_lamda_0_plus = [np.log10(l) for l in lamda_0_plus]

        func = interpolate.interp2d(temp_0, log10_lamda_0_plus, log10_k_ff_plus, kind='linear', bounds_error=False, fill_value=-30)

        return func


if __name__ == "__main__":
    print("This module is for the computation of continuous and hence analytical opacities. So far it includes Mister H- and Miss He- looking forward to be joined by others.")
