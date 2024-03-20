# ==============================================================================
# Module used to add additional heating terms to HELIOS
# Copyright (C) 2020 - 2022 Matej Malik
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


import numpy as np
from scipy import interpolate


def read_heating_file(quant):

    add_heat_file = np.genfromtxt(quant.add_heating_path, names=True, dtype=None, skip_header=quant.add_heating_file_header_lines)

    press = add_heat_file[quant.add_heating_file_press_name]

    if quant.add_heating_file_press_unit == "cgs":

        pass

    elif quant.add_heating_file_press_unit == "bar":

        press *= 1e6

    elif quant.add_heating_file_press_unit == "Pa":

        press *= 1e1

    else:

        raise IOError("Unknown pressure unit in additional heating file. Please double-check your input.")

    heat_dens = add_heat_file[quant.add_heating_file_data_name]

    heat_dens *= quant.add_heating_file_data_conv_factor

    return press, heat_dens


def load_heating_terms_or_not(quant):

    if quant.add_heating == 1:

        press_orig, heat_dens_orig = read_heating_file(quant)

        # get logarithm of pressure because makes more sense for interpolation
        log_press_orig = [np.log10(p) for p in press_orig]
        log_helios_press = [np.log10(p) for p in quant.p_lay]

        heat_dens_helios = interpolate.interp1d(log_press_orig, heat_dens_orig, kind='linear', bounds_error=False,
                                          fill_value=(heat_dens_orig[-1], heat_dens_orig[0]))(log_helios_press)

        quant.add_heat_dens = heat_dens_helios

    elif quant.add_heating == 0:

        quant.add_heat_dens = np.zeros(quant.nlayer)



if __name__ == "__main__":
    print("This module adds more heating to the model. But be careful, when there is more heating there is also more cooling.")
