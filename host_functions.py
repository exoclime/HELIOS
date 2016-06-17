# ==============================================================================
# Module for host function definitions that are not strictly read or write related
# Copyright (C) 2016 Matej Malik
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
from numpy.polynomial.legendre import leggauss as G
import phys_const as pc
import planets_and_stars as ps


def gaussian_weights(quant):
    """ sets the gaussian weights """

    quant.opac_weight = G(20)[1]


def spec_heat_cap(quant):
    """ calculates the specific heat capacity """

    n_dof = 5  # degrees of freedom
    quant.c_p = (2 + n_dof) / (2 + quant.mu) * pc.R_UNIV
    quant.c_p = np.float64(quant.c_p)


def planet_param(quant):
    """ takes the correct planetary parameters from dictionary if desired """

    if quant.planet in ps.planet_list:

        index = ps.planet_list.index(quant.planet)

        quant.g = np.float64(ps.dict_list[index]['g'])
        quant.a = np.float64(ps.dict_list[index]['a'] * pc.AU)
        quant.R_star = np.float64(ps.dict_list[index]['R_star'] * pc.R_SUN)
        quant.T_star = np.float64(ps.dict_list[index]['T_star'])
        print("\nUsing the planetary and orbital parameters of "+quant.planet+".")
    elif quant.planet == "manual":
        quant.g = np.float64(quant.g)
        quant.a = np.float64(quant.a * pc.AU)
        quant.R_star = np.float64(quant.R_star * pc.R_SUN)
        quant.T_star = np.float64(quant.T_star)
        print("\nUsing manual input for the planetary and orbital parameters.")
    else:
        print("\nInvalid choice for planetary parameters. Aborting...")
        raise SystemExit()


def effective_temp(quant):
    """ calculates the effective temperature corresponding to the chosen heat re-distribution (f factor) """

    result = quant.f_factor**0.25 * (quant.R_star/quant.a)**0.5 * quant.T_star

    return result


def initial_temp(quant, read):
    """ determines the initial temperature profile """

    if quant.restart == 0:
        for n in range(quant.nlayer):
            quant.T_lay.append(effective_temp(quant))

        print("\nStarting with an isothermal TP-profile with T_eff = {:.1f}".format(effective_temp(quant)), " K.")

    elif quant.restart == 1:
        read.read_restart_file(quant)
        quant.T_lay = quant.T_restart

        print("\nStarting with chosen restart temperature profile.")

    else:
        print("\nAbort! Restart parameter corrupt. Check that the value is 0 or 1.")


def sum_mean_optdepth(quant, i, opac):
    """ returns the summed up optical depth from the TOA with an opacity source, e.g. Rosseland mean opacity """

    tau = 0
    for j in np.arange(quant.nlayer-1, i-1, -1):
        tau += quant.delta_colmass[j] * opac[j]
    return tau


def success_message():
    """ prints the message that you have been desperately waiting for """

    print("\nDone! Everything appears to have worked fine :-)\n")

if __name__ == "__main__":
    print("This module stores the definitions for the functions living on the host.")