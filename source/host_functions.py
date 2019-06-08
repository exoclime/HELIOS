# ==============================================================================
# Module for host function definitions that are not strictly read or write related
# Copyright (C) 2018 Matej Malik
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
import scipy.stats as st
from source import phys_const as pc



def planet_param(quant, read):
    """ takes the correct planetary parameters from dictionary if desired """

    if quant.planet == "manual":
        print("\nUsing manual input for the planetary and orbital parameters.")
    else:
        read.read_planet_file(quant)
        print("\nUsing planetary parameters for", quant.planet,", read from", read.planet_file)

    # convert to cgs units
    if quant.g < 10:
        quant.g = quant.fl_prec(10 ** quant.g)
    quant.a = quant.fl_prec(quant.a * pc.AU)
    quant.R_planet = quant.fl_prec(quant.R_planet * pc.R_JUP)
    quant.R_star = quant.fl_prec(quant.R_star * pc.R_SUN)
    # avoiding T_star = 0 to prevent numerical issues
    quant.T_star = quant.fl_prec(max(quant.T_star, 1))


def approx_f_from_formula(quant):
    " calculates the f redistribution factor from the approximative formula from Koll et al. in prep."

    # read in tau_lw from output file if it exists
    if "_post" in quant.name:
        name = quant.name[:-5]
    else:
        name = quant.name
    try:
        with open("./output/" + name + "/" + name + "_tau_lw_sw.dat", "r") as entr_file:
            next(entr_file)
            next(entr_file)
            for line in entr_file:
                column = line.split()
                quant.tau_lw = float(column[0])
        print("\ntau_lw read in from previous output file!")

    except IOError:
        print("\nWarning: Unable to read in tau_lw from file. Using input values!")

    # calculates the f factor
    T_eq = (quant.R_star / (2*quant.a))**0.5 * quant.T_star

    term = quant.tau_lw * (quant.p_boa / 1e6) ** (2 / 3) * (T_eq / 6000) ** (-4 / 3)

    quant.f_factor = 2 / 3 - 5 / 12 * term / (2 + term)


def calc_planck(lamda, temp):

    term1 = 2 * pc.H * pc.C**2 / lamda**5

    term2 = np.exp(pc.H * pc.C / (lamda * pc.K_B * temp)) - 1

    result = term1 * 1 / term2

    return result


def calc_tau_lw_sw(quant):

    num_lw = 0
    denom_lw = 0
    num_sw = 0
    denom_sw = 0

    for x in range(quant.nbin):

        tau_from_top = 0

        for i in range(quant.nlayer):

            tau_from_top += quant.delta_tau_band[x + i * quant.nbin]

        B_surface = calc_planck(quant.opac_wave[x], quant.T_surf)

        num_lw += B_surface * np.exp(-tau_from_top) * quant.opac_deltawave[x]
        denom_lw += B_surface * quant.opac_deltawave[x]

        if quant.T_star > 10:
            B_star = calc_planck(quant.opac_wave[x], quant.T_star)

            num_sw += B_star * tau_from_top * quant.opac_deltawave[x]
            denom_sw += B_star * quant.opac_deltawave[x]

    print(num_lw, denom_lw)

    tau_lw_tot = num_lw / denom_lw

    if quant.T_star > 10:
        tau_sw_tot = -np.log(num_sw / denom_sw)
    else:
        tau_sw_tot = 0

    with open("./output/" + quant.name + "/" + quant.name + "_tau_lw_sw.dat", "w") as file:
        file.writelines("This file contains the total longwave and shortwave optical depths at BOA (or surface if there), tau_lw and tau_sw")
        file.writelines("\n{:<10}{:<10}".format("tau_lw", "tau_sw"))
        file.writelines("\n{:<10g}{:<10g}".format(tau_lw_tot, tau_sw_tot))


def initial_temp(quant, read, Vmod):
    """ determines the initial temperature profile """

    if quant.singlewalk == 0:

        # calculate effective planetary temperature
        T_eff = (1.0-quant.dir_beam) * quant.f_factor ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star \
                + quant.dir_beam * abs(quant.mu_star) ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star

        # for efficiency reasons initial temperature has a lower limit of 100 K
        T_start = max(T_eff, 500)

        for n in range(quant.nlayer):
            quant.T_lay.append(T_start)

        print("\nStarting with an isothermal TP-profile with {:g}".format(T_start)+" K.")

    elif quant.singlewalk == 1:

        read.read_temperature_file(quant)

        quant.T_lay = quant.T_restart

        print("\nStarting with chosen temperature profile.")


def temp_calcs(quant):
    """ computes the final effective temperature """

    T_eff_global = 0.25 ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star

    T_eff_dayside = 0.667 ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star

    T_eff_model = (1.0 - quant.dir_beam) * quant.f_factor ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star \
            + quant.dir_beam * abs(quant.mu_star) ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star

    T_star_brightness = (quant.F_down_tot[quant.ninterface - 1] / pc.SIGMA_SB) ** 0.25
    T_planet_brightness = (quant.F_up_tot[quant.ninterface - 1] / pc.SIGMA_SB) ** 0.25

    return T_eff_global, T_eff_dayside, T_eff_model, T_star_brightness, T_planet_brightness


def calc_F_intern(quant):
    """ calculates the internal flux """

    quant.F_intern = pc.SIGMA_SB * quant.T_intern ** 4.0


def set_up_numerical_parameters(quant):
    """ sets up additional parameters used in the calculation """

    # limit for w_0 where to start to apply the pure scattering solutions
    quant.w_0_limit = quant.fl_prec(1.0 - 1e-6)

    # limit where to switch from noniso to iso equations to make them more stable
    quant.delta_tau_limit = quant.fl_prec(1e-4)

    # limit where global energy equilibrium is assumed
    quant.global_limit = quant.fl_prec(1e-3)

    # limit where radiative convergence is assumed
    if quant.prec == "double":
        quant.local_limit_rad_iter = quant.fl_prec(1e-7)
    elif quant.prec == "single":
        quant.local_limit_rad_iter = quant.fl_prec(1e-5)

    # limit where local rad.-conv. convergence is assumed
    quant.local_limit_conv_iter = quant.fl_prec(1e-7)

    # sets the appropriate gaussian weights
    quant.gauss_weight = G(quant.ny)[1]


def check_for_global_eq(quant):
    """ checks for global equilibrium """

    criterion = 0

    if quant.T_intern != 0: # with internal heat
        lim_quant = abs(quant.F_intern - quant.F_net[quant.ninterface - 1]) / quant.F_intern
    elif quant.T_intern == 0: # without internal heat. in this case there must be a star
        lim_quant = abs(quant.F_net[quant.ninterface - 1]) / quant.F_down_tot[quant.ninterface - 1]

    # user feedback
    if quant.iter_value % 10 == 0:
        print("The relative difference between TOA and BOA net flux is : {:.2e}".format(lim_quant)
              + " and should be less than {:.2e}".format(quant.global_limit) + ".")

    if lim_quant < quant.global_limit:
        criterion = 1

    return criterion


def calc_surf_temperature(quant):
    """ calculates the surface temperature from the downward incoming (and absorbed) flux and the interior heat flux """

    # leaving away (1-albedo)/emissivity in the first term for clarity
    quant.T_surf = (quant.F_down_tot[0]/pc.SIGMA_SB + quant.T_intern**4.0/(1.0 - quant.surf_albedo))**0.25

    # change the upward BOA flux to match the Stefan-Boltzmann law and to include the surface albedo
    quant.F_up_tot[0] = quant.surf_albedo * quant.F_down_tot[0] + (1.0 - quant.surf_albedo) * pc.SIGMA_SB * quant.T_surf**4.0

    # the corresponding net flux needs to be tweaked as well in order for the temperature iteration to work properly
    quant.F_net[0] = quant.F_up_tot[0] - quant.F_down_tot[0]


def check_for_local_eq(quant):
    """ checks for local equilibrium """

    criterion = 0

    converged_list = []
    quant.marked_red = np.zeros(quant.nlayer)

    for i in range(quant.nlayer):

        if quant.conv_layer[i] == 0:

            # temperature smoothing (analogous to rad_temp_iter kernel)
            t_mid = quant.T_lay[i]

            if quant.smooth == 1:
                if quant.p_lay[i] < 1e6 and i < quant.nlayer - 1:
                    t_mid = (quant.T_lay[i - 1] + quant.T_lay[i + 1]) / 2

            F_temp = (t_mid - quant.T_lay[i])**7
            combined_F_net = quant.F_net_diff[i] + F_temp

            div_lim_quant = abs(combined_F_net) / (pc.SIGMA_SB * quant.T_lay[i] ** 4.0)

            # check for criterion satisfaction
            if div_lim_quant < quant.local_limit_conv_iter:
                converged_list.append(1)
            else:
                quant.marked_red[i] = 1
                if quant.iter_value % 10 == 9:
                    print("layer: {:<5g}, delta_flux/BB_layer: {:<12.3e}".format(i, div_lim_quant))

    if len(converged_list) == quant.nlayer - sum(quant.conv_layer):
        criterion = 1

    # user feedback
    if quant.iter_value % 10 == 0:
        print("Radiative convergence: " + str(len(converged_list)) +
              " out of " + str(quant.nlayer - sum(quant.conv_layer)) + " radiative layers.\n")

    return criterion


def check_for_global_local_equilibrium(quant):
    """ checks for rad.-conv. convergence in terms of global and local equilibrium """

    # criterion for the global rad. equilibrium
    crit_global = check_for_global_eq(quant)

    # criterion for the local rad. equilibrium in the radiative zones
    crit_local = check_for_local_eq(quant)

    # combine global and local criteria
    crit = crit_global * crit_local

    return crit


def sum_mean_optdepth(quant, i, opac):
    """ returns the summed up optical depth from the TOA with an opacity source, e.g. Rosseland mean opacity """

    tau = 0
    for j in np.arange(quant.nlayer-1, i-1, -1):
        if opac[j] == -3:
            continue
        else:
            tau += quant.delta_colmass[j] * opac[j]

    if tau > 0:
        return tau
    else:
        return -3


def conv_check(quant):
    """ checks whether the lapse rate exceeds the allowed dry adiabat, if yes mark it for convective correction """

    # erase all previous information and start to check from zero
    quant.conv_unstable = np.zeros(quant.nlayer, np.int32)

    for i in range(quant.nlayer-1):

        T_in_between_lim = quant.T_lay[i] * (quant.p_int[i+1] / quant.p_lay[i]) ** (quant.kappa_lay[i] * (1+1e-6)) # was 1e-10

        T_ad_lim = T_in_between_lim * (quant.p_lay[i+1] / quant.p_int[i+1]) ** (quant.kappa_int[i+1] * (1+1e-6))

        if quant.T_lay[i+1] < T_ad_lim:

            quant.conv_unstable[i] = 1
            quant.conv_unstable[i+1] = 1


def conv_correct(quant):
    """ corrects unstable lapse rates to dry adiabats, conserving the total enthalpy """

    # with external stellar radiation
    if quant.T_star >= 10:

        # dampara as in damping parameter. It is a try to play with words.
        quant.dampara = 16  # stable: 512. Other numbers like 16, 32, 64, 128 also possible, but may prove unstable in terms of convergence

        # allows to correct for global equilibrium. With fudge_factor == 1, you satisfy local equilibrium, but never reach a global one
        if quant.F_intern != 0:
            fudge_factor = (100 * quant.F_intern / (quant.F_net[quant.ninterface - 21] + 99 * quant.F_intern)) ** (1.0 / quant.dampara)
        elif quant.F_intern == 0:
            fudge_factor = (quant.F_down_tot[quant.ninterface - 21] / quant.F_up_tot[quant.ninterface - 21]) ** (1.0 / quant.dampara)

    # same procedure without a stellar energy source
    else:
        quant.dampara = max(quant.iter_value, 4096.0)  # value of 4096 found to be stable...
        # careful! too small values (e.g. < 128) may cause irregularities in the convergence

        fudge_factor = (quant.F_intern / quant.F_net[quant.ninterface - 1]) ** (1.0 / quant.dampara)

    unstable_list = []
    start_layers = []
    end_layers = []

    for i in range(quant.nlayer):
        if quant.conv_unstable[i] == 1:
            unstable_list.append(i)

    for i in range(len(unstable_list)):
        if unstable_list[i]-1 not in unstable_list:
            start_layers.append(unstable_list[i])
        if unstable_list[i]+1 not in unstable_list:
            end_layers.append(unstable_list[i])

    if len(start_layers) != len(end_layers):
        print("Error in convective calculation. Aborting...")
        raise SystemExit()

    for n in range(len(start_layers)):

        num = 0
        denom = 0

        for i in range(start_layers[n], end_layers[n]+1):

            num += quant.c_p_lay[i] * quant.T_lay[i] * (quant.p_int[i] - quant.p_int[i+1])

            denom_element = 1

            if i != start_layers[n]:

                for j in range(start_layers[n], i):

                    denom_element *= (quant.p_lay[j]/quant.p_int[j])**quant.kappa_int[j] * (quant.p_int[j+1]/quant.p_lay[j])**quant.kappa_lay[j]

            denom_element *= (quant.p_lay[i]/quant.p_int[i])**quant.kappa_int[i] * quant.c_p_lay[i] * (quant.p_int[i] - quant.p_int[i+1])

            denom += denom_element

        mean_pot_temp = num / denom

        mean_pot_temp *= fudge_factor

        for i in range(start_layers[n], end_layers[n] + 1):

            factor = 1

            if i != start_layers[n]:

                for j in range(start_layers[n], i):

                    factor *= (quant.p_lay[j]/quant.p_int[j])**quant.kappa_int[j] * (quant.p_int[j+1]/quant.p_lay[j])**quant.kappa_lay[j]

            factor *= (quant.p_lay[i]/quant.p_int[i])**quant.kappa_int[i]

            quant.T_lay[i] = mean_pot_temp * factor


def convective_adjustment(quant):
    """ adjusts the atmosphere for convective equilibrium. """

    do_check = 1
    while do_check:

        conv_check(quant)

        do_check = sum(quant.conv_unstable) > 0

        if do_check:

            conv_correct(quant)


def mark_convective_layers(quant):
    """ marks the layers where convection dominates over radiative transfer """

    quant.conv_layer = np.zeros(quant.nlayer, np.int32)

    for i in range(quant.nlayer - 1):

        T_in_between_lim = quant.T_lay[i] * (quant.p_int[i + 1] / quant.p_lay[i]) ** (quant.kappa_lay[i] * (1 - 1e-6))

        T_ad_lim = T_in_between_lim * (quant.p_lay[i + 1] / quant.p_int[i + 1]) ** (quant.kappa_int[i + 1] * (1 - 1e-6))

        if quant.T_lay[i+1] < T_ad_lim:
            quant.conv_layer[i] = 1
            quant.conv_layer[i+1] = 1
        else:
            quant.conv_layer[i + 1] = 0


def calculate_conv_flux(quant):
    """ calculates the convective net flux in the atmosphere. """

    quant.F_net_conv = np.zeros(quant.ninterface, quant.fl_prec)

    if quant.F_intern > 0:

        for i in range(0, quant.nlayer):

            quant.F_net_conv[i] = quant.F_intern - quant.F_net[i]

            # nullifies Fnet if very small. This helps to distinguish between radiative and convective layers
            if quant.F_net[i] != 0:
                if quant.F_net_conv[i] / quant.F_net[i] < 0.01:

                    quant.F_net_conv[i] = 0


def calc_F_ratio(quant):
    """ calculates the planet to star flux ratio for sec. eclipse data interpretation """

    if quant.T_star > 1:
        orbital_factor = (quant.R_planet / quant.R_star) ** 2

        for x in range(quant.nbin):

            # original means here: without the energy correction factor
            original_star_BB_flux = np.pi * quant.planckband_lay[quant.nlayer + x * (quant.nlayer+2)] / quant.star_corr_factor

            if original_star_BB_flux != 0:
                ratio = orbital_factor * quant.F_up_band[x + quant.nlayer * quant.nbin] / original_star_BB_flux
            else:
                ratio = 0

            quant.F_ratio.append(ratio)


def calculate_height_z(quant):
    """ prints the message that you have been desperately waiting for """

    # rocky planets: white light radius is the surface
    i_white_light_radius = 0

    # gas planets with pressures of more than 10 bar: white light radius at 10 bar
    if quant.p_lay[0] >= 1e7:
        i_white_light_radius = max([i for i in range(quant.nlayer) if quant.p_lay[i] >= 1e7])

    quant.z_lay[i_white_light_radius] = 0

    # calculates the height of layers above and the height of layers below z = 0
    for i in range(i_white_light_radius + 1, quant.nlayer):

        quant.z_lay[i] = quant.z_lay[i-1] + 0.5 * quant.delta_z_lay[i-1] + 0.5 * quant.delta_z_lay[i]

    for i in range(i_white_light_radius - 1, 0 - 1, -1):

        quant.z_lay[i] = quant.z_lay[i + 1] - 0.5 * quant.delta_z_lay[i + 1] - 0.5 * quant.delta_z_lay[i]


def success_message(quant):
    """ prints the message that you have been desperately waiting for """

    T_eff_global, T_eff_dayside, T_eff_model, T_star_brightness, T_planet_brightness = temp_calcs(quant)

    print("\nDone! Everything appears to have worked fine :-)\n")

    run_type = "an iterative" if quant.singlewalk == 0 else "a post-processing"

    print("This has been " + run_type + " run with name " + quant.name + ".\n")

    print("\nFinal Check for numerical energy balance:")
    print("\tTheoretical effective temperature of planet. global (f=0.25): {:g} K,".format(T_eff_global),
          "day-side (f=2/3): {:g} K,".format(T_eff_dayside), "\n\tused in model ({:.2f}): {:g} K.".format(quant.f_factor, T_eff_model))
    print("\tIncident TOA brightness temperature: {:g} K, Interior temperature: {:g} K.\n".format(T_star_brightness, quant.T_intern),
          "\tOutgoing (planetary) brightness temperature: {:g} K.".format(T_planet_brightness))

    if quant.T_intern != 0:
        relative_energy_imbalance = (quant.F_intern - quant.F_net[quant.ninterface-1])/quant.F_intern
    else:
        relative_energy_imbalance = -quant.F_net[quant.ninterface - 1] / quant.F_down_tot[quant.ninterface - 1]
    print("--> Global energy imbalance: {:.2e} (positive: too much uptake, negative: too much loss).".format(relative_energy_imbalance), "\n")


if __name__ == "__main__":
    print("This module stores the definitions for the functions living on the host. "
          "It is spacier on the host than on the device but also warmer.")