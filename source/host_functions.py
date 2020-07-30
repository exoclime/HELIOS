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
import sys
from numpy.polynomial.legendre import leggauss as G
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from source import phys_const as pc
from source import realtime_plotting as rtp


def planet_param(quant, read):
    """ sets the planetary, stellar and orbital parameters and converts them to the correct units """

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
    # avoiding T_star = 0 to prevent numerical issues -- taking the CMB radiation BB temperature instead
    quant.T_star = quant.fl_prec(max(quant.T_star, 2.3))


def approx_f_from_formula(quant, read):
    """ calculates the f redistribution factor from the approximative formula from Koll et al. (2019) """

    # read in tau_lw from output file if it exists
    if "_post" in quant.name:
        name = quant.name[:-5]
    else:
        name = quant.name
    try:
        with open(read.output_path + name + "/" + name + "_tau_lw_sw.dat", "r") as entr_file:
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

    term = quant.tau_lw * (quant.p_boa / 1e6) ** (2 / 3) * (T_eq / 600) ** (-4 / 3)

    quant.f_factor = 2 / 3 - 5 / 12 * term / (2 + term)


def calc_planck(lamda, temp):
    """ calculates the Planckian blackbody function at a given wavelength and temperature """

    term1 = 2 * pc.H * pc.C**2 / lamda**5

    term2 = np.exp(pc.H * pc.C / (lamda * pc.K_B * temp)) - 1

    result = term1 * 1 / term2

    return result


def calc_tau_lw_sw(quant, read):
    """ estimates the shortwave and longwave optical depths from the TOA to BOA. This is required when using the f approximation formula of Koll et al. (2019) """

    num_lw = 0
    denom_lw = 0
    num_sw = 0
    denom_sw = 0

    for x in range(quant.nbin):

        tau_from_top = 0

        for i in range(quant.nlayer):

            tau_from_top += quant.delta_tau_band[x + i * quant.nbin]

        B_surface = calc_planck(quant.opac_wave[x], quant.T_lay[quant.nlayer])

        num_lw += B_surface * np.exp(-tau_from_top) * quant.opac_deltawave[x]
        denom_lw += B_surface * quant.opac_deltawave[x]

        if quant.T_star > 10:
            B_star = calc_planck(quant.opac_wave[x], quant.T_star)

            num_sw += B_star * np.exp(-tau_from_top) * quant.opac_deltawave[x]
            denom_sw += B_star * quant.opac_deltawave[x]

    tau_lw_tot = -np.log(num_lw / denom_lw)

    if quant.T_star > 10:
        tau_sw_tot = -np.log(num_sw / denom_sw)
    else:
        tau_sw_tot = 0

    with open(read.output_path + quant.name + "/" + quant.name + "_tau_lw_sw.dat", "w") as file:
        file.writelines("This file contains the total longwave and shortwave optical depths at BOA (or surface if there), tau_lw and tau_sw")
        file.writelines("\n{:<10}{:<10}".format("tau_lw", "tau_sw"))
        file.writelines("\n{:<10g}{:<10g}".format(tau_lw_tot, tau_sw_tot))


def initial_temp(quant, read, Vmod):
    """ determines the initial temperature profile """

    if quant.singlewalk == 0:

        # calculate effective planetary temperature
        T_eff = (1.0-quant.dir_beam) * quant.f_factor ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star \
                + quant.dir_beam * abs(quant.mu_star) ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star

        # for efficiency reasons initial temperature has a lower limit of 500 K
        quant.T_lay = np.ones(quant.nlayer+1) * max(T_eff, 500)

        print("\nStarting with an isothermal TP-profile at {:g}".format(max(T_eff, 500))+" K.")

    elif quant.singlewalk == 1:

        read.read_temperature_file(quant)

        quant.T_lay = np.append(quant.T_restart[1:], quant.T_restart[0])

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
    """ calculates the internal heat flux """

    quant.F_intern = pc.SIGMA_SB * quant.T_intern ** 4.0


def set_up_numerical_parameters(quant):
    """ sets up additional parameters used in the calculation """

    # limit for w_0
    quant.w_0_limit = quant.fl_prec(1.0 - 1e-10)

    # limit where to switch from noniso to iso equations to keep model stable
    quant.delta_tau_limit = quant.fl_prec(1e-4)

    # relative criterion for global energy equilibrium
    quant.global_limit = quant.fl_prec(1e-3)

    # relative criterion for local energy equilibrium
    if quant.prec == "double":
        quant.local_limit_rad_iter = quant.fl_prec(1e-7)
    elif quant.prec == "single":
        quant.local_limit_rad_iter = quant.fl_prec(1e-5)

    # relative criterion for local energy equilibrium in the convective adjustment part
    quant.local_limit_conv_iter = quant.fl_prec(1e-7)

    # sets the appropriate gaussian weights
    quant.gauss_weight = G(quant.ny)[1]


def check_for_global_eq(quant):
    """ checks for global equilibrium """

    criterion = 0

    conv_layer_list = []
    start_conv_layers = []
    end_conv_layers = []

    for i in range(quant.nlayer):
        if quant.conv_layer[i] == 1:
            conv_layer_list.append(i)

    for i in range(len(conv_layer_list)):
        if conv_layer_list[i] - 1 not in conv_layer_list:
            start_conv_layers.append(conv_layer_list[i])
        if conv_layer_list[i] + 1 not in conv_layer_list:
            end_conv_layers.append(conv_layer_list[i])

    criterion_list = np.zeros(len(start_conv_layers))

    lim_quant = np.zeros(len(start_conv_layers))

    for n in range(len(start_conv_layers)):
        #print(n, len(quant.start_layers))
        # interface to adjust the radiative net flux to
        interface_to_be_tested = quant.ninterface - 1

        if n != len(start_conv_layers) - 1:
            interface_to_be_tested = int((end_conv_layers[n] + start_conv_layers[n + 1]) / 2)

        if quant.T_intern != 0:  # with internal heat
            lim_quant[n] = abs(quant.F_intern - quant.F_net[interface_to_be_tested]) / quant.F_intern
        elif quant.T_intern == 0:  # without internal heat. in this case there must be a star
            lim_quant[n] = abs(quant.F_net[interface_to_be_tested]) / quant.F_down_tot[interface_to_be_tested]

        # user feedback -- only once per iterstep
        if quant.iter_value % 100 == 0:
            if n == len(start_conv_layers) - 1:
                print("  The relative difference between TOA and BOA net flux is : {:.2e}".format(lim_quant[n])
                      + " and should be less than {:.2e}".format(quant.global_limit) + ".")
            else:
                print("  The relative net flux difference at intermediate layer "
                      "#{:g} ({:g}/{:g}) is : {:.2e}".format(interface_to_be_tested, n, len(start_conv_layers)-2, lim_quant[n])
                      + " and should be less than {:.2e}".format(quant.global_limit) + ".")

        if lim_quant[n] < quant.global_limit:
            criterion_list[n] = 1

    if sum(criterion_list) == len(criterion_list):

        criterion = 1

    return criterion


def relax_global_limit(quant):
    """ makes the global convergence limit less strict for difficult cases """

    quant.global_limit = quant.fl_prec(1e-2)


def check_for_local_eq(quant):
    """ checks for local equilibrium """

    criterion = 0

    quant.converged = np.zeros(quant.nlayer+1)
    quant.marked_red = np.zeros(quant.nlayer+1)

    for i in range(quant.nlayer+1):  # including surface/BOA "ghost layer"

        if quant.conv_layer[i] == 0:

            if i < quant.nlayer:

                # temperature smoothing (analogous to rad_temp_iter kernel)
                t_mid = quant.T_lay[i]

                if quant.smooth == 1:
                    if quant.p_lay[i] < 1e6 and i < quant.nlayer - 1:
                        t_mid = (quant.T_lay[i - 1] + quant.T_lay[i + 1]) / 2

                F_temp = (t_mid - quant.T_lay[i])**7

                combined_F_net = quant.F_net_diff[i] + F_temp

            elif i == quant.nlayer:

                combined_F_net = quant.F_intern - quant.F_net[0]

            if quant.T_lay[i] == 0:
                print(i ,quant.T_lay[i])

            div_lim_quant = abs(combined_F_net) / (pc.SIGMA_SB * quant.T_lay[i] ** 4.0)

            # check for criterion satisfaction
            if div_lim_quant < quant.local_limit_conv_iter:
                quant.converged[i]=1
            else:
                quant.marked_red[i] = 1
                # uncomment for debugging
                # if quant.iter_value % 100 == 99:
                #     print("layer: {:<5g}, delta_flux/BB_layer: {:<12.3e}, delta_flux: {:<12.3e}, BB: {:<12.3e}".format(i, div_lim_quant, abs(combined_F_net), pc.SIGMA_SB * quant.T_lay[i] ** 4.0))

    if sum(quant.converged) == (quant.nlayer + 1) - sum(quant.conv_layer):
        criterion = 1

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
    quant.conv_unstable = np.zeros(quant.nlayer + 1, np.int32)  # including the surface/BOA "ghost layer"

    for i in range(quant.nlayer-1):

        if quant.p_lay[i] <= 1e2:  # ignore top atmosphere, since artificial/numerical temperature peaks might occur there
            break

        T_in_between_lim = quant.T_lay[i] * (quant.p_int[i+1] / quant.p_lay[i]) ** (quant.kappa_lay[i] * (1+1e-6))

        T_ad_lim = T_in_between_lim * (quant.p_lay[i+1] / quant.p_int[i+1]) ** (quant.kappa_int[i+1] * (1+1e-6))

        if quant.T_lay[i+1] < T_ad_lim:

            quant.conv_unstable[i] = 1
            quant.conv_unstable[i+1] = 1

    # do the surface/BOA condition
    T_ad_lim = quant.T_lay[quant.nlayer] * (quant.p_lay[0] / quant.p_int[0]) ** (quant.kappa_int[0] * (1 + 1e-6))

    if quant.T_lay[0] < T_ad_lim:
        quant.conv_unstable[quant.nlayer] = 1
        quant.conv_unstable[0] = 1

    # for debugging purposes uncomment next line
    # print("unstable layers:", i, i+1, quant.T_lay[i], quant.T_lay[i+1])


def conv_correct(quant, fudging):
    """ corrects unstable lapse rates to dry adiabats, conserving the total enthalpy """

    to_be_corrected_list = []
    start_layers = []
    end_layers = []

    for i in range(quant.nlayer+1):  # including surface/BOA "ghost layer"
        if quant.conv_unstable[i] == 1 or quant.conv_layer[i] == 1:
            to_be_corrected_list.append(i)

    if quant.nlayer in to_be_corrected_list:
        to_be_corrected_list = np.insert(to_be_corrected_list[:-1], 0, -1)

    for i in range(len(to_be_corrected_list)):
        if to_be_corrected_list[i]-1 not in to_be_corrected_list:
            start_layers.append(to_be_corrected_list[i])
        if to_be_corrected_list[i]+1 not in to_be_corrected_list:
            end_layers.append(to_be_corrected_list[i])

    # quick self-check
    if len(start_layers) != len(end_layers):
        print("Error in convective calculation. Aborting...")
        raise SystemExit()

    fudge_factor = np.ones(len(start_layers))

    if fudging == 1:

        for n in range(len(start_layers)):

            # interface to adjust the radiative net flux to
            # interface_to_be_tested = quant.ninterface - 1

            for m in range(n, len(start_layers)):

                if m != len(start_layers) - 1:

                    p_top = quant.p_lay[start_layers[m+1]]
                    if end_layers[m] != -1:
                        p_bot = quant.p_lay[end_layers[m]]
                    elif end_layers[m] == -1:
                        p_bot = quant.p_int[0]

                    if (p_top / p_bot) < (1 / np.e):  # (=H) avoiding small RT zones of width < H
                        interface_to_be_tested = int((end_layers[m] + start_layers[m+1]) / 2)
                        break

                else: # toplayers

                    interface_to_be_tested = int(0.8 * end_layers[m] + 0.2 * (quant.ninterface - 1))

            ####### OLD CHUNK (keep for the moment in case new version fails somehow) ############################
            # # with external stellar radiation
            # if quant.T_star >= 10:
            #
            #     # dampara as in damping parameter. It is a sad attempt at playing with words.
            #     if quant.input_dampara == "auto":
            #         quant.dampara = 16  # 16 is found to lead to the fastest convergence hence taking as nominal value
            #     else:
            #         quant.dampara = int(quant.input_dampara)
            #
            #     # allows to correct for global equilibrium. With fudge_factor == 1, you satisfy local equilibrium, but never reach a global one
            #     if quant.T_intern > 0:
            #         fudge_factor[n] = (100 * quant.F_intern / (max(0,quant.F_net[interface_to_be_tested]) + 99 * quant.F_intern)) ** (1.0 / quant.dampara)
            #     else:
            #         fudge_factor[n] = (quant.F_down_tot[interface_to_be_tested] / quant.F_up_tot[interface_to_be_tested]) ** (1.0 / quant.dampara)
            #
            #     # uncomment for debugging
            #     # print("F_intern: {:.2e}, F_net_TOA: {:.2e}".format(quant.F_intern, quant.F_net[interface_to_be_tested]))
            #
            # # same procedure without a stellar energy source
            # else:
            #
            #     if quant.input_dampara == "auto":
            #         quant.dampara = 1024  # 1024 is found to lead to the fastest convergence hence taking as nominal value
            #     else:
            #         quant.dampara = int(quant.input_dampara)
            #
            #     fudge_factor[n] = (quant.F_intern / quant.F_net[interface_to_be_tested]) ** (1.0 / quant.dampara)
            ###############################################

            if quant.input_dampara == 'adaptive':

                if quant.iter_value < 1000:

                    quant.dampara = 4

                else:

                    if quant.iter_value % 5 == 0:

                        if n < len(start_layers) - 1:

                            local_convergence = ((start_layers[n+1]-1) - end_layers[n]) == sum(quant.converged[end_layers[n]+1:start_layers[n+1]])

                            if local_convergence:

                                quant.dampara /= 1.05

                            else:

                                quant.dampara *= 1.001

                            # comment out for debugging
                            # print(n, local_convergence, quant.dampara)

            elif quant.input_dampara == "auto":

                quant.dampara = 4

                # boost for very bottom
                # proceeds from 2**2 to 2**-2 as time goes by
                if n == 0:
                    quant.dampara = 2 ** max(-2, (- 1 / 500 * max(0, quant.iter_value - 1000) + 2))

                    # comment out for debugging
                    # if quant.iter_value % 10 == 0:
                    #     print(quant.dampara)

            else:
                quant.dampara = float(quant.input_dampara)

            if n < len(start_layers) - 1:
                fudge_factor[n] = ((quant.F_intern + quant.F_down_tot[interface_to_be_tested]) / quant.F_up_tot[interface_to_be_tested]) ** (1.0 / quant.dampara)
            else:
                fudge_factor[n] = ((quant.F_intern + quant.F_down_tot[interface_to_be_tested]) / quant.F_up_tot[interface_to_be_tested]) ** (1.0 / 4.0)

            fudge_factor[n] = min(1.01, max(0.99, fudge_factor[n]))  # to prevent instabilities

        ### uncomment next few lines for debugging
        # if quant.iter_value % 100 == 0:
        #     for n in range(len(start_layers)):
        #         if n < len(start_layers) - 1:
        #             print("\tIntermediate RT layers:", end_layers[n],"-", start_layers[n+1], "fudge_factor=",
        #                   fudge_factor[n], ". pressure ratio.:", quant.p_lay[start_layers[n+1]]/quant.p_lay[end_layers[n]])
        #         else:
        #             print("\tTop RT layer fudge_factor= ", fudge_factor[n], ".")

    for n in range(len(start_layers)):

        num = 0
        denom = 0

        # next two lines to prevent having index -1, which could happen if surface/BOA is in the to_be_corrected_list
        start_index = max(0, start_layers[n])
        stop_index = max(0, end_layers[n])

        for i in range(start_index, stop_index + 1):

            num += quant.c_p_lay[i] / (quant.meanmolmass_lay[i] * pc.N_A) * quant.T_lay[i] * (quant.p_int[i] - quant.p_int[i+1]) # converting c_p from "per mole" to "per gram"

            denom_element = 1

            if i != start_index:

                for j in range(start_index, i):

                    denom_element *= (quant.p_lay[j]/quant.p_int[j])**quant.kappa_int[j] * (quant.p_int[j+1]/quant.p_lay[j])**quant.kappa_lay[j]

            denom_element *= (quant.p_lay[i]/quant.p_int[i])**quant.kappa_int[i] * quant.c_p_lay[i] / (quant.meanmolmass_lay[i] * pc.N_A) * (quant.p_int[i] - quant.p_int[i+1])

            denom += denom_element

        mean_pot_temp = num / denom

        mean_pot_temp *= fudge_factor[n]

        for i in range(start_index, stop_index + 1):

            factor = 1

            if i != start_index:

                for j in range(start_index, i):

                    factor *= (quant.p_lay[j]/quant.p_int[j])**quant.kappa_int[j] * (quant.p_int[j+1]/quant.p_lay[j])**quant.kappa_lay[j]

            factor *= (quant.p_lay[i]/quant.p_int[i])**quant.kappa_int[i]

            quant.T_lay[i] = mean_pot_temp * factor

        # correct surface/BOA temperature to the convective zones mean potential temperature to satisfy stability
        if start_layers[n] == -1:

            quant.T_lay[quant.nlayer] = mean_pot_temp


def convective_adjustment(quant):
    """ adjusts the atmosphere for convective equilibrium. """

    iter = 0

    conv_check(quant)
    unstable_found = sum(quant.conv_unstable) > 0

    while unstable_found:

        ### uncomment for debugging
        # sys.stdout.write("Adjusting temperatures: {:6d} \r".format(iter))
        # sys.stdout.flush()

        mark_convective_layers(quant, stitching=0)

        conv_correct(quant, fudging=0)

        conv_check(quant)
        unstable_found = sum(quant.conv_unstable) > 0

        ### uncomment next three lines for debugging
        # check_for_local_eq(quant)
        # mark_convective_layers(quant, stitching=0)
        # rtp.Plot().plot_convective_feedback(quant)

        iter += 1

    mark_convective_layers(quant, stitching=1)

    conv_correct(quant, fudging=1)

    ### uncomment for debugging
    # sys.stdout.write("Adjusting temperatures: {:6s}\r".format("DONE"))
    # sys.stdout.flush()


def mark_convective_layers(quant, stitching):
    """ marks the layers where convection dominates over radiative transfer """

    # reset bottom boundary
    quant.conv_layer[quant.nlayer] = 0
    quant.conv_layer[0] = 0

    for i in range(quant.nlayer - 1):

        if quant.p_lay[i] <= 1e2:  # ignore top atmosphere, since artificial/numerical temperature peaks might occur there
            break

        T_in_between_lim = quant.T_lay[i] * (quant.p_int[i + 1] / quant.p_lay[i]) ** (quant.kappa_lay[i] * (1 - 1e-6))

        T_ad_lim = T_in_between_lim * (quant.p_lay[i + 1] / quant.p_int[i + 1]) ** (quant.kappa_int[i + 1] * (1 - 1e-6))

        if quant.T_lay[i+1] < T_ad_lim:
            quant.conv_layer[i] = 1
            quant.conv_layer[i+1] = 1
        else:
            quant.conv_layer[i + 1] = 0

    # do the surface/BOA condition
    T_ad_lim = quant.T_lay[quant.nlayer] * (quant.p_lay[0] / quant.p_int[0]) ** (quant.kappa_int[0] * (1 - 1e-6))

    if quant.T_lay[0] < T_ad_lim:
        quant.conv_layer[quant.nlayer] = 1
        quant.conv_layer[0] = 1

    # stitch holes if taking too long to converge
    if stitching == 1:
        if quant.iter_value > 5e3:  # warning: hardcoded number
            stitching_convective_zone_holes(quant)


def stitching_convective_zone_holes(quant):
    """ fills up holes in the convective zones if they are preventing convergence """

    start_layers = []
    end_layers = []

    for i in range(quant.nlayer):

        if quant.conv_layer[i] == 1:

            if i > 0:
                if quant.conv_layer[i-1] == 0:
                    start_layers.append(i)
            elif i == 0:
                if quant.conv_layer[quant.nlayer] == 0:
                    start_layers.append(i)

            if i < quant.nlayer - 1:
                if quant.conv_layer[i+1] == 0:
                    end_layers.append(i)
            elif i == quant.nlayer - 1:
                end_layers.append(i)

    # do surface/BC "ghost layer"
    if quant.conv_layer[quant.nlayer] == 1:

        start_layers.append(quant.nlayer)
        start_layers = np.insert(start_layers[:-1], 0, -1)

        if quant.conv_layer[0] == 0:
            end_layers.append(quant.nlayer)
            end_layers = np.insert(end_layers[:-1], 0, -1)

    # quick self-check
    if len(start_layers) != len(end_layers):
        print("Error in stitching calculation. Aborting...")
        raise SystemExit()

    for n in range(len(start_layers)-1):

        p_top = quant.p_lay[start_layers[n+1]]
        if end_layers[n] != - 1:
            p_bot = quant.p_lay[end_layers[n]]
        elif end_layers[n] == - 1:
            p_bot = quant.p_int[0]

        if (p_top / p_bot) > (1 / np.e):  # (=H) stitching small RT zones of width < H

            for m in range(end_layers[n]+1, start_layers[n+1]):

                quant.conv_layer[m] = 1


def calculate_conv_flux(quant):
    """ calculates the convective net flux in the atmosphere. """

    quant.F_net_conv = np.zeros(quant.ninterface, quant.fl_prec)

    if quant.F_intern > 0:

        for i in range(0, quant.nlayer):

            quant.F_net_conv[i] = quant.F_intern - quant.F_net[i]

            # nullifies Fnet if very small. This helps to distinguish between radiative and convective layers
            if quant.F_net[i] != 0:
                if quant.F_net_conv[i] / quant.F_net[0] < 0.01:

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
    """ calculates the altitude of the layer centers, either above ground or 10 bar pressure level """

    if quant.planet_type == 'gas':

        # gas planets with pressures of more than 10 bar: white light radius at 10 bar
        i_white_light_radius = max([i for i in range(quant.nlayer) if quant.p_lay[i] >= 1e7])

        quant.z_lay[i_white_light_radius] = 0

        # calculates the height of layers above and the height of layers below z = 0
        for i in range(i_white_light_radius + 1, quant.nlayer):

            quant.z_lay[i] = quant.z_lay[i-1] + 0.5 * quant.delta_z_lay[i-1] + 0.5 * quant.delta_z_lay[i]

        for i in range(i_white_light_radius - 1, 0 - 1, -1):

            quant.z_lay[i] = quant.z_lay[i + 1] - 0.5 * quant.delta_z_lay[i + 1] - 0.5 * quant.delta_z_lay[i]

    elif quant.planet_type == 'rocky':

        quant.z_lay[0] = 0.5 * quant.delta_z_lay[0]

        for i in range(1, quant.nlayer):

            quant.z_lay[i] = quant.z_lay[i-1] + 0.5 * quant.delta_z_lay[i-1] + 0.5 * quant.delta_z_lay[i]


def construct_grid(quant):

    press_levels = [quant.p_boa * (quant.p_toa/quant.p_boa)**(i/(2 * quant.nlayer - 1)) for i in range(2 * quant.nlayer)]

    quant.p_lay = [press_levels[i] for i in range(1, 2 * quant.nlayer, 2)]

    quant.p_int = [press_levels[i] for i in range(0, 2 * quant.nlayer, 2)]

    quant.p_int.append(quant.p_toa * (quant.p_toa/quant.p_boa)**(1/(2 * quant.nlayer - 1)))

    for i in range(quant.nlayer):

        quant.delta_colmass.append((quant.p_int[i] - quant.p_int[i + 1]) / quant.g)
        quant.delta_col_upper.append((quant.p_lay[i] - quant.p_int[i + 1]) / quant.g)
        quant.delta_col_lower.append((quant.p_int[i] - quant.p_lay[i]) / quant.g)


def success_message(quant):
    """ prints the message that you have been desperately waiting for """

    T_eff_global, T_eff_dayside, T_eff_model, T_star_brightness, T_planet_brightness = temp_calcs(quant)

    print("\nDone! Everything appears to have worked fine :-)\n")

    run_type = "an iterative" if quant.singlewalk == 0 else "a post-processing"

    print("This has been " + run_type + " run with name " + quant.name + ".\n")

    print("\nFinal Check for numerical energy balance:")
    print("  --> Theoretical effective temperature of planet: \n\tglobal (f=0.25): {:g} K,".format(T_eff_global),
          "\n\tday-side (f=2/3): {:g} K,".format(T_eff_dayside), "\n\tused in model (f={:.2f}): {:g} K.".format(quant.f_factor, T_eff_model))
    print("  --> Incident TOA brightness temperature: {:g} K \n      Interior temperature: {:g} K".format(T_star_brightness, quant.T_intern),
          "\n      Outgoing (planetary) brightness temperature: {:g} K".format(T_planet_brightness))

    relative_energy_imbalance = (quant.F_intern - quant.F_net[quant.ninterface - 1]) / quant.F_up_tot[quant.ninterface - 1]

    print("  --> Global energy imbalance: {:.1f}ppm (positive: too much uptake, negative: too much loss).".format(relative_energy_imbalance*1e6), "\n")


if __name__ == "__main__":
    print("This module stores the definitions for the functions living on the host. "
          "It is spacier on the host than on the device but also warmer.")