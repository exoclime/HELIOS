# ==============================================================================
# Module for host function definitions that are not strictly read or write related
# Copyright (C) 2018 - 2022 Matej Malik
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
import math
from numpy.polynomial.legendre import leggauss as G
from scipy import interpolate
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from source import phys_const as pc


def planet_param(quant, read):
    """ sets the planetary, stellar and orbital parameters and converts them to the correct units """

    if quant.planet == "manual":
        print("\nUsing manual input for the planetary and orbital parameters.")
    else:
        read.read_planet_database(quant)

    # convert to cgs units
    if quant.g < 10:
        quant.g = quant.fl_prec(10 ** quant.g)
    quant.a = quant.fl_prec(quant.a * pc.AU)
    quant.R_planet = quant.fl_prec(quant.R_planet * pc.R_JUP)
    quant.R_star = quant.fl_prec(quant.R_star * pc.R_SUN)
    # avoiding T_star = 0 to prevent numerical issues -- taking the CMB radiation BB temperature instead
    quant.T_star = quant.fl_prec(max(quant.T_star, 2.7))


def approx_f_from_formula(quant, read):
    """ calculates the f redistribution factor with Eq. (10) in Koll (2021) """

    # read in tau_lw from output file if it exists
    if "_post" in quant.name:
        name = quant.name[:-5]
    else:
        name = quant.name
    try:
        with open(read.output_path + name + "/" + name + "_tau_lw_tau_sw_f_factor.dat", "r") as entr_file:
            next(entr_file)
            next(entr_file)
            for line in entr_file:
                column = line.split()
                quant.tau_lw = float(column[0])
        print("\ntau_lw read in from previous output file!")
        print("\ntau_lw = ", quant.tau_lw)

    except IOError:
        print("\nWarning: Unable to read in tau_lw from file. Using either commandline values or starting from 1 per default.")

    # calculates the f factor via Eq.(10) in Koll (2021)
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

    # trying Koll calculation
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

    # if too large values, going alternatively withouth the exponential
    if math.isinf(tau_lw_tot):

        for x in range(quant.nbin):

            tau_from_top = 0

            for i in range(quant.nlayer):

                tau_from_top += quant.delta_tau_band[x + i * quant.nbin]

            B_surface = calc_planck(quant.opac_wave[x], quant.T_lay[quant.nlayer])

            num_lw += B_surface * tau_from_top * quant.opac_deltawave[x]
            denom_lw += B_surface * quant.opac_deltawave[x]

            if quant.T_star > 10:
                B_star = calc_planck(quant.opac_wave[x], quant.T_star)

                num_sw += B_star * tau_from_top * quant.opac_deltawave[x]
                denom_sw += B_star * quant.opac_deltawave[x]

        tau_lw_tot = num_lw / denom_lw

        if quant.T_star > 10:

            tau_sw_tot = num_sw / denom_sw
        else:
            tau_sw_tot = 0

    with open(read.output_path + quant.name + "/" + quant.name + "_tau_lw_tau_sw_f_factor.dat", "w") as file:
        file.writelines("This file contains the total longwave and shortwave optical depths at BOA (=surface), tau_lw and tau_sw, and the f factor as used in the model")
        file.writelines("\n{:<15}{:<15}{:<15}".format("tau_lw", "tau_sw", "f_factor"))
        file.writelines("\n{:<15g}{:<15g}{:<15g}".format(tau_lw_tot, tau_sw_tot, quant.f_factor))


def initial_temp(quant, read):
    """ determines the initial temperature profile """

    if quant.singlewalk == 0 and (quant.force_start_tp_from_file == 0 or quant.physical_tstep == 0):

        # calculate effective planetary temperature
        T_eff = (1.0-quant.dir_beam) * quant.f_factor ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star \
                + quant.dir_beam * abs(quant.mu_star) ** 0.25 * (quant.R_star / quant.a) ** 0.5 * quant.T_star

        # for efficiency reasons initial temperature has a lower limit of 500 K
        quant.T_lay = np.ones(quant.nlayer+1) * max(T_eff, 500)

        print("\nStarting with an isothermal TP-profile at {:g}".format(max(T_eff, 500))+" K.")

    elif quant.singlewalk == 1 or (quant.force_start_tp_from_file == 1 and quant.physical_tstep != 0):

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

    # setting upper limit for w_0 because equations are not valid for w_0 = 1
    quant.w_0_limit = quant.fl_prec(1.0 - 1e-10)

    # setting limit where to switch from flux matrix method with scattering to pure absorption equations
    quant.w_0_scat_limit = quant.fl_prec(1e-3)

    # limit where to switch from noniso to iso equations to keep model stable in the top atmosphere
    quant.delta_tau_limit = quant.fl_prec(1e-4)

    # sets the appropriate gaussian weights
    quant.gauss_weight = G(quant.ny)[1]


### to be implemented in future ###
# def include_sensible_heat_flux(quant):
#
#     # copy arrays to host
#     quant.F_up_tot = quant.dev_F_up_tot.get()
#     quant.F_net = quant.dev_F_net.get()
#     quant.T_lay = quant.dev_T_lay.get()
#
#     calc_sensible_heat_flux(quant)
#
#     quant.F_up_tot[0] += quant.F_sens
#     quant.F_net[0] += quant.F_sens
#
#     # copy arrays to device
#     quant.dev_F_up_tot = gpuarray.to_gpu(quant.F_up_tot)
#     quant.dev_F_net = gpuarray.to_gpu(quant.F_net)


def relax_radiative_convergence_criterion(quant):
    """ makes the radiative convergence criterion less strict over time """

    quant.rad_convergence_limit *=10.0

    quant.relaxed_criterion_trigger = 1


def check_for_radiative_eq(quant):
    """ checks for local equilibrium during the convective iteration """

    criterion = 0

    quant.converged = np.zeros(quant.nlayer+1, np.int32)
    quant.marked_red = np.zeros(quant.nlayer+1, np.int32)

    for i in range(quant.nlayer+1):  # including surface/BOA "ghost layer"

        if quant.T_lay[i] == 0:  # just checking whether temperatures reach zero somewhere, which they obviously shouldn't
            print("WARNING WARNING WARNING: Found zero temperature at layer:", i, quant.T_lay[i])

        if quant.conv_layer[i] == 0:

            if i < quant.nlayer:

                local_F_net_diff = abs(quant.F_intern + quant.F_add_heat_sum[i] + quant.F_smooth_sum[i] - quant.F_net[i+1])

            elif i == quant.nlayer:

                local_F_net_diff = abs(quant.F_intern - quant.F_net[0])

            # check for criterion satisfaction
            if local_F_net_diff < quant.rad_convergence_limit * (quant.F_down_tot[quant.nlayer] + quant.F_intern):
                quant.converged[i] = 1
            else:
                quant.marked_red[i] = 1

    if quant.iter_value % 100 == 1:
        print("Number of radiative layers converged: {:d} out of {:d}.".format(int(sum(quant.converged)), int((quant.nlayer + 1) - sum(quant.conv_layer))))

    if sum(quant.converged) == (quant.nlayer + 1) - sum(quant.conv_layer):
        criterion = 1

    return criterion


def give_feedback_on_convergence(quant):

    start_layers = []
    end_layers = []
    radiative_layers = []
    for i in range(quant.nlayer + 1):  # including surface/BOA "ghost layer"

        if quant.conv_layer[i] == 0:
            radiative_layers.append(i)

    if quant.nlayer in radiative_layers:
        radiative_layers = np.insert(radiative_layers[:-1], 0, -1)

    for i in range(len(radiative_layers)):
        if radiative_layers[i] - 1 not in radiative_layers:
            start_layers.append(radiative_layers[i])
        if radiative_layers[i] + 1 not in radiative_layers:
            end_layers.append(radiative_layers[i])

    for n in range(len(start_layers)):

        if n < len(start_layers) - 1:

            interface_to_be_tested = int((start_layers[n] + end_layers[n] + 1) / 2)

            local_F_net_diff = abs(quant.F_intern + quant.F_add_heat_sum[interface_to_be_tested - 1] - quant.F_net[interface_to_be_tested]) / (quant.F_down_tot[quant.nlayer] + quant.F_intern)
            print("Radiative energy imbalance in intermediate rad. layers is {:.3e} and should be less than {:.1e}".format(local_F_net_diff, quant.rad_convergence_limit))
        else:
            local_F_net_diff = abs(quant.F_intern + quant.F_add_heat_sum[quant.nlayer - 1] - quant.F_net[quant.nlayer]) / (quant.F_down_tot[quant.nlayer] + quant.F_intern)
            print("Global energy imbalance is {:.3e} and should be less than {:.1e}".format(local_F_net_diff, quant.rad_convergence_limit))


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

        if quant.p_lay[i] <= 1e1:  # ignore top atmosphere, since artificial/numerical temperature peaks might occur there
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

    for i in range(quant.nlayer + 1):  # including surface/BOA "ghost layer"
        if quant.conv_unstable[i] == 1 or quant.conv_layer[i] == 1:
            to_be_corrected_list.append(i)

    # for i in range(quant.nlayer+1):  # including surface/BOA "ghost layer"
    #     if quant.conv_unstable[i] == 1:
    #         to_be_corrected_list.append(i)
    #     elif i < quant.nlayer - 1:
    #         if quant.conv_layer[i] == 1 and quant.conv_layer[i+1] == 1:
    #             to_be_corrected_list.append(i)
    #     elif quant.conv_layer[i] == 1:
    #         to_be_corrected_list.append(i)

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

                else:  # toplayers

                    interface_to_be_tested = int(0.8 * end_layers[m] + 0.2 * (quant.ninterface - 1))

            if quant.input_dampara == 'automatic':

                if quant.T_star > 10:  # values for cases with a stellar irradiation

                    if n < len(start_layers) - 1:  # in the case of intermediate radiative zone, fudging needs to be stronger

                        quant.dampara = 0.5  # dampara = damping parameter (because dampens the conv. zone fudging )

                    else:  # for topmost conv. zone, dampara = 4 appears to be the most stable

                        quant.dampara = 4.0

                else:  # for self-luminous planets, dampara = 8 appears to work consistently

                    quant.dampara = 8.0

            else:
                quant.dampara = float(quant.input_dampara)

            fudge_factor[n] = ((quant.F_intern + quant.F_add_heat_sum[interface_to_be_tested-1] + quant.F_smooth_sum[interface_to_be_tested-1] + quant.F_down_tot[interface_to_be_tested]) / quant.F_up_tot[interface_to_be_tested]) ** (1.0 / quant.dampara)

            fudge_factor[n] = min(1.01, max(0.99, fudge_factor[n]))  # to prevent instabilities

        ## uncomment next few lines for debugging
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

            num += quant.c_p_lay[i] / quant.meanmolmass_lay[i] * quant.T_lay[i] * (quant.p_int[i] - quant.p_int[i+1])

            denom_element = 1

            if i != start_index:

                for j in range(start_index, i):

                    denom_element *= (quant.p_lay[j]/quant.p_int[j])**quant.kappa_int[j] * (quant.p_int[j+1]/quant.p_lay[j])**quant.kappa_lay[j]

            denom_element *= (quant.p_lay[i]/quant.p_int[i])**quant.kappa_int[i] * quant.c_p_lay[i] / quant.meanmolmass_lay[i] * (quant.p_int[i] - quant.p_int[i+1])

            denom += denom_element

        mean_pot_temp = num / denom
        # Note that strictly speaking, the mean molecular mass has to be divided by AMU in the above expressions to obtain the correct g/mol units.
        # However, since the same factor is both in the numerator and denominator it cancels out and that's why it is not present in above expressions.

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

        if quant.p_lay[i] <= 1e1:  # ignore top atmosphere, since artificial/numerical temperature peaks might occur there
            break

        T_in_between_lim = quant.T_lay[i] * (quant.p_int[i + 1] / quant.p_lay[i]) ** (quant.kappa_lay[i] * (1 - 1e-6))

        T_ad_lim = T_in_between_lim * (quant.p_lay[i + 1] / quant.p_int[i + 1]) ** (quant.kappa_int[i + 1] * (1 - 1e-6))

        if quant.T_lay[i+1] < T_ad_lim:
            quant.conv_layer[i] = 1
            quant.conv_layer[i+1] = 1
        else:
            quant.conv_layer[i + 1] = 0

    # # to avoid kinks at top edge of conv. zone
    for i in range(quant.nlayer - 1):
        if quant.T_lay[i+1] > quant.T_lay[i]:
            quant.conv_layer[i] = 0

    # do the surface/BOA condition
    T_ad_lim = quant.T_lay[quant.nlayer] * (quant.p_lay[0] / quant.p_int[0]) ** (quant.kappa_int[0] * (1 - 1e-6))

    if quant.T_lay[0] < T_ad_lim:
        quant.conv_layer[quant.nlayer] = 1
        quant.conv_layer[0] = 1

    # stitch holes if taking too long to converge
    if stitching == 1:
        if quant.iter_value > 5000:  # warning: hardcoded number
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

    for i in range(1, quant.ninterface):

        if quant.conv_layer[i-1] == 1:
            quant.F_net_conv[i] = quant.F_intern + quant.F_add_heat_sum[i-1] + quant.F_smooth_sum[i-1] - quant.F_net[i]
            # note: F_add_heat_sum and F_smooth_sum are interface quantities, but they start at interface=1 and have length nlayer

    # BOA / surface layer
    if quant.conv_layer[quant.nlayer] == 1:
        quant.F_net_conv[0] = quant.F_intern - quant.F_net[0]


def calc_F_ratio(quant):
    """ calculates the planet to star flux ratio for sec. eclipse data interpretation """

    if quant.T_star > 10:
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

    elif quant.planet_type in ['rocky', 'no_atmosphere']:

        quant.z_lay[0] = 0.5 * quant.delta_z_lay[0]

        for i in range(1, quant.nlayer):

            quant.z_lay[i] = quant.z_lay[i-1] + 0.5 * quant.delta_z_lay[i-1] + 0.5 * quant.delta_z_lay[i]


def calc_add_heating_flux(quant):
    """ calculates the UV heating flux -- individual layers and added up to get the total additional atmospheric heating """

    quant.F_add_heat_lay = quant.add_heat_dens * quant.delta_z_lay

    for i in range(quant.nlayer):

        if i == 0:
            quant.F_add_heat_sum[i] = quant.F_add_heat_lay[i]
        else:
            quant.F_add_heat_sum[i] = quant.F_add_heat_sum[i-1] + quant.F_add_heat_lay[i]


def calculate_pressure_levels(quant):

    press_levels = [quant.p_boa * (quant.p_toa/quant.p_boa)**(i/(2 * quant.nlayer - 1)) for i in range(2 * quant.nlayer)]

    p_layer = [press_levels[i] for i in range(1, 2 * quant.nlayer, 2)]

    p_interface = [press_levels[i] for i in range(0, 2 * quant.nlayer, 2)]

    p_interface.append(quant.p_toa * (quant.p_toa/quant.p_boa)**(1/(2 * quant.nlayer - 1)))

    return p_layer, p_interface


def construct_grid(quant):

    quant.p_lay, quant.p_int = calculate_pressure_levels(quant)

    for i in range(quant.nlayer):

        quant.delta_colmass.append((quant.p_int[i] - quant.p_int[i + 1]) / quant.g)
        quant.delta_col_upper.append((quant.p_lay[i] - quant.p_int[i + 1]) / quant.g)
        quant.delta_col_lower.append((quant.p_int[i] - quant.p_lay[i]) / quant.g)

# keep this for a later implementation -- later as in some time in the future... in the far future
# def calc_sensible_heat_flux(quant):
#     """ calculates the sensible heat flux for the surface layers """
#
#     # get arrays from GPU
#     quant.meanmolmass_lay = quant.dev_meanmolmass_lay.get()
#     quant.c_p_lay = quant.dev_c_p_lay.get()
#
#     # horizontal wind speed in cgs
#     U = 1e4
#
#     # simplified drag coefficient for simple tests
#     C_D = 0.1
#
#     # drag coefficient after similarity theory -- commented out for the moment
#     # K_vk = 0.4 # von Karman constant
#     # C_D = (K_vk / np.log(z_surf_lay_top/z_surf))**2
#
#     # temperature at the bottom of the surface layer is the surface/BOA temperature
#     T_surf_lay_bot = quant.T_lay[quant.nlayer]
#
#     rho_surface = quant.p_int[0] * quant.meanmolmass_lay[0] / (pc.K_B * T_surf_lay_bot)
#
#     # calculating the top temperature of the surface layer
#     # top altitude in cgs
#     z_surf_lay_top = 1e5  # 1 km
#     z_surf_lay_bot = 1  # 1 cm
#
#     delta_z_surf_lay = z_surf_lay_top - z_surf_lay_bot
#     p_surf_lay_top = quant.p_int[0] - rho_surface * quant.g * delta_z_surf_lay
#
#     T_surf_lay_top = (np.log10(quant.p_int[0]/p_surf_lay_top) * quant.T_lay[0] + np.log10(p_surf_lay_top/quant.p_lay[0]) * T_surf_lay_bot) \
#                      / np.log10(quant.p_int[0]/quant.p_lay[0])
#
#     # c_p needs to be converted from per mole to per mass first
#     quant.F_sens = quant.c_p_lay[0] / quant.meanmolmass_lay[0] * rho_surface * C_D * U * (T_surf_lay_bot - T_surf_lay_top)
#
#     # screen feedback for debugging purposes
#     # if quant.iter_value % 10 == 0:
#     #     print("P_surf: {:.2e}, P_surf_layer_top: {:.2e}, P_lay[0]: {:.2e}".format(quant.p_int[0], p_surf_lay_top, quant.p_lay[0]))
#     #     print("T_surf - T_surf_layer_top: {:.2e}".format(quant.T_lay[quant.nlayer] - T_surf_lay_top))
#     #     print("F_up_rad: {:.2e}, F_sens: {:.2e}".format(quant.F_up_tot[0], quant.F_sens))
#
#     # side note: c_p_lay because c_p_int is not calculated and the difference will be negligible


def interpolate_vmr_to_opacity_grid(read, quant, vmr):

    temp_old = read.fastchem_temp
    press_old = read.fastchem_press
    temp_new = quant.ktemp
    press_new = quant.kpress

    vmr_old = vmr

    vmr_new = np.zeros(quant.npress * quant.ntemp)

    old_nt = len(temp_old)
    old_np = len(press_old)

    ### some on-screen feedback -- commented out to reduce on-screen spam because info not really necessary ###
    # print("\nInterpolating FastChem VMR...")
    # print("FastChem temperature grid:\n", temp_old[:3], "...", temp_old[-3:])
    # print("opacity table temperature grid:\n", temp_new[:3], "...", temp_new[-3:])
    # print("FastChem pressure grid:\n", press_old[:3], "...", press_old[-3:])
    # print("opacity table pressure grid:\n", press_new[:3], "...", press_new[-3:])
    #
    # print("opacity table N_P:", quant.npress, "opacity table N_T:", quant.ntemp)
    # print("FastChem N_P:", old_np, "FastChem N_T:", old_nt)

    for i in range(quant.ntemp):

        for j in range(quant.npress):

            reduced_t = 0
            reduced_p = 0

            try:
                t_left = max([t for t in range(len(temp_old)) if temp_old[t] <= temp_new[i]])
            except ValueError:
                t_left = 0
                reduced_t = 1

            try:
                p_left = max([p for p in range(len(press_old)) if press_old[p] <= press_new[j]])
            except ValueError:
                p_left = 0
                reduced_p = 1

            if t_left == len(temp_old) - 1:
                reduced_t = 1

            if p_left == len(press_old) - 1:
                reduced_p = 1

            if reduced_p == 1 and reduced_t == 1:

                vmr_new[j + quant.npress * i] = vmr_old[p_left + old_np * t_left]

            elif reduced_p != 1 and reduced_t == 1:

                p_right = p_left + 1

                vmr_new[j + quant.npress * i] = \
                        (vmr_old[p_right + old_np * t_left] * (np.log10(press_new[j]) - np.log10(press_old[p_left])) \
                         + vmr_old[p_left + old_np * t_left] * (np.log10(press_old[p_right]) - np.log10(press_new[j])) \
                         ) / (np.log10(press_old[p_right]) - np.log10(press_old[p_left]))

            elif reduced_p == 1 and reduced_t != 1:

                t_right = t_left + 1

                vmr_new[j + quant.npress * i] = \
                    (vmr_old[p_left + old_np * t_right] * (temp_new[i] - temp_old[t_left]) \
                     + vmr_old[p_left + old_np * t_left] * (temp_old[t_right] - temp_new[i]) \
                     ) / (temp_old[t_right] - temp_old[t_left])

            elif reduced_p != 1 and reduced_t != 1:

                p_right = p_left + 1
                t_right = t_left + 1

                vmr_new[j + quant.npress * i] = \
                    (
                        vmr_old[p_right + old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (np.log10(press_new[j]) - np.log10(press_old[p_left])) \
                        + vmr_old[p_left + old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (np.log10(press_old[p_right]) - np.log10(press_new[j])) \
                        + vmr_old[p_right + old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (np.log10(press_new[j]) - np.log10(press_old[p_left])) \
                        + vmr_old[p_left + old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (np.log10(press_old[p_right]) - np.log10(press_new[j])) \
                        ) / ((temp_old[t_right] - temp_old[t_left]) * (np.log10(press_old[p_right]) - np.log10(press_old[p_left])))

            if np.isnan(vmr_new[j + quant.npress * i]):
                print("NaN-Error at entry with indices:", "pressure:", j, "temperature:", i)
                raise SystemExit()

    return vmr_new


def calculate_vmr_for_all_species(quant):
    """ calculates VMR for all species in on-the-fly opacity mixing mode """

    quant.T_lay = quant.dev_T_lay.get()
    quant.T_int = quant.dev_T_int.get()
    quant.p_lay = quant.dev_p_lay.get()
    quant.p_int = quant.dev_p_int.get()

    log_p_lay = np.log10(quant.p_lay)
    log_p_int = np.log10(quant.p_int)

    log_kpress = np.log10(quant.kpress)

    # one loop to get vmrs
    for s in range(len(quant.species_list)):

        # interpolate pretabulated VMR for FastChem grid to vertical VMR profiles
        if quant.species_list[s].source_for_vmr == "FastChem":

            vmr_2D = quant.species_list[s].vmr_pretab.reshape((quant.ntemp, quant.npress))

            quant.species_list[s].vmr_layer = interpolate_grid_to_lay_or_int(log_kpress, quant.ktemp, vmr_2D, log_p_lay, quant.T_lay)
            if quant.iso == 0:
                quant.species_list[s].vmr_interface = interpolate_grid_to_lay_or_int(log_kpress, quant.ktemp, vmr_2D, log_p_int, quant.T_int)

            # convert to numpy arrays in order to have the correct format for copying to GPU
            quant.species_list[s].vmr_layer = np.array(quant.species_list[s].vmr_layer, quant.fl_prec)
            quant.species_list[s].vmr_interface = np.array(quant.species_list[s].vmr_interface, quant.fl_prec)


def interpolate_grid_to_lay_or_int(log_press, temp, vmr_2D, log_press_profile, temp_profile):
    """ interpolates quantities on 2D to atmospheric T-P profile """

    func = interpolate.RectBivariateSpline(temp, log_press, vmr_2D, kx=1, ky=1)
    vmr_lay_or_int = [func(temp_profile[p], log_press_profile[p])[0][0] for p in range(len(log_press_profile))]

    return vmr_lay_or_int


def calculate_meanmolecularmass(quant):
    """ calculates mean molecular mass in on-the-fly opacity mixing mode """

    # calculating mean molecular mass for each layer
    quant.meanmolmass_lay = calc_meanmolmass(quant, type='layer')

    quant.dev_meanmolmass_lay = gpuarray.to_gpu(quant.meanmolmass_lay)

    if quant.iso == 0:
        quant.meanmolmass_int = calc_meanmolmass(quant, type='interface')

        quant.dev_meanmolmass_int = gpuarray.to_gpu(quant.meanmolmass_int)


def calc_meanmolmass(quant, type='layer'):
    """ calculates mean molecular mass based on VMR of all species """

    if type == "layer":
        nlayer_or_ninterface = quant.nlayer

    elif type == "interface":
        nlayer_or_ninterface = quant.ninterface

    meanmolmass_lay_or_int = np.zeros(nlayer_or_ninterface)

    for i in range(nlayer_or_ninterface):

        vmr_lay_or_int_total = 0

        for s in range(len(quant.species_list)):

            if ("CIA" not in quant.species_list[s].name) and (quant.species_list[s].name != "H-_ff") and (quant.species_list[s].name != "He-"):

                if type == "layer":
                    vmr_lay_or_int = quant.species_list[s].vmr_layer
                elif type == "interface":
                    vmr_lay_or_int = quant.species_list[s].vmr_interface

                meanmolmass_lay_or_int[i] += vmr_lay_or_int[i] * quant.species_list[s].weight

                vmr_lay_or_int_total += vmr_lay_or_int[i]

        meanmolmass_lay_or_int[i] /= vmr_lay_or_int_total  # normalizing with respect to total VMR of all species

    meanmolmass_lay_or_int = np.array(meanmolmass_lay_or_int * pc.AMU, quant.fl_prec)  # converting weight (or molar mass) to molecular mass

    return meanmolmass_lay_or_int


def calculate_coupling_convergence(quant, read):
    """ test whether TP profile converged and ends coupled iteration """

    coupl_convergence = 0

    if quant.coupling_iter_nr > 0 and quant.singlewalk == 0:
        # read in temperatures from the last two iterations
        previous_temp = []
        current_temp = []

        if quant.coupling_full_output == 1:

            base_name = None

            # get the previous directory name
            for n in range(len(quant.name) - 1, 0, -1):
                if quant.name[n] == "_":
                    base_name = quant.name[:n + 1]
                    break

            previous_name = base_name + str(quant.coupling_iter_nr-1)

            file_path_previous = read.output_path + previous_name + "/" + previous_name + "_tp_coupling_" + str(quant.coupling_iter_nr-1) + ".dat"

        else:

            file_path_previous = read.output_path + quant.name + "/" + quant.name + "_tp_coupling_" + str(quant.coupling_iter_nr - 1) + ".dat"

        with open(file_path_previous, "r") as previous_file:
            next(previous_file)
            for line in previous_file:
                column = line.split()
                if len(column) > 1:
                    previous_temp.append(quant.fl_prec(column[1]))

        with open(read.output_path + quant.name + "/" + quant.name + "_tp_coupling_" + str(quant.coupling_iter_nr) + ".dat", "r") as current_file:
            next(current_file)
            for line in current_file:
                column = line.split()
                if len(column) > 1:
                    current_temp.append(quant.fl_prec(column[1]))

        converged_list = []

        for t in range(len(current_temp)):

            if abs(previous_temp[t] - current_temp[t]) / current_temp[t] < quant.coupl_convergence_limit:

                converged_list.append(1)

        if len(converged_list) == len(current_temp):

            coupl_convergence = 1

        # write out result
        with open(read.output_path + quant.name + "/" + quant.name + "_coupling_convergence.dat", "w") as file:
            file.writelines(str(coupl_convergence))


def success_message(quant):
    """ prints the message that you have been desperately waiting for """

    T_eff_global, T_eff_dayside, T_eff_model, T_star_brightness, T_planet_brightness = temp_calcs(quant)

    print("\nDone! Everything appears to have worked fine :-)\n")

    run_type = "an iterative" if quant.singlewalk == 0 else "a post-processing"

    print("This has been " + run_type + " run with name " + quant.name + ".\n")

    # displays following message for usual run until equilibrium
    if quant.physical_tstep == 0:
        print("\nFinal Check for numerical energy balance:")
        print("  --> Theoretical effective temperature of planet: \n\tglobal (f=0.25): {:g} K,".format(T_eff_global),
              "\n\tday-side (f=2/3): {:g} K,".format(T_eff_dayside), "\n\tused in model (f={:.3f}): {:g} K.".format(quant.f_factor, T_eff_model))
        print("  --> Incident TOA brightness temperature: {:g} K \n      Interior temperature: {:g} K".format(T_star_brightness, quant.T_intern),
              "\n      Outgoing (planetary) brightness temperature: {:g} K".format(T_planet_brightness))

        relative_energy_imbalance = (quant.F_intern + quant.F_add_heat_sum[quant.ninterface - 2] + quant.F_smooth_sum[quant.ninterface - 2] - quant.F_net[quant.ninterface - 1]) / (quant.F_down_tot[quant.ninterface - 1] + quant.F_intern)

        print("  --> Global energy imbalance: {:.3f}ppm (positive: too much uptake, negative: too much loss).".format(relative_energy_imbalance*1e6), "\n")
    # otherwise, just display physical timestep info
    else:
        print("Model has run in physical timestep mode with a step of {:g} s, and stopped after reaching the runtime limit of {:g} s.".format(quant.physical_tstep, quant.runtime_limit))
        if quant.convection == 1:
            print("  --> If convectively unstable layers have been found, convective adjustment has been performed once(!) at the very end.")


def nullify_opac_scat_arrays(quant):

    quant.dev_opac_wg_lay = gpuarray.to_gpu(quant.opac_wg_lay)
    quant.dev_opac_wg_int = gpuarray.to_gpu(quant.opac_wg_int)

    quant.dev_scat_cross_lay = gpuarray.to_gpu(quant.scat_cross_lay)
    quant.dev_scat_cross_int = gpuarray.to_gpu(quant.scat_cross_int)


if __name__ == "__main__":
    print("This module stores the definitions for the functions living on the host. "
          "It is more spacious on the host than on the device, but also warmer.")