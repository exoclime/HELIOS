# ==============================================================================
# Module for writing the output quantities of HELIOS to files.
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

import os
import shutil
from source import phys_const as pc
from source import host_functions as hsfunc


class Write(object):
    """ class that possesses all the methods to write out data """

    def __init__(self):
        pass

    @staticmethod
    def convert_1_0_to_yes_no(variable):
        """ converts a variable with value 1 or 0 to a "yes" or "no" message """
        if variable == 1:
            return "yes"
        else:
            return "no"

    @staticmethod
    def write_physical_timestep(variable):
        """ returns physical timestep variable in the correct format """

        if variable == 0:

            return "no"

        else:
            return "{:g}".format(variable)

    @staticmethod
    def write_mean_werror(quantity):
        """ writes an error message for quantities that could not be calculated """

        if quantity == -3:
            return "{:<20}".format("temp_too_low")
        else:
            return "{:<20g}".format(quantity)

    @staticmethod
    def write_abort_file(quant, read):
        """ writes a file that tells you that the run has been aborted due to exceeding iteration steps """

        # create directory if necessary
        try:
            os.makedirs(read.output_path + quant.name)
        except OSError:
            if not os.path.isdir(read.output_path + quant.name):
                raise

        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_ABORT.dat", "w") as file:
                file.writelines("The run exceeded the maximum number of iteration steps and was aborted. Sorry.")
        except TypeError:
            print("ABORT file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_criterion_warning_file(quant, read):
        """ writes a file that tells you that the convergence criterion has been made more loose """

        if quant.relaxed_criterion_trigger == 1:

            # create directory if necessary
            try:
                os.makedirs(read.output_path + quant.name)
            except OSError:
                if not os.path.isdir(read.output_path + quant.name):
                    raise

            with open(read.output_path + quant.name + "/" + quant.name + "_convergence_warning.dat", "w") as file:
                file.writelines("WARNING: Due to exceeding runtime the convergence criterion has been made more loose over time.\n")
                file.writelines("The final relative criterion used is: {:.1e} \n".format(quant.rad_convergence_limit))
                file.writelines("Even with a looser (not loser) criterion, the model results may still be accurate enough. Use at your own discretion!")

    @staticmethod
    def create_output_dir_and_copy_param_file(read, quant):

        try:
            os.makedirs(read.output_path + quant.name)
        except OSError:
            if not os.path.isdir(read.output_path + quant.name):
                raise

        cwd = os.getcwd()

        source = cwd + "/" + read.param_file
        destination = read.output_path +quant.name+"/" + quant.name + "_" + read.param_file

        shutil.copyfile(source, destination)

    @staticmethod
    def write_tp(quant, read):
        """ writes the TP-profile and the effective atmospheric temperature to a file """

        _, _, _, _, T_bright = hsfunc.temp_calcs(quant)

        try:
            with open(read.output_path +quant.name+"/" + quant.name + "_tp.dat", "w") as file:
                file.writelines("This file contains the corresponding layer temperatures and pressures, and the altitude and the height of each layer.")

                file.writelines(
                    "\n{:<8}{:<18}{:<24}{:<21}{:<23}{:<30}{:<32}{:<18}".format(
                        "layer", "temp.[K]", "press.[10^-6bar]", "altitude[cm]", "height.of.layer[cm]",
                        "conv.unstable?[1:yes,0:no]", "conv.lapse-rate?[1:yes,0:no]", "pl.eff.temp.[K]")
                )
                file.writelines("\n{:<8}{:<18g}{:<24g}{:<21g}{:<23}".format(
                    "BOA", quant.T_lay[quant.nlayer], quant.p_int[0], quant.z_lay[0] - 0.5 * quant.delta_z_lay[0], "not_avail."))

                if quant.iso == 0 and quant.convection == 1:
                    file.writelines("{:<30g}{:<32g}".format(quant.conv_unstable[quant.nlayer], quant.conv_layer[quant.nlayer]))

                if quant.iso == 1 or quant.convection == 0:
                    file.writelines("{:<30}{:<32}".format("not_calculated", "not_calculated"))

                file.writelines("{:<18g}".format(T_bright))

                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<8g}".format(i)
                        + "{:<18g}".format(quant.T_lay[i])
                        + "{:<24g}".format(quant.p_lay[i])
                        + "{:<21g}".format(quant.z_lay[i])
                        + "{:<23g}".format(quant.delta_z_lay[i]))
                    if quant.iso == 0 and quant.convection == 1:
                        file.writelines("{:<30g}{:<32g}".format(quant.conv_unstable[i], quant.conv_layer[i]))
                    if quant.iso == 1 or quant.convection == 0:
                        file.writelines("{:<30}{:<32}".format("not_calculated", "not_calculated"))
        except TypeError:
            print("TP-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_tp_cut(quant, read):
        """ writes the TP-profile up to a height of P = 1e-6 bar to a file """

        try:
            with open(read.output_path + quant.name+"/" + quant.name + "_tp_cut.dat", "w") as file:
                file.writelines(
                    "This file contains the corresponding layer temperatures and pressures.")

                file.writelines(
                    "\n{:<8}{:<18}{:<24}".format("layer", "temp.[K]", "press.[10^-6bar]")
                )

                file.writelines("\n{:<8}{:<18g}{:<24g}".format("BOA", quant.T_lay[quant.nlayer], quant.p_int[0]))

                for i in range(quant.nlayer):
                    if quant.p_lay[i] > 0.099:
                        file.writelines(
                            "\n{:<8g}".format(i)
                            + "{:<18g}".format(quant.T_lay[i])
                            + "{:<24g}".format(quant.p_lay[i]))
        except TypeError:
            print("File '*_tp_cut.dat generation' corrupted. You might want to look into it!")

    @staticmethod
    def write_colmass_mu_cp_entropy(quant, read):
        """ writes the layer column mass, mean molecular weight and specific heat capacity to a file """

        with open(read.output_path + quant.name + "/" + quant.name + "_colmass_mu_cp_kappa_entropy.dat", "w") as file:
            file.writelines(
                "This file contains the total pressure and the column mass difference, mean molecular weight and specific heat capacity of each layer.")
            file.writelines(
                "\n{:<8}{:<24}{:<26}{:<21}{:<32}{:<23}{:<30}".format(
                    "layer", "cent.press.[10^-6bar]", "delta_col.mass[g cm^-2]",
                    "mean mol. weight", "spec.heat cap.[erg mol^-1 K^-1]", "adiabatic coefficient", "entropy [erg g^-1 K^-1]")
            )
            for i in range(quant.nlayer):
                file.writelines(
                    "\n{:<8g}".format(i)
                    + "{:<24g}".format(quant.p_lay[i])
                    + "{:<26g}".format(quant.delta_colmass[i])
                    + "{:<21g}".format(quant.meanmolmass_lay[i] / pc.AMU)
                )
                if quant.c_p_lay[i] == 0:
                    file.writelines("{:<32s}".format("not_calculated"))
                else:
                    file.writelines("{:<32g}".format(quant.c_p_lay[i]))
                if quant.kappa_lay[i] == 0:
                    file.writelines("{:<23s}".format("not_calculated"))
                else:
                    file.writelines("{:<23g}".format(quant.kappa_lay[i]))
                if quant.entropy_lay[i] == 0:
                    file.writelines("{:<30s}".format("not_calculated"))
                else:
                    file.writelines("{:<30g}".format(quant.entropy_lay[i]))

    @staticmethod
    def write_phase_state(quant, read):

        # only executed for "water_atmo" file format
        if quant.input_kappa_value == "water_atmo":

            with open(read.output_path + quant.name + "/" + quant.name + "_state.dat", "w") as file:

                file.writelines(
                    "Checks the phase state of the water atmosphere. If '1' the water in the atmosphere is vaporous or supercritical. "
                    "If '<1' atmosphere might be unstable, i.e., water in liquid or solid form.")

                file.writelines(
                    "\n{:<8}{:<18}{:<24}{:<24}".format("layer", "temp.[K]", "press.[10^-6bar]",
                                                       "state_of_water (0: liquid or solid, 1: vapor or supercritical)")
                )

                for i in range(quant.nlayer):
                    if quant.p_lay[i] > 0.99:
                        file.writelines(
                            "\n{:<8g}".format(i)
                            + "{:<18g}".format(quant.T_lay[i])
                            + "{:<24g}".format(quant.p_lay[i]))
                        file.writelines("{:<24g}".format(quant.phase_number_lay[i]))

    @staticmethod
    def write_integrated_flux(quant, read):
        """ writes the integrated total and net fluxes to a file """
        try:
            with open(read.output_path + quant.name+"/" + quant.name + "_integrated_flux.dat", "w") as file:
                file.writelines("This file contains the integrated total and net fluxes at each interface resp. "
                                "layer. \nFluxes given in [erg s^-1 cm^-2].")
                file.writelines(
                    "\n{:<20}{:<24}{:<25}{:<25}{:<23}{:<25}{:<34}{:<24}{:<24}{:<12}".format(
                        "interface", "press.[10^-6bar]", "F_down", "F_up", "F_net", "F_dir", "delta_F_net (layer quantity)", "F_net_conv", "F_add_heat", "F_intern")
                )
                for i in range(quant.ninterface):
                    file.writelines(
                        "\n{:<20g}".format(i)
                        + "{:<24g}".format(quant.p_int[i])
                        + "{:<25g}".format(quant.F_down_tot[i])
                        + "{:<25g}".format(quant.F_up_tot[i])
                        + "{:<23g}".format(quant.F_net[i])
                        + "{:<25g}".format(quant.F_dir_tot[i]))
                    if quant.singlewalk == 0 and i < quant.nlayer:
                        file.writelines("{:<34g}".format(quant.F_net_diff[i]))
                    else:
                        file.writelines("{:<34}".format("not_avail."))
                    file.writelines("{:<24g}".format(quant.F_net_conv[i]))
                    if i < quant.nlayer:
                        file.writelines("{:<24g}".format(quant.F_add_heat_lay[i]))
                    else:
                        file.writelines("{:<24}".format("not_avail."))
                    if i == 0:
                        file.writelines("{:<12g}".format(quant.F_intern))

        except TypeError:
            print("Integrated flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_upward_spectral_flux(quant, read):
        """ writes the upward spectral flux to a file """
        try:
            with open(read.output_path + quant.name+"/" + quant.name + "_spec_upflux.dat", "w") as file:
                file.writelines("This file contains the upward spectral flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.ninterface):
                    file.writelines("{:<5}{:g}{:<4}".format("F_up[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<16.8e}".format(quant.F_up_band[x + i * quant.nbin]))
        except TypeError:
            print("Upward spectral flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_downward_spectral_flux(quant, read):
        """ writes the downward spectral flux to a file """
        try:
            with open(read.output_path + quant.name+"/" + quant.name + "_spec_downflux.dat", "w", encoding='utf-8') as file:
                file.writelines("This file contains the downward spectral flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.ninterface):
                    file.writelines("{:<7}{:g}{:<4}".format("F_down[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<16.8e}".format(quant.F_down_band[x + i * quant.nbin]))
        except TypeError:
            print("Downward spectral flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_TOA_flux_eclipse_depth(quant, read):
        """ writes the TOA fluxes to a file """
        try:
            with open(read.output_path + quant.name+"/" + quant.name + "_TOA_flux_eclipse.dat", "w") as file:
                file.writelines("This file contains the downward and upward spectral flux (per wavelength) at TOA "
                                "and the secondary eclipse depth (= planet to star flux ratio)."
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:<21}{:<19}{:<16}{:<16}{:<24}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]", "F_down_at_TOA", "F_up_at_TOA", "planet/star flux ratio")
                )
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    file.writelines("{:<16g}".format(quant.F_down_band[x + quant.nlayer * quant.nbin])
                                    + "{:<16g}".format(quant.F_up_band[x + quant.nlayer * quant.nbin])
                                    )
                    if quant.T_star > 10:
                        file.writelines("{:<24g}".format(quant.F_ratio[x]))
                    else:
                        file.writelines("{:<24}".format("not_avail."))
        except TypeError:
            print("TOA flux file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_flux_ratio_only(quant, read):
        """ writes only the planetary and stellar flux ratio to a file, e.g., to be readable by Pandexo """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_flux_ratio.dat", "w") as file:
                for x in range(quant.nbin):
                    file.writelines("{:<18.9g}".format(quant.opac_wave[x] * 1e4))
                    if quant.T_star > 10:
                        file.writelines("{:<12g}\n".format(quant.F_ratio[x]))
                    else:
                        file.writelines("{:<12}\n".format("not_avail."))
        except TypeError:
            print("Flux ratio file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_surface_albedo(quant, read):
        """ writes the surface albedo per wavelength """

        with open(read.output_path + quant.name + "/" + quant.name + "_surf_albedo.dat", "w") as file:
            file.writelines("This file contains the surface albedo per wavelength.")
            if read.input_surf_albedo == "file":
                file.writelines("\nThe surface material used is: " + read.albedo_file_surface_name)
            else:
                file.writelines("\nA value was chosen manually, hence all the values below are constant.")
            file.writelines(
                "\n{:<8}{:<18}{:<21}{:<19}{:<16}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]", "surface_albedo")
            )
            for x in range(quant.nbin):
                file.writelines("\n{:<8g}".format(x)
                                + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                + "{:<16g}".format(quant.surf_albedo[x]))

    @staticmethod
    def write_direct_spectral_beam_flux(quant, read):
        """ writes the direct irradiation beam flux to a file """
        try:
            with open(read.output_path + quant.name+"/" + quant.name + "_direct_beamflux.dat", "w") as file:
                file.writelines("This file contains the direct irradiation flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:18}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.ninterface):
                    file.writelines("{:<6}{:g}{:<4}".format("F_dir[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<16.8e}".format(quant.F_dir_band[x + i * quant.nbin]))
        except TypeError:
            print("Direct irradiation flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_planck_interface(quant, read):
        """ writes the Planck function at interfaces to a file """
        if quant.iso == 0:
            try:
                with open(read.output_path + quant.name+"/" + quant.name + "_planck_int.dat", "w") as file:
                    file.writelines("This file contains the Planck (blackbody) function at each interface. "
                                    "\nPlanck function given in [erg s^-1 cm^-3 sr^-1].")
                    file.writelines(
                        "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<6}{:g}{:<4}".format("B_int[", i, "]"))
                    for x in range(quant.nbin):
                        file.writelines("\n{:<8g}".format(x)
                                        + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                        + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                        + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                        )
                        for i in range(quant.ninterface):
                            file.writelines("{:<16g}".format(quant.planckband_int[i + x * quant.ninterface]))
            except TypeError:
                print("Planck-file (interfaces) generation corrupted. You might want to look into it!")

    @staticmethod
    def write_planck_center(quant, read):
        """ writes the Planck function at layer centers (+ stellar temp. and internal temp.) to a file. """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_planck_cent.dat", "w") as file:
                file.writelines("This file contains the Planck (blackbody) function at each layer center and "
                                "from the stellar (2nd last column) and internal (last column) temperatures. "
                                "\nPlanck function given in [erg s^-1 cm^-3 sr^-1].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<6}{:g}{:<4}".format("B_lay[", i, "]"))
                file.writelines("{:<16}{:<16}".format("Planck_T_star", "Planck_T_intern")
                                )
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer+2):
                        file.writelines("{:<16g}".format(quant.planckband_lay[i + x * (quant.nlayer+2)]))
        except TypeError:
            print("Planck-file (layer centers) generation corrupted. You might want to look into it!")

    @staticmethod
    def write_opacities(quant, read):
        """ writes the bin integrated opacities to a file. """

        with open(read.output_path + quant.name + "/" + quant.name + "_opacities.dat", "w") as file:
            file.writelines("This file contains the bin integrated opacities at each layer center "
                            "\nOpacity given in [cm^2 g^-1].")
            file.writelines(
                "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
            )
            for i in range(quant.nlayer):
                file.writelines("{:<9}{:g}{:<4}".format("opac_lay[", i, "]"))
            for x in range(quant.nbin):
                file.writelines("\n{:<8g}".format(x)
                                + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                )
                for i in range(quant.nlayer):
                    file.writelines("{:<15g}".format(quant.opac_band_lay[x + quant.nbin * i]))

    @staticmethod
    def write_cloud_mixing_ratio(quant, read):
        """ writes the vertical cloud mixing ratio to a file """

        with open(read.output_path + quant.name+"/" + quant.name + "_cloud_mixing_ratio.dat", "w") as file:
            file.writelines(
                "This file contains the cloud volume mixing ratio (= n_cloud/n_gas) at each vertical layer.")

            file.writelines(
                "\n{:<8}{:<24}{:<18}".format("layer", "press.[10^-6bar]", "cloud_vmr")
            )

            for i in range(quant.nlayer):
                file.writelines(
                    "\n{:<8g}".format(i)
                    + "{:<24g}".format(quant.p_lay[i])
                    + "{:<18g}".format(quant.f_all_clouds_lay[i]))

    @staticmethod
    def write_cloud_opacities(quant, read):
        """ writes the cloud opacity per bin to a file"""

        with open(read.output_path + quant.name + "/" + quant.name + "_cloud_opacities.dat", "w") as file:
            file.writelines("This file contains the cloud opacities at each layer center "
                            "\nOpacity given in [cm^2 g^-1].")
            file.writelines(
                "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
            )
            for i in range(quant.nlayer):
                file.writelines("{:<11}{:g}{:<4}".format("cloud_opac[", i, "]"))
            for x in range(quant.nbin):
                file.writelines("\n{:<8g}".format(x)
                                + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                )
                for i in range(quant.nlayer):
                    file.writelines("{:<17g}".format(quant.abs_cross_all_clouds_lay[x + quant.nbin * i]/quant.meanmolmass_lay[i]))

    @staticmethod
    def write_Rayleigh_cross_sections(quant, read):
        """ writes the scattering cross sections to a file. """

        with open(read.output_path + quant.name + "/" + quant.name + "_Rayleigh_cross_sect.dat", "w") as file:
            file.writelines("This file contains Rayleigh scattering cross sections "
                            "per wavelength at each layer center. "
                            "\nCross sections given in [cm^2].")
            file.writelines(
                "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
            )
            for i in range(quant.nlayer):
                file.writelines("{:<20}{:g}{:<4}".format("scat_cross_sect_lay[", i, "]"))
            for x in range(quant.nbin):
                file.writelines("\n{:<8g}".format(x)
                                + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                )
                for i in range(quant.nlayer):
                    file.writelines("{:<24g}".format(quant.scat_cross_lay[x + quant.nbin * i]))

    @staticmethod
    def write_cloud_scat_cross_sections(quant, read):
        """ writes the scattering cross sections of the cloud to a file. """

        with open(read.output_path + quant.name + "/" + quant.name + "_cloud_scat_cross_sect.dat", "w") as file:
            file.writelines("This file contains the cloud scattering cross sections "
                            "per wavelength at each layer center. "
                            "\nCross sections given in [cm^2].")
            file.writelines(
                "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
            )
            for i in range(quant.nlayer):
                file.writelines("{:<21}{:g}{:<4}".format("cloud_cross_sect_lay[", i, "]"))
            for x in range(quant.nbin):
                file.writelines("\n{:<8g}".format(x)
                                + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                )
                for i in range(quant.nlayer):
                    file.writelines("{:<25g}".format(quant.scat_cross_all_clouds_lay[x + quant.nbin * i]))

    @staticmethod
    def write_g_0(quant, read):
        """ writes the scattering asymmetry paramater values to a file. """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_g_0.dat", "w") as file:
                file.writelines("This file contains the scattering asymmetry parameter values per wavelength at each layer center."
                                "\nValues are between -1 and 1.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<8}{:g}{:<4}".format("g_0_lay[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<16g}".format(quant.g_0_tot_lay[x + quant.nbin * i]))
        except TypeError:
            print("Asymmetry parameter file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_transmission(quant, read):
        """ writes the transmission function for each layer to a file. """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_transmission.dat", "w") as file:
                file.writelines("This file contains the transmission function for each layer and waveband.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<11}{:g}{:<4}".format("transm_lay[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<18g}".format(quant.trans_band[x + i * quant.nbin]))
        except TypeError:
            print("Transmission file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_opt_depth(quant, read):
        """ writes the optical depth for each layer to a file. """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_optdepth.dat", "w") as file:
                file.writelines("This file contains the optical depth for each layer and waveband.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<14}{:g}{:<4}".format("delta_tau_lay[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<20g}".format(quant.delta_tau_band[x + i * quant.nbin]))
        except TypeError:
            print("Transmission file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_cloud_opt_depth(quant, read):
        """ writes the cloud optical depth for each layer to a file. """

        with open(read.output_path + quant.name + "/" + quant.name + "_cloud_optdepth.dat", "w") as file:
            file.writelines("This file contains the cloud optical depth for each layer and waveband.")
            file.writelines(
                "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
            )
            for i in range(quant.nlayer):
                file.writelines("{:<16}{:g}{:<4}".format("cloud_delta_tau[", i, "]"))
            for x in range(quant.nbin):
                file.writelines("\n{:<8g}".format(x)
                                + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                )
                for i in range(quant.nlayer):
                    file.writelines("{:<22g}".format(quant.delta_tau_all_clouds[x + i * quant.nbin]))

    @staticmethod
    def write_contribution_function(quant, read):
        """ writes the contribution function for each layer and waveband to a file. """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_contribution.dat", "w") as file:
                file.writelines("This file contains the contribution function for each layer and waveband.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<15}{:g}{:<4}".format("contr_func_lay[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<22g}".format(quant.contr_func_band[x + i * quant.nbin]))
        except TypeError:
            print("Contribution function file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_trans_weight_function(quant, read):
        """ writes the transmission weighting function for each layer and waveband to a file. """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_transweight.dat", "w") as file:
                file.writelines("This file contains the transmission weighting function for each layer and waveband. "
                                "The units are [erg s^-1 cm^-3 sr^-1]")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<18}{:g}{:<4}".format("transm_weight_lay[", i, "]"))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18.9g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21.9g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19.9g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<25g}".format(quant.trans_weight_band[x + i * quant.nbin]))
        except TypeError:
            print("Tranmission weighting function file generation corrupted. You might want to look into it!")

    def write_mean_extinction(self, quant, read):
        """ writes the Planck and Rosseland mean opacities & optical depths to a file. """
        try:
            with open(read.output_path + quant.name + "/" + quant.name + "_mean_extinct.dat", "w") as file:
                file.writelines("This file contains the Rosseland and Planck mean opacities of layers & optical depths "
                                "summed up to a certain layer, weighted either by the blackbody function "
                                "with the stellar or the planetary atmospheric temperature."
                                "\nMean opacity given in [cm^2 g^-1].")

                file.writelines("\n{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format(
                                "layer","press.[10^-6bar]",
                                "Planck_opac_T_lay","Ross_opac_T_lay",
                                "Planck_opac_T_star","Ross_opac_T_star",
                                "Planck_tau_T_lay","Ross_tau_T_lay",
                                "Planck_tau_T_star","Ross_tau_T_star")
                                )
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<8g}".format(i)
                        + "{:<20g}".format(quant.p_lay[i])
                        + self.write_mean_werror(quant.planck_opac_T_pl[i])
                        + self.write_mean_werror(quant.ross_opac_T_pl[i])
                        + self.write_mean_werror(quant.planck_opac_T_star[i])
                        + self.write_mean_werror(quant.ross_opac_T_star[i])
                        + self.write_mean_werror(hsfunc.sum_mean_optdepth(quant, i, quant.planck_opac_T_pl))
                        + self.write_mean_werror(hsfunc.sum_mean_optdepth(quant, i, quant.ross_opac_T_pl))
                        + self.write_mean_werror(hsfunc.sum_mean_optdepth(quant, i, quant.planck_opac_T_star))
                        + self.write_mean_werror(hsfunc.sum_mean_optdepth(quant, i, quant.ross_opac_T_star))
                    )
        except TypeError:
            print("Mean opacities and optical depths- file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_tp_for_coupling(quant, read):
        """ writes the TP-profile to a file """

        T_current = [quant.T_lay[i] for i in range(len(quant.T_lay) - 1)]
        T_current.insert(0, quant.T_lay[quant.nlayer])

        T_new = T_current

        if quant.coupling_speed_up == 1 and quant.coupling_iter_nr > 0:

            T_previous = []

            # read previous temperature profile
            if quant.coupling_full_output == 1:

                base_name = None

                # get the previous directory name
                for n in range(len(quant.name)-1, 0, -1):
                    if quant.name[n] == "_":
                        base_name = quant.name[:n + 1]
                        break

                previous_name = base_name + str(quant.coupling_iter_nr - 1)

                file_path_previous = read.output_path + previous_name + "/" + previous_name + "_tp_coupling_" + str(quant.coupling_iter_nr - 1) + ".dat"

            else:

                file_path_previous = read.output_path + quant.name + "/" + quant.name + "_tp_coupling_" + str(quant.coupling_iter_nr - 1) + ".dat"

            with open(file_path_previous, "r") as previous_file:
                next(previous_file)
                for line in previous_file:
                    column = line.split()
                    if len(column) > 1:
                        T_previous.append(quant.fl_prec(column[1]))

            # make average of current and previous temperature profiles
            T_average = [0.5 * T_current[i] + 0.5 * T_previous[i] for i in range(len(T_previous))]

            T_new = T_average

        with open(read.output_path + quant.name + "/" + quant.name + "_tp_coupling_" + str(quant.coupling_iter_nr) + ".dat", "w") as file:
            file.writelines(
                "{:<24}{:<18}".format("press.[10^-6bar]", "temp.[K]")
            )

            file.writelines("\n{:<24g}{:<18g}".format(quant.p_int[0], T_new[0]))

            for i in range(quant.nlayer):
                file.writelines(
                    "\n{:<24g}".format(quant.p_lay[i])
                    + "{:<18g}".format(T_new[i+1])
                )


if __name__ == "__main__":
    print("This module is for writing stuff and producing output. "
          "It consists of mostly temperature profiles, flux arrays... and occasional ice cream.")
