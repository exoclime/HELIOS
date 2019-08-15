# ==============================================================================
# Module for writing the output quantities of HELIOS to files.
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

import numpy as npy
import os
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
    def write_mean_werror(quantity):
        """ writes an error message for quantities that could not be calculated """

        if quantity == -3:
            return "{:<20}".format("temp_too_low")
        else:
            return "{:<20g}".format(quantity)

    @staticmethod
    def write_abort_file(quant):
        """ writes a file that tells you that the run has been aborted due to exceeding iteration steps """

        # create directory if necessary
        try:
            os.makedirs("./output/" + quant.name)
        except OSError:
            if not os.path.isdir("./output/" + quant.name):
                raise

        try:
            with open("./output/" + quant.name + "/" + quant.name + "_ABORT.dat", "w") as file:
                file.writelines("The run had to be aborted as it exceeded the maximum number of iteration steps. Sorry.")
        except TypeError:
            print("ABORT file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_restart_file(quant):
        """ writes temporary output during the iteration process which can be used as restart profile """

        #  first we need to get the device arrays back to the host
        quant.T_lay = quant.dev_T_lay.get()
        quant.p_lay = quant.dev_p_lay.get()

        # create directory if necessary
        try:
            os.makedirs("./output/" + quant.name)
        except OSError:
            if not os.path.isdir("./output/" + quant.name):
                raise

        try:
            with open("./output/" + quant.name + "/" + quant.name + "_restart_tp.dat", "w") as file:
                file.writelines("This file contains the temporary (restart) central layer temperatures and pressures.")
                file.writelines(
                    "\n{:<8}{:<18}{:<24}".format(
                        "layer", "cent.temp.[K]", "cent.press.[10^-6bar]"
                    )
                )
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<8g}".format(i)
                        + "{:<18g}".format(quant.T_lay[i])
                        + "{:<24g}".format(quant.p_lay[i])
                    )
        except TypeError:
            print("Temporary/Restart file generation corrupted. You might want to look into it!")

    def write_info(self, quant, read, Vmod):
        """ writes the information file """

        print("\nWriting output files ... ")

        try:
            os.makedirs("./output/" + quant.name)
        except OSError:
            if not os.path.isdir("./output/" + quant.name):
                raise

        try:
            with open("./output/"+quant.name+"/" + quant.name + "_output_info.dat", "w") as file:
                file.write("""
H E L I O S   O U T P U T   I N F O R M A T I O N
=================================================

These are the parameters used in the production of this HELIOS output.
                """)
                file.writelines("\n")
                file.writelines("\n### GENERAL ###")
                file.writelines("\n")
                file.writelines("\nname = " + quant.name)
                file.writelines("\nprecision = " + quant.prec)
                file.writelines("\nrealtime plotting = " + self.convert_1_0_to_yes_no(quant.realtime_plot))
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n### GRID ###")
                file.writelines("\n")
                file.writelines("\nisothermal layers = " + self.convert_1_0_to_yes_no(quant.iso))
                file.writelines("\nnumber of layers = {:g}".format(quant.nlayer))
                file.writelines("\nTOA pressure [10^-6 bar] = {:g}".format(quant.p_toa))
                file.writelines("\nBOA pressure [10^-6 bar] = {:g}".format(quant.p_boa))
                file.writelines("\nplanet type = " + quant.planet_type)
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n### ITERATION ###")
                file.writelines("\n")
                file.writelines("\npost-processing only = " + self.convert_1_0_to_yes_no(quant.singlewalk))
                file.writelines("\npath to temperature file = " + read.temp_path)
                file.writelines("\ntemperature file format & P unit = " + read.temp_format+" "+read.temp_pressure_unit)
                file.writelines("\nvarying timestep = " + self.convert_1_0_to_yes_no(quant.varying_tstep))
                file.writelines("\ntimestep [s] = {:g}".format(quant.tstep))
                file.writelines("\nadaptive interval = {:g}".format(quant.adapt_interval))
                file.writelines("\nTP-profile smoothing = " + self.convert_1_0_to_yes_no(quant.smooth))
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n### RADIATION ###")
                file.writelines("\n")
                file.writelines("\ndirect irradiation beam = " + self.convert_1_0_to_yes_no(quant.dir_beam))
                file.writelines("\nscattering = " + self.convert_1_0_to_yes_no(quant.scat))
                file.writelines("\nimp. scattering corr. = " + self.convert_1_0_to_yes_no(quant.scat_corr))
                file.writelines("\nasymmetry factor g_0 = {:g}".format(quant.g_0))
                file.writelines("\npath to opacity file = " + read.ktable_path)
                file.writelines("\ndiffusivity factor = {:g}".format(quant.diffusivity))
                file.writelines("\nf factor = {:g}".format(quant.f_factor))
                file.writelines("\nstellar zenith angle [deg] = {:g}".format(180 - quant.dir_angle*180/npy.pi))
                file.writelines("\ngeom. zenith angle corr. = " + self.convert_1_0_to_yes_no(quant.geom_zenith_corr))
                file.writelines("\ninternal temperature [K] = {:g}".format(quant.T_intern))
                file.writelines("\nsurface temperature [K] = {:g}".format(quant.T_surf))
                file.writelines("\nsurface albedo = {:g}".format(quant.surf_albedo))
                file.writelines("\nenergy budget correction = " + self.convert_1_0_to_yes_no(quant.energy_correction))
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n### CONVECTIVE ADJUSTMENT ###")
                file.writelines("\n")
                file.writelines("\nconvective adjustment = " + self.convert_1_0_to_yes_no(quant.convection))
                file.writelines("\nkappa value = " + quant.kappa_manual_value)
                file.writelines("\nentropy/kappa file path = " + read.entr_kappa_path)
                file.writelines("\ndamping parameter = " + quant.input_dampara)
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n### ASTRONOMICAL PARAMETERS ###")
                file.writelines("\n")
                file.writelines("\nstellar spectral model = " + quant.stellar_model)
                file.writelines("\npath to stellar spectrum file = " + read.stellar_path)
                file.writelines("\n")
                file.writelines("\n--pre-tabulated--")
                file.writelines("\nplanet = " + quant.planet)
                file.writelines("\nsurface gravity [cm s^-2] = {:g}".format(quant.g))
                file.writelines("\norbital distance [AU] = {:g}".format(quant.a/pc.AU))
                file.writelines("\nradius planet [R_jup] = {:g}".format(quant.R_planet / pc.R_JUP))
                file.writelines("\nradius star [R_sun] = {:g}".format(quant.R_star/pc.R_SUN))
                file.writelines("\ntemperature star [K] = {:g}".format(quant.T_star))
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n############## DANGER ZONE ############## PROCEED UPON OWN RISK ##############")
                file.writelines("\n")
                file.writelines("\n### EXPERIMENTAL ###")
                file.writelines("\n")
                file.writelines("\nclouds = " + self.convert_1_0_to_yes_no(quant.clouds))
                file.writelines("\npath to cloud file = " + quant.cloud_path)
                file.writelines("\ntotal cloud opacity [cm^2 g^-1] = {:g}".format(quant.cloud_opac_tot))
                file.writelines("\ncloud center pressure [10^-6 bar] = {:g}".format(quant.cloud_press))
                file.writelines("\ncloud width (std.dev.) [dex] = {:g}".format(quant.cloud_width))
                file.writelines("\nnumber of run-in timesteps = {:g}".format(quant.foreplay))
                file.writelines("\nartificial shortw. opacity = {:g}".format(quant.fake_opac))
                file.writelines("\nuse f approximation formula = " + self.convert_1_0_to_yes_no(quant.approx_f))
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n### VULCAN COUPLING ###")
                file.writelines("\n")
                file.writelines("\nVULCAN coupling = " + self.convert_1_0_to_yes_no(Vmod.V_coupling))
                file.writelines("\npath to individual opacities = " + Vmod.mol_opac_path)
                file.writelines("\nspecies file = " + Vmod.species_file)
                file.writelines("\nmixing ratios file = " + Vmod.mix_file)
                file.writelines("\n---")
                file.writelines("\n")
                file.writelines("\n\n\nThank you for using HELIOS :-)")
                file.writelines("\n\nBy the way, this file can be again used as an input parameter file.\n"
                                "Just rename it and put it in the main directory.")
        except TypeError:
            print("Information file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_tp(quant):
        """ writes the TP-profile and the effective atmospheric temperature to a file """

        _, _, _, _, T_bright = hsfunc.temp_calcs(quant)

        try:
            with open("./output/"+quant.name+"/" + quant.name + "_tp.dat", "w") as file:
                file.writelines("This file contains the corresponding layer temperatures and pressures, and the altitude and the height of each layer.")
                file.writelines(
                    "\n{:<8}{:<18}{:<24}{:<21}{:<23}{:<30}{:<32}{:<18}{:<19}".format(
                        "layer", "cent.temp.[K]", "cent.press.[10^-6bar]", "cent.altitude[cm]", "height.of.layer[cm]",
                        "conv.unstable?[1:yes,0:no]", "conv.lapse-rate?[1:yes,0:no]", "pl.eff.temp.[K]", "pl.surf.temp.[K]")
                )
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<8g}".format(i)
                        + "{:<18g}".format(quant.T_lay[i])
                        + "{:<24g}".format(quant.p_lay[i])
                        + "{:<21g}".format(quant.z_lay[i])
                        + "{:<23g}".format(quant.delta_z_lay[i]))
                    if quant.iso == 0 and quant.convection == 1:
                        file.writelines("{:<30g}".format(quant.conv_unstable[i])
                                        + "{:<32g}".format(quant.conv_layer[i]))
                    if quant.iso == 1 or quant.convection == 0:
                        file.writelines("{:<30}".format("not_calculated")
                                        + "{:<32}".format("not_calculated"))
                    if i == 0:
                        file.writelines("{:<18g}".format(T_bright))
                        file.writelines("{:<18g}".format(quant.T_surf))
        except TypeError:
            print("TP-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_tp_pure(quant):
        """ writes the TP-profile to a file """
        try:
            with open("./output/"+quant.name+"/" + quant.name + "_tp_only.dat", "w") as file:
                file.writelines("This file contains the corresponding layer temperatures and pressures.")
                file.writelines(
                    "\n{:<8}{:<18}{:<24}".format(
                        "layer", "cent.temp.[K]", "cent.press.[10^-6bar]")
                )
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<8g}".format(i)
                        + "{:<18g}".format(quant.T_lay[i])
                        + "{:<24g}".format(quant.p_lay[i])
                    )
        except TypeError:
            print("Pure TP-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_tp_cut(quant):
        """ writes the TP-profile up to a height of P = 1e-6 bar to a file """

        _, _, _, _, T_bright = hsfunc.temp_calcs(quant)

        try:
            with open("./output/"+quant.name+"/" + quant.name + "_tp_cut.dat", "w") as file:
                file.writelines(
                    "This file contains the corresponding layer temperatures and pressures, and the altitude and the height of each layer.")
                file.writelines(
                    "\n{:<8}{:<18}{:<24}{:<21}{:<23}{:<30}{:<32}{:<18}{:<19}".format(
                        "layer", "cent.temp.[K]", "cent.press.[10^-6bar]", "cent.altitude[cm]", "height.of.layer[cm]",
                        "conv.unstable?[1:yes,0:no]", "conv.lapse-rate?[1:yes,0:no]", "pl.eff.temp.[K]", "pl.surf.temp.[K]")
                )
                for i in range(quant.nlayer):
                    if quant.p_lay[i] > 0.99:
                        file.writelines(
                            "\n{:<8g}".format(i)
                            + "{:<18g}".format(quant.T_lay[i])
                            + "{:<24g}".format(quant.p_lay[i])
                            + "{:<21g}".format(quant.z_lay[i])
                            + "{:<23g}".format(quant.delta_z_lay[i]))
                        if quant.iso == 0 and quant.convection == 1:
                            file.writelines("{:<30g}".format(quant.conv_unstable[i])
                                            + "{:<32g}".format(quant.conv_layer[i]))
                        if quant.iso == 1 or quant.convection == 0:
                            file.writelines("{:<30}".format("not_calculated")
                                            + "{:<32}".format("not_calculated"))
                        if i == 0:
                            file.writelines("{:<18g}".format(T_bright))
                            file.writelines("{:<18g}".format(quant.T_surf))
        except TypeError:
            print("Cut TP-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_colmass_mu_cp_entropy(quant):
        """ writes the layer column mass, mean molecular weight and specific heat capacity to a file """

        try:
            with open("./output/"+quant.name+"/" + quant.name + "_colmass_mu_cp_kappa_entropy.dat", "w") as file:
                file.writelines("This file contains the total pressure and the column mass difference, "
                                "mean molecular weight and specific heat capacity of each layer.")
                file.writelines(
                    "\n{:<8}{:<24}{:<26}{:<21}{:<32}{:<23}{:<30}".format(
                        "layer", "cent.press.[10^-6bar]", "delta_col.mass[g cm^-2]",
                        "mean mol. weight", "spec.heat cap.[erg g^-1 K^-1]", "adiabat. index kappa", "entropy [Boltzmann const.]")
                )
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<8g}".format(i)
                        + "{:<24g}".format(quant.p_lay[i])
                        + "{:<26g}".format(quant.delta_colmass[i])
                        + "{:<21g}".format(quant.meanmolmass_lay[i]/pc.AMU)
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
        except TypeError:
            print("Col_mass, mu, c_p file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_integrated_flux(quant):
        """ writes the integrated total and net fluxes to a file """
        try:
            with open("./output/"+quant.name+"/" + quant.name + "_integrated_flux.dat", "w") as file:
                file.writelines("This file contains the integrated total and net fluxes at each interface resp. "
                                "layer. \nFluxes given in [erg s^-1 cm^-2].")
                file.writelines(
                    "\n{:<20}{:<24}{:<25}{:<25}{:<23}{:<25}{:<44}{:<24}{:<12}".format(
                        "interface/layer", "cent.press.[10^-6bar]", "F_bol_down_at_interf.", "F_bol_up_at_interf.", "F_net_at_interf.", "F_bol_dir_at_interf.", "delta_F_net_at_cent. (only iterative run)", "F_net_conv_at_interf.", "F_intern")
                )
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<20g}".format(i)
                        + "{:<24g}".format(quant.p_lay[i])
                        + "{:<25g}".format(quant.F_down_tot[i])
                        + "{:<25g}".format(quant.F_up_tot[i])
                        + "{:<23g}".format(quant.F_net[i])
                        + "{:<25g}".format(quant.F_dir_tot[i]))
                    if quant.singlewalk == 0:
                        file.writelines("{:<44g}".format(quant.F_net_diff[i]))
                    else:
                        file.writelines("{:<44}".format("NA"))
                    file.writelines("{:<24g}".format(quant.F_net_conv[i]))
                    if i == 0:
                        file.writelines("{:<12g}".format(quant.F_intern))
                file.writelines(
                    "\n{:<20g}".format(quant.ninterface-1)
                    + "{:<24}".format("NA")
                    + "{:<25g}".format(quant.F_down_tot[quant.ninterface-1])
                    + "{:<25g}".format(quant.F_up_tot[quant.ninterface-1])
                    + "{:<23g}".format(quant.F_net[quant.ninterface-1]))
                if quant.singlewalk == 0:
                    file.writelines("{:<44}".format("NA"))
                file.writelines("{:<24g}".format(quant.F_net_conv[quant.ninterface-1]))
        except TypeError:
            print("Integrated flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_upward_spectral_flux(quant):
        """ writes the upward spectral flux to a file """
        try:
            with open("./output/"+quant.name+"/" + quant.name + "_spec_upflux.dat", "w") as file:
                file.writelines("This file contains the upward spectral flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.ninterface):
                    file.writelines("{:<15}{:<4g}".format("F_up_at_interf_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<19.8e}".format(quant.F_up_band[x + i * quant.nbin]))
        except TypeError:
            print("Upward spectral flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_downward_spectral_flux(quant):
        """ writes the downward spectral flux to a file """
        try:
            with open("./output/"+quant.name+"/" + quant.name + "_spec_downflux.dat", "w", encoding='utf-8') as file:
                file.writelines("This file contains the downward spectral flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.ninterface):
                    file.writelines("{:<17}{:<4g}".format("F_down_at_interf_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<21.8e}".format(quant.F_down_band[x + i * quant.nbin]))
        except TypeError:
            print("Downward spectral flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_TOA_flux_eclipse_depth(quant):
        """ writes the TOA fluxes to a file """
        try:
            with open("./output/"+quant.name+"/" + quant.name + "_TOA_flux_eclipse.dat", "w") as file:
                file.writelines("This file contains the downward and upward spectral flux (per wavelength) at TOA "
                                "and the secondary eclipse depth (= planet to star flux ratio)."
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:<21}{:<19}{:<16}{:<16}{:<24}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]", "F_down_at_TOA", "F_up_at_TOA", "planet/star flux ratio")
                )
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    file.writelines("{:<16g}".format(quant.F_down_band[x + quant.nlayer * quant.nbin])
                                    + "{:<16g}".format(quant.F_up_band[x + quant.nlayer * quant.nbin])
                                    )
                    if quant.T_star > 1:
                        file.writelines("{:<24g}".format(quant.F_ratio[x]))
                    else:
                        file.writelines("{:<24}".format("not avail."))
        except TypeError:
            print("TOA & BOA flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_flux_ratio_only(quant):
        """ writes only the planetary and stellar flux ratio to a file, e.g., to be readable by Pandexo """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_ratio_only.dat", "w") as file:
                for x in range(quant.nbin):
                    file.writelines("{:<12g}".format(quant.opac_wave[x] * 1e4))
                    if quant.T_star > 1:
                        file.writelines("{:<12g}\n".format(quant.F_ratio[x]))
                    else:
                        file.writelines("{:<12}\n".format("not avail."))
        except TypeError:
            print("flux ratio file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_TOA_flux_Ang(quant):
        """ writes the TOA flux in Angstrom """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_TOA_flux_Ang.dat", "w") as file:
                file.writelines("This file contains the upward spectral flux (per Angstrom) at TOA ."
                                "\nSpectral fluxes given in [erg s^-1 cm^-2 Angs.-1].")
                file.writelines(
                    "\n{:<8}{:<18}{:<21}{:<19}{:<16}".format("bin", "cent_lambda[Angs]", "low_int_lambda[Angs]",
                                                                         "delta_lambda[Angs]", "F_up_at_TOA")
                )
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e8)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e8)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e8)
                                    + "{:<16g}".format(quant.F_up_band[x + quant.nlayer * quant.nbin] * 1e-8)
                                    )
        except TypeError:
            print("TOA flux file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_direct_spectral_beam_flux(quant):
        """ writes the direct irradiation beam flux to a file """
        try:
            with open("./output/"+quant.name+"/" + quant.name + "_direct_beamflux.dat", "w") as file:
                file.writelines("This file contains the direct irradiation flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:18}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.ninterface):
                    file.writelines("{:<19}{:<4g}".format("F_direct_at_interf_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<23.8e}".format(quant.F_dir_band[x + i * quant.nbin]))
        except TypeError:
            print("Direct irradiation flux-file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_planck_interface(quant):
        """ writes the Planck function at interfaces to a file """
        if quant.iso == 0:
            try:
                with open("./output/"+quant.name+"/" + quant.name + "_planck_int.dat", "w") as file:
                    file.writelines("This file contains the Planck (blackbody) function at each interface. "
                                    "\nPlanck function given in [erg s^-1 cm^-3 sr^-1].")
                    file.writelines(
                        "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                    )
                    for i in range(quant.ninterface):
                        file.writelines("{:<17}{:<4g}".format("Planck_at_interf_", i))
                    for x in range(quant.nbin):
                        file.writelines("\n{:<8g}".format(x)
                                        + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                        + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                        + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                        )
                        for i in range(quant.ninterface):
                            file.writelines("{:<21g}".format(quant.planckband_int[i + x * quant.ninterface]))
            except TypeError:
                print("Planck-file (interfaces) generation corrupted. You might want to look into it!")

    @staticmethod
    def write_planck_center(quant):
        """ writes the Planck function at layer centers (+ stellar temp. and internal temp.) to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_planck_cent.dat", "w") as file:
                file.writelines("This file contains the Planck (blackbody) function at each layer center and "
                                "from the stellar (2nd last column) and internal (last column) temperatures. "
                                "\nPlanck function given in [erg s^-1 cm^-3 sr^-1].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<15}{:<4g}".format("Planck_at_cent_", i))
                file.writelines("{:<17}{:<17}".format("Planck_T_star", "Planck_T_intern")
                                )
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer+2):
                        file.writelines("{:<19g}".format(quant.planckband_lay[i + x * (quant.nlayer+2)]))
        except TypeError:
            print("Planck-file (layer centers) generation corrupted. You might want to look into it!")

    @staticmethod
    def write_opacities(quant):
        """ writes the bin integrated opacities and scattering cross sections to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_opacities.dat", "w") as file:
                file.writelines("This file contains the bin integrated opacities at each layer center "
                                "\nOpacity given in [cm^2 g^-1].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<17}{:<4g}".format("opacity_at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<21g}".format(quant.opac_band_lay[x + quant.nbin * i]))
        except TypeError:
            print("Opacity file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_Rayleigh_cross_sections(quant):
        """ writes the scattering cross sections to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_Rayleigh_cross_sect.dat", "w") as file:
                file.writelines("This file contains Rayleigh scattering cross sections "
                                "per wavelength at each layer center. "
                                "\nCross sections given in [cm^2].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<20}{:<4g}".format("cross_sect_at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<24g}".format(quant.scat_cross_lay[x + quant.nbin * i]))
        except TypeError:
            print("Scattering cross section file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_cloud_scat_cross_sections(quant):
        """ writes the scattering cross sections of the cloud to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_cloud_scat_cross_sect.dat", "w") as file:
                file.writelines("This file contains the cloud scattering cross sections "
                                "per wavelength at each layer center. "
                                "\nCross sections given in [cm^2].")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<25}{:<4g}".format("scat_cross_sect_at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<29g}".format(quant.cloud_scat_cross_lay[x + quant.nbin * i]))
        except TypeError:
            print("Cloud scattering cross section file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_cloud_absorption(quant):
        """ writes the absorption opacities of the cloud to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_cloud_absorption.dat", "w") as file:
                file.writelines("This file contains the cloud absorption opacities "
                                "at each layer center.")
                file.writelines(
                    "\n{:<8}{:<25}{:<22}".format("layer", "cent.press.[10^-6bar]", "opacity[cm^2 g^-1]")
                )
                for i in range(quant.nlayer):
                    file.writelines("\n{:<8g}".format(i)
                                    + "{:<25g}".format(quant.p_lay[i])
                                    + "{:<22g}".format(quant.cloud_opac_lay[i])
                                    )
        except TypeError:
            print("Cloud absorption file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_g_0(quant):
        """ writes the asymmetry paramater values to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_g_0.dat", "w") as file:
                file.writelines("This file contains the asymmetry parameter values per wavelength at each layer center."
                                "\nValues are between -1 and 1.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<13}{:<4g}".format("g_0_at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<17g}".format(quant.g_0_tot_lay[x + quant.nbin * i]))
        except TypeError:
            print("Asymmetry parameter file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_transmission(quant):
        """ writes the transmission function for each layer to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_transmission.dat", "w") as file:
                file.writelines("This file contains the transmission function for each layer and waveband.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<22}{:<4g}".format("transmission_at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<26g}".format(quant.trans_band[x + i * quant.nbin]))
        except TypeError:
            print("Transmission file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_opt_depth(quant):
        """ writes the optical depth for each layer to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_optdepth.dat", "w") as file:
                file.writelines("This file contains the optical depth for each layer and waveband.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<19}{:<4g}".format("delta_tau_at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<23g}".format(quant.delta_tau_band[x + i * quant.nbin]))
        except TypeError:
            print("Transmission file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_contribution_function(quant):
        """ writes the contribution function for each layer and waveband to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_contribution.dat", "w") as file:
                file.writelines("This file contains the contribution function for each layer and waveband.")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]","low_int_lambda[um]","delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<22}{:<4g}".format("contribution_at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<26g}".format(quant.contr_func_band[x + i * quant.nbin]))
        except TypeError:
            print("Contribution function file generation corrupted. You might want to look into it!")

    @staticmethod
    def write_trans_weight_function(quant):
        """ writes the transmission weighting function for each layer and waveband to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_transweight.dat", "w") as file:
                file.writelines("This file contains the transmission weighting function for each layer and waveband. "
                                "The units are [erg s^-1 cm^-3 sr^-1]")
                file.writelines(
                    "\n{:<8}{:<18}{:21}{:19}".format("bin", "cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]")
                )
                for i in range(quant.nlayer):
                    file.writelines("{:<23}{:<4g}".format("trans.weight._at_layer_", i))
                for x in range(quant.nbin):
                    file.writelines("\n{:<8g}".format(x)
                                    + "{:<18g}".format(quant.opac_wave[x] * 1e4)
                                    + "{:<21g}".format(quant.opac_interwave[x] * 1e4)
                                    + "{:<19g}".format(quant.opac_deltawave[x] * 1e4)
                                    )
                    for i in range(quant.nlayer):
                        file.writelines("{:<27g}".format(quant.trans_weight_band[x + i * quant.nbin]))
        except TypeError:
            print("Tranmission weighting function file generation corrupted. You might want to look into it!")

    def write_mean_extinction(self, quant):
        """ writes the Planck and Rosseland mean opacities & optical depths to a file. """
        try:
            with open("./output/" + quant.name + "/" + quant.name + "_mean_extinct.dat", "w") as file:
                file.writelines("This file contains the Rosseland and Planck mean opacities of layers & optical depths "
                                "summed up to a certain layer, weighted either by the blackbody function "
                                "with the stellar or the planetary atmospheric temperature."
                                "\nMean opacity given in [cm^2 g^-1].")

                file.writelines("\n{:<10}{:<24}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format(
                                "layer","cent.press.[10^-6bar]",
                                "Planck_opac_T_lay","Ross_opac_T_lay",
                                "Planck_opac_T_star","Ross_opac_T_star",
                                "Planck_tau_T_lay","Ross_tau_T_lay",
                                "Planck_tau_T_star","Ross_tau_T_star")
                                )
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:<8g}".format(i)
                        + "{:<24g}".format(quant.p_lay[i])
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
    def write_T10(quant):
        """ writes the temperature at 10 bar (which is a proxy to the interior entropy) to a file. """

        l = k = j = -2

        # get layer with 10 bar
        for i in range(len(quant.p_lay)):
            if 0.99e9 < quant.p_lay[i] < 1.01e9:
                l = i
            elif 0.99e8 < quant.p_lay[i] < 1.01e8:
                k = i
            elif 0.99e7 < quant.p_lay[i] < 1.01e7:
                j = i

        # check that all layers below j, k, l are convective
        conv_answer_10 = "yes"
        try:
            for i in range(j + 1):
                if quant.conv_layer[i] == 0:
                    conv_answer_10 = "no"
        except:
            pass

        conv_answer_100 = "yes"
        try:
            for i in range(k + 1):
                if quant.conv_layer[i] == 0:
                    conv_answer_100 = "no"
        except:
            pass

        conv_answer_1000 = "yes"
        try:
            for i in range(l + 1):
                if quant.conv_layer[i] == 0:
                    conv_answer_1000 = "no"
        except:
            pass

        try:
            with open("./output/" + quant.name + "/" + quant.name + "_T10.dat", "w") as file:
                file.writelines("This file contains the T_10 temperature (which is an entropy proxy) "
                                "together with the corresponding planetary parameters used for the model.")
                file.writelines("\nPARAMETERS")
                file.writelines("\ninternal temperature [K] = {:g}".format(quant.T_intern))
                file.writelines("\nsurface gravity [dex(cgs)] = {:g}".format(npy.log10(quant.g)))
                file.writelines("\ntemperature star [K] = {:g}".format(quant.T_star))
                file.writelines("\nENTROPY PROXY")
                if j >= 0:
                    file.writelines("\ntemperature at 10 bar [K] = {:g}".format(quant.T_lay[j]))
                    file.writelines("\nis 10 bar within convective zone?  " + conv_answer_10)
                if k >= 0:
                    file.writelines("\ntemperature at 100 bar [K] = {:g}".format(quant.T_lay[k]))
                    file.writelines("\nis 100 bar within convective zone?  " + conv_answer_100)
                if l >= 0:
                    file.writelines("\ntemperature at 1000 bar [K] = {:g}".format(quant.T_lay[l]))
                    file.writelines("\nis 1000 bar within convective zone?  " + conv_answer_1000)
        except TypeError:
            print("T10-file generation corrupted. You might want to look into it!")


if __name__ == "__main__":
    print("This module is for writing stuff and producing output. "
          "It consists of mostly temperature profiles, flux arrays... and occasional ice cream.")
