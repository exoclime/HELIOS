# ==============================================================================
# Module for writing the output quantities of HELIOS to files.
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

import phys_const as pc
import host_functions as hsfunc

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

    def write_restart_file(self, quant):
        """ writes temporary output during the iteration process which can be used as restart profile """

        #  first we need to get the device arrays back to the host
        quant.T_lay = quant.dev_T_lay.get()
        quant.p_lay = quant.dev_p_lay.get()

        try:
            with open("./output/" + quant.name + "_restart_tp.dat", "w") as file:
                file.writelines("This file contains the temporary (restart) layer temperatures and pressures.")
                file.writelines("\nlayer nr.    center temperature [K]    center pressure [10^-6 bar]")
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:g}".format(i) + "    {:g}".format(quant.T_lay[i]) + "    {:g}".format(quant.p_lay[i])
                    )
        except TypeError:
            print("Temporary/Restart file generation corrupted. You might want to look into it!")

    def write_info(self, quant, read):
        """ writes the information file """
        try:
            with open("./output/HELIOS_info.dat", "w") as file:
                file.write("""
H E L I O S   O U T P U T   I N F O R M A T I O N
=================================================

These are the parameters used in the production of this HELIOS output.
                """)
                file.writelines("\n---")
                file.writelines("\nGRID")
                file.writelines("\nisothermal layers = "+self.convert_1_0_to_yes_no(quant.iso))
                file.writelines("\nnumber of layers = {:g}".format(quant.nlayer))
                file.writelines("\nTOA pressure [cgs] = {:g}".format(quant.p_toa))
                file.writelines("\nBOA pressure [cgs] = {:g}".format(quant.p_boa))
                file.writelines("\npre-tabulate = "+self.convert_1_0_to_yes_no(quant.tabu))
                file.writelines("\n---")
                file.writelines("\nITERATION")
                file.writelines("\npost-processing only = "+self.convert_1_0_to_yes_no(quant.singlewalk))
                file.writelines("\nrestart temperatures = "+self.convert_1_0_to_yes_no(quant.restart))
                file.writelines("\npath to restart temperatures = "+read.restart_path)
                file.writelines("\nvarying timestep = "+self.convert_1_0_to_yes_no(quant.varying_tstep))
                file.writelines("\ntimestep [s] = {:g}".format(quant.tstep))
                file.writelines("\n---")
                file.writelines("\nRADIATION")
                file.writelines("\nscattering = "+self.convert_1_0_to_yes_no(quant.scat))
                file.writelines("\nexact solution = "+self.convert_1_0_to_yes_no(quant.direct))
                file.writelines("\npath to opacity file = "+read.ktable_path)
                file.writelines("\ndiffusivity factor = {:g}".format(quant.diffusivity))
                file.writelines("\nf factor = {:g}".format(quant.f_factor))
                file.writelines("\ninternal temperature [K] = {:g}".format(quant.T_intern))
                file.writelines("\nasymmetry factor g_0 = {:g}".format(quant.g_0))
                file.writelines("\n---")
                file.writelines("\nORBITAL/PLANETARY PARAMETERS")
                file.writelines("\nmean molecular weight [m_p] = {:g}".format(quant.mu))
                file.writelines("\nplanet = "+quant.planet)
                file.writelines("\nsurface gravity [cgs] = {:g}".format(quant.g))
                file.writelines("\norbital distance [AU] = {:g}".format(quant.a/pc.AU))
                file.writelines("\nradius star [R_sun] = {:g}".format(quant.R_star/pc.R_SUN))
                file.writelines("\ntemperature star [K] = {:g}".format(quant.T_star))
                file.writelines("\n---")
                file.writelines("\nSTELLAR MODEL")
                file.writelines("\nmodel = "+quant.model)
                file.writelines("\npath to stellar model file = "+read.stellar_path)
                file.writelines("\n---")
                file.writelines("\nMISCELLANEOUS")
                file.writelines("\nname = "+quant.name)
                file.writelines("\nnumber of run-in timesteps = {:g}".format(quant.foreplay))
                file.writelines("\nartificial shortw. opacity = {:g}".format(quant.fake_opac))
                file.writelines("\nrealtime plotting = "+self.convert_1_0_to_yes_no(quant.realtime_plot))
                file.writelines("\n\n\nThank you for using HELIOS :)")
                file.writelines("\n\nBy the way, this file can be again used as an input parameter file.\n"
                                "Just rename it and put it in the main directory.")
        except TypeError:
            print("Information file generation corrupted. You might want to look into it!")

    def write_tp(self, quant):
        """ writes the TP-profile to a file """
        try:
            with open("./output/"+quant.name+"_tp.dat", "w") as file:
                file.writelines("This file contains the corresponding layer temperatures and pressures.")
                file.writelines("\nlayer nr.    center temperature [K]    center pressure [10^-6 bar]")
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:g}".format(i)+"    {:g}".format(quant.T_lay[i])+"    {:g}".format(quant.p_lay[i])
                    )
        except TypeError:
            print("TP-file generation corrupted. You might want to look into it!")

    def write_column_mass(self, quant):
        """ writes the column mass to a file """
        try:
            with open("./output/"+quant.name+"_colmass.dat", "w") as file:
                file.writelines("This file contains the total pressure and the column mass difference at each layer.")
                file.writelines("\nlayer nr.    center pressure [10^-6 bar]"
                                "    layer column mass difference [g cm^-2]")
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:g}".format(i)+"    {:g}".format(quant.p_lay[i])+"    {:g}".format(quant.delta_colmass[i])
                    )
        except TypeError:
            print("Column mass file generation corrupted. You might want to look into it!")

    def write_integrated_flux(self, quant):
        """ writes the integrated total and net fluxes to a file """
        if quant.singlewalk == 0:
            try:
                with open("./output/"+quant.name+"_integrated_flux.dat", "w") as file:
                    file.writelines("This file contains the integrated total and net fluxes at each interface resp. layer. "
                                    "\nFluxes given in [erg s^-1 cm^-2].")
                    file.writelines("\ninterface/layer nr.    center pressure [10^-6 bar]    "
                                    "total downward flux at interf.    total upward flux at interf.    net flux at layer")
                    for i in range(quant.nlayer):
                        file.writelines(
                            "\n{:g}".format(i)+"    {:g}".format(quant.p_lay[i])+"    {:g}".format(quant.Fdown_tot[i])
                            +"    {:g}".format(quant.Fup_tot[i])+"    {:g}".format(quant.F_net[i])
                        )
                    file.writelines(
                        "\n{:g}".format(quant.ninterface-1)
                        +"    not_available    {:g}".format(quant.Fdown_tot[quant.ninterface-1])
                        +"    {:g}".format(quant.Fup_tot[quant.ninterface-1])+"    not_available"
                    )
            except TypeError:
                print("Integrated flux-file generation corrupted. You might want to look into it!")

    def write_upward_spectral_flux(self, quant):
        """ writes the upward spectral flux to a file """
        try:
            with open("./output/"+quant.name+"_spec_upflux.dat", "w") as file:
                file.writelines("This file contains the upward spectral flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines("\nbin nr.    center wavelength [micron]")
                for i in range(quant.ninterface):
                    file.writelines("    upward flux at interface {:g}".format(i))
                for x in range(quant.nbin):
                    file.writelines("\n{:g}".format(x)+"    {:g}".format(quant.opac_wave[x] * 1e4))
                    for i in range(quant.ninterface):
                        file.writelines("    {:g}".format(quant.Fup_band[x + i * quant.nbin]))
        except TypeError:
            print("Upward spectral flux-file generation corrupted. You might want to look into it!")

    def write_downward_spectral_flux(self, quant):
        """ writes the downward spectral flux to a file """
        try:
            with open("./output/"+quant.name+"_spec_downflux.dat", "w") as file:
                file.writelines("This file contains the downward spectral flux (per wavelength) at each interface. "
                                "\nSpectral fluxes given in [erg s^-1 cm^-3].")
                file.writelines("\nbin nr.    center wavelength [micron]")
                for i in range(quant.ninterface):
                    file.writelines("    downward flux at interface {:g}".format(i))
                for x in range(quant.nbin):
                    file.writelines("\n{:g}".format(x)+"    {:g}".format(quant.opac_wave[x] * 1e4))
                    for i in range(quant.ninterface):
                        file.writelines("    {:g}".format(quant.Fdown_band[x + i * quant.nbin]))
        except TypeError:
            print("Downward spectral flux-file generation corrupted. You might want to look into it!")

    def write_planck_interface(self, quant):
        """ writes the Planck function at interfaces to a file """
        if quant.iso == 0:
            try:
                with open("./output/"+quant.name+"_planck_int.dat", "w") as file:
                    file.writelines("This file contains the Planck (blackbody) function at each interface. "
                                    "\nPlanck function given in [erg s^-1 cm^-3 sr^-1].")
                    file.writelines("\nbin nr.    center wavelength [micron]")
                    for i in range(quant.ninterface):
                        file.writelines("    planck funct. at interface {:g}".format(i))
                    for x in range(quant.nbin):
                        file.writelines("\n{:g}".format(x)+"    {:g}".format(quant.opac_wave[x] * 1e4))
                        for i in range(quant.ninterface):
                            file.writelines("    {:g}".format(quant.planckband_int[i + x * quant.ninterface]))
            except TypeError:
                print("Planck-file (interfaces) generation corrupted. You might want to look into it!")

    def write_planck_center(self, quant):
        """ writes the Planck function at layer centers (+ stellar temp. and internal temp.) to a file. """
        try:
            with open("./output/" + quant.name + "_planck_cent.dat", "w") as file:
                file.writelines("This file contains the Planck (blackbody) function at each layer center and "
                                "from the stellar (2nd last column) and internal (last column) temperatures. "
                                "\nPlanck function given in [erg s^-1 cm^-3 sr^-1].")
                file.writelines("\nbin nr.    center wavelength [micron]")
                for i in range(quant.nlayer):
                    file.writelines("    planck funct. at layer {:g}".format(i))
                    file.writelines("    planck funct., stellar T    planck funct., internal T")
                for x in range(quant.nbin):
                    file.writelines("\n{:g}".format(x) + "    {:g}".format(quant.opac_wave[x] * 1e4))
                    for i in range(quant.nlayer+2):
                        file.writelines("    {:g}".format(quant.planckband_lay[i + x * (quant.nlayer+2)]))
        except TypeError:
            print("Planck-file (layer centers) generation corrupted. You might want to look into it!")

    def write_opacities(self, quant):
        """ writes the bin integrated opacities and scattering cross sections to a file. """
        try:
            with open("./output/" + quant.name + "_opacities.dat", "w") as file:
                file.writelines("This file contains the bin integrated opacities at each layer center "
                                "as well as the Rayleigh scattering cross sections. "
                                "\nOpacity given in [cm^2 g^-1].")
                file.writelines("\nbin nr.    center wavelength [micron]")
                for i in range(quant.nlayer):
                    file.writelines("    opacity at layer {:g}".format(i))
                file.writelines("    Rayleigh cross sect. [cm^2]")
                for x in range(quant.nbin):
                    file.writelines("\n{:g}".format(x) + "    {:g}".format(quant.opac_wave[x] * 1e4))
                    for i in range(quant.nlayer):
                        file.writelines("    {:g}".format(quant.opac_lay[i + x * quant.nlayer]))
                    file.writelines("    {:g}".format(quant.cross_scat[x]))
        except TypeError:
            print("Opacity file generation corrupted. You might want to look into it!")


    def write_transmission(self, quant):
        """ writes the transmission function for each layer to a file. """
        try:
            with open("./output/" + quant.name + "_transmission.dat", "w") as file:
                file.writelines("This file contains the transmission function for each layer.")
                file.writelines("\nbin nr.    center wavelength [micron]")
                for i in range(quant.nlayer):
                    file.writelines("    transm. funct. for layer {:g}".format(i))
                for x in range(quant.nbin):
                    file.writelines("\n{:g}".format(x) + "    {:g}".format(quant.opac_wave[x] * 1e4))
                    for i in range(quant.nlayer):
                        file.writelines("    {:g}".format(quant.transmission[x + i * quant.nbin]))
        except TypeError:
            print("Transmission file generation corrupted. You might want to look into it!")

    def write_mean_extinction(self, quant):
        """ writes the Planck and Rosseland mean opacities & optical depths to a file. """
        try:
            with open("./output/" + quant.name + "_mean_extinct.dat", "w") as file:
                file.writelines("This file contains the Rosseland and Planck mean opacities of layers & optical depths "
                                "summed up to a certain layer, weighted either by the blackbody function "
                                "with the stellar or the planetary atmospheric temperature."
                                "\nMean opacity given in [cm^2 g^-1].")
                file.writelines("\nlayer nr.    center pressure [10^-6 bar]    "
                                "Planck mean opac., atmo. T    Ross. mean opac., atmo. T    "
                                "Planck mean opac., stellar T    Ross. mean opac., stellar T    "
                                "Planck mean opt.depth, atmo. T    Ross. mean opt.depth, atmo. T    "
                                "Planck mean opt.depth, stellar T    Ross. mean opt.depth, stellar T")
                for i in range(quant.nlayer):
                    file.writelines(
                        "\n{:g}".format(i) + "    {:g}".format(quant.p_lay[i])
                        + "    {:g}".format(quant.planck_opac_T_pl[i]) + "    {:g}".format(quant.ross_opac_T_pl[i])
                        + "    {:g}".format(quant.planck_opac_T_star[i]) + "    {:g}".format(quant.ross_opac_T_star[i])
                        + "    {:g}".format(hsfunc.sum_mean_optdepth(quant, i, quant.planck_opac_T_pl))
                        + "    {:g}".format(hsfunc.sum_mean_optdepth(quant, i, quant.ross_opac_T_pl))
                        + "    {:g}".format(hsfunc.sum_mean_optdepth(quant, i, quant.planck_opac_T_star))
                        + "    {:g}".format(hsfunc.sum_mean_optdepth(quant, i, quant.ross_opac_T_star))
                    )
        except TypeError:
            print("Mean opacities and optical depths- file generation corrupted. You might want to look into it!")


if __name__ == "__main__":
    print("This module is for writing stuff and producing output. "
          "It consists of mostly temperature profiles, flux arrays... and occasional ice cream.")
