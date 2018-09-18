# ==============================================================================
# Module used to couple HELIOS and VULCAN (module still experimental!)
# Copyright (C) 2018 Matej Malik
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

import h5py
import numpy as npy
from scipy import interpolate
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from source import phys_const as pc

class Vcoupling(object):
    """ class that embodies everything needed to couple HELIOS and VULCAN """

    def __init__(self):

        self.V_iter_nr = 0 # may be overwritten by command-line input
        self.write_V_output = None
        self.mixfile_path = None
        
        # PyCUDA variables
        self.kernel_file = open("./source/Vmod.cu")
        self.kernels = self.kernel_file.read()
        self.mod = SourceModule(self.kernels)

    @staticmethod
    def read_ind_molecule_opac(quant, name):
        """ reads the file with the opacity table """

        opac_k = []
        opac_y = []
        opac_wave = []
        ktemp = []
        kpress = []

        try:
            with h5py.File(name, "r") as opac_file:
                for k in opac_file["kpoints"][:]:
                    opac_k.append(k)
                for y in opac_file["ypoints"][:]:
                    opac_y.append(y)
                for x in opac_file["center wavelengths"][:]:
                    opac_wave.append(x)
                for t in opac_file["temperatures"][:]:
                    ktemp.append(t)
                for p in opac_file["pressures"][:]:
                    kpress.append(p)
        except OSError:
            print("\nABORT - \" " + name + ", \" not found!")
            raise SystemExit()

        if len(kpress) == quant.npress and len(opac_wave) == quant.nbin and len(opac_y) == quant.ny:

            return opac_k

        else:
            print("The molecular opacity table " + name + " does not pass the dimension test. Aborting...")
            raise SystemExit()

    def read_molecular_opacities(self, quant):
        """ reads the individual molecular opacities """

        self.opac_k_h2o = self.read_ind_molecule_opac(quant, self.mol_opac_path+"h2o_opacities.h5")

        # with h5py.File(self.mol_opac_path+"watercontinuum.h5", "r") as conti_file:
        #     if quant.nbin == 300:
        #         for k in conti_file["300"][:]:
        #             self.opac_k_h2o_conti.append(k)
        #     elif quant.nbin == 3000:
        #         for k in conti_file["3000"][:]:
        #             self.opac_k_h2o_conti.append(k)

        # H2O continuum discontinued because exomol opacities do not require continuum contribution
        self.opac_k_h2o_conti = npy.zeros(quant.nbin)

        self.opac_k_co2 = self.read_ind_molecule_opac(quant, self.mol_opac_path+"co2_opacities.h5")
        self.opac_k_co = self.read_ind_molecule_opac(quant, self.mol_opac_path+"co_opacities.h5")
        self.opac_k_ch4 = self.read_ind_molecule_opac(quant, self.mol_opac_path+"ch4_opacities.h5")
        self.opac_k_nh3 = self.read_ind_molecule_opac(quant, self.mol_opac_path+"nh3_opacities.h5")
        self.opac_k_hcn = self.read_ind_molecule_opac(quant, self.mol_opac_path+"hcn_opacities.h5")
        self.opac_k_c2h2 = self.read_ind_molecule_opac(quant, self.mol_opac_path+"c2h2_opacities.h5")
        self.opac_k_cia_h2h2 = self.read_ind_molecule_opac(quant, self.mol_opac_path+"cia_H2-H2_opacities.h5")
        self.opac_k_cia_h2he = self.read_ind_molecule_opac(quant, self.mol_opac_path+"cia_H2-He_opacities.h5")

    @staticmethod
    def interpolate_to_own_press(old_press, old_array, new_press):

        new_array = interpolate.interp1d(old_press, old_array, bounds_error=False,
                                                     fill_value=(old_array[-1], old_array[0]))(new_press)

        return new_array

    def read_layer_molecular_abundance(self, quant):
        """ reads the molecular abundance from the VULCAN output file """

        mixfile = npy.genfromtxt(self.mixfile_path+"helios_mix.txt",
                                names=True, dtype=None, skip_header=0)

        # first check that dimensions and values correct
        chem_press = mixfile['P']

        own_press = [quant.p_boa * npy.exp(npy.log(quant.p_toa/quant.p_boa) * p / (quant.nlayer - 1.0))
                     for p in range(quant.nlayer)]

        quant.meanmolmass_lay = self.interpolate_to_own_press(chem_press, mixfile['mu'], own_press)
        quant.meanmolmass_lay = [mu * pc.AMU for mu in quant.meanmolmass_lay]

        self.f_h2o_lay = self.interpolate_to_own_press(chem_press, mixfile['H2O'], own_press)
        self.f_co2_lay = self.interpolate_to_own_press(chem_press, mixfile['CO2'], own_press)
        self.f_co_lay = self.interpolate_to_own_press(chem_press, mixfile['CO'], own_press)
        self.f_ch4_lay = self.interpolate_to_own_press(chem_press, mixfile['CH4'], own_press)
        self.f_nh3_lay = self.interpolate_to_own_press(chem_press, mixfile['NH3'], own_press)
        self.f_hcn_lay = self.interpolate_to_own_press(chem_press, mixfile['HCN'], own_press)
        self.f_c2h2_lay = self.interpolate_to_own_press(chem_press, mixfile['C2H2'], own_press)
        self.f_h2_lay = self.interpolate_to_own_press(chem_press, mixfile['H2'], own_press)
        self.f_he_lay = self.interpolate_to_own_press(chem_press, mixfile['He'], own_press)

    def interpolate_f_molecule_and_meanmolmass(self, quant):
        """ interpolates the layer mixing ratios and the mean molecular mass to the interfaces """

        temp_inter = self.mod.get_function("f_mol_and_meanmass_inter")

        temp_inter(self.dev_f_h2o_lay,
                   self.dev_f_co2_lay,
                   self.dev_f_co_lay,
                   self.dev_f_ch4_lay,
                   self.dev_f_nh3_lay,
                   self.dev_f_hcn_lay,
                   self.dev_f_c2h2_lay,
                   self.dev_f_h2_lay,
                   self.dev_f_he_lay,
                   self.dev_f_h2o_int,
                   self.dev_f_co2_int,
                   self.dev_f_co_int,
                   self.dev_f_ch4_int,
                   self.dev_f_nh3_int,
                   self.dev_f_hcn_int,
                   self.dev_f_c2h2_int,
                   self.dev_f_h2_int,
                   self.dev_f_he_int,
                   quant.dev_meanmolmass_lay,
                   quant.dev_meanmolmass_int,
                   quant.ninterface,
                   block=(16, 1, 1),
                   grid=((int(quant.ninterface)+15)//16, 1, 1)
                   )

        cuda.Context.synchronize()

    def interpolate_molecular_and_mixed_opac(self, quant):
        """ interpolates the molecular opacity tables to layer and interface values """

        opac_mol_mixed_interpol = self.mod.get_function("opac_mol_mixed_interpol")

        opac_mol_mixed_interpol(quant.dev_T_lay,
                          quant.dev_ktemp,
                          quant.dev_p_lay,
                          quant.dev_kpress,
                          quant.dev_opac_k,
                          self.dev_opac_k_h2o,
                          self.dev_opac_k_co2,
                          self.dev_opac_k_co,
                          self.dev_opac_k_ch4,
                          self.dev_opac_k_nh3,
                          self.dev_opac_k_hcn,
                          self.dev_opac_k_c2h2,
                          self.dev_opac_k_cia_h2h2,
                          self.dev_opac_k_cia_h2he,
                          quant.dev_opac_wg_lay,
                          self.dev_opac_h2o_wg_lay,
                          self.dev_opac_co2_wg_lay,
                          self.dev_opac_co_wg_lay,
                          self.dev_opac_ch4_wg_lay,
                          self.dev_opac_nh3_wg_lay,
                          self.dev_opac_hcn_wg_lay,
                          self.dev_opac_c2h2_wg_lay,
                          self.dev_opac_cia_h2h2_wg_lay,
                          self.dev_opac_cia_h2he_wg_lay,
                          quant.npress,
                          quant.ntemp,
                          quant.ny,
                          quant.nbin,
                          quant.nlayer,
                          block=(16, 16, 1),
                          grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                          )

        cuda.Context.synchronize()

        if quant.iso == 0:

            opac_mol_mixed_interpol(quant.dev_T_int,
                              quant.dev_ktemp,
                              quant.dev_p_int,
                              quant.dev_kpress,
                              quant.dev_opac_k,
                              self.dev_opac_k_h2o,
                              self.dev_opac_k_co2,
                              self.dev_opac_k_co,
                              self.dev_opac_k_ch4,
                              self.dev_opac_k_nh3,
                              self.dev_opac_k_hcn,
                              self.dev_opac_k_c2h2,
                              self.dev_opac_k_cia_h2h2,
                              self.dev_opac_k_cia_h2he,
                              quant.dev_opac_wg_int,
                              self.dev_opac_h2o_wg_int,
                              self.dev_opac_co2_wg_int,
                              self.dev_opac_co_wg_int,
                              self.dev_opac_ch4_wg_int,
                              self.dev_opac_nh3_wg_int,
                              self.dev_opac_hcn_wg_int,
                              self.dev_opac_c2h2_wg_int,
                              self.dev_opac_cia_h2h2_wg_int,
                              self.dev_opac_cia_h2he_wg_int,
                              quant.npress,
                              quant.ntemp,
                              quant.ny,
                              quant.nbin,
                              quant.ninterface,
                              block=(16, 16, 1),
                              grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                              )

            cuda.Context.synchronize()

    def combine_to_mixed_opacities(self, quant):
        """ combine the individual molecular opacities to layer/interface opacities """

        comb_opacities = self.mod.get_function("comb_opac")

        comb_opacities(self.dev_f_h2o_lay,
                       self.dev_f_co2_lay,
                       self.dev_f_co_lay,
                       self.dev_f_ch4_lay,
                       self.dev_f_nh3_lay,
                       self.dev_f_hcn_lay,
                       self.dev_f_c2h2_lay,
                       self.dev_f_h2_lay,
                       self.dev_f_he_lay,
                       self.dev_f_h2o_int,
                       self.dev_f_co2_int,
                       self.dev_f_co_int,
                       self.dev_f_ch4_int,
                       self.dev_f_nh3_int,
                       self.dev_f_hcn_int,
                       self.dev_f_c2h2_int,
                       self.dev_f_h2_int,
                       self.dev_f_he_int,
                       self.dev_opac_h2o_wg_lay,
                       self.dev_opac_co2_wg_lay,
                       self.dev_opac_co_wg_lay,
                       self.dev_opac_ch4_wg_lay,
                       self.dev_opac_nh3_wg_lay,
                       self.dev_opac_hcn_wg_lay,
                       self.dev_opac_c2h2_wg_lay,
                       self.dev_opac_cia_h2h2_wg_lay,
                       self.dev_opac_cia_h2he_wg_lay,
                       self.dev_opac_h2o_wg_int,
                       self.dev_opac_co2_wg_int,
                       self.dev_opac_co_wg_int,
                       self.dev_opac_ch4_wg_int,
                       self.dev_opac_nh3_wg_int,
                       self.dev_opac_hcn_wg_int,
                       self.dev_opac_c2h2_wg_int,
                       self.dev_opac_cia_h2h2_wg_int,
                       self.dev_opac_cia_h2he_wg_int,
                       quant.dev_opac_wg_lay,
                       quant.dev_opac_wg_int,
                       quant.dev_meanmolmass_lay,
                       quant.dev_meanmolmass_int,
                       quant.ny,
                       quant.nbin,
                       quant.nlayer,
                       block=(16, 16, 1),
                       grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                       )

        cuda.Context.synchronize()

    def combine_to_scat_cross(self, quant):
        """ create layer scattering cross-sections """

        comb_scat_cross = self.mod.get_function("comb_scat_cross")

        comb_scat_cross(self.dev_f_h2_lay,
                        self.dev_f_h2_int,
                        quant.dev_opac_scat_cross,
                        quant.dev_scat_cross_lay,
                        quant.dev_scat_cross_int,
                        quant.nbin,
                        quant.nlayer,
                        block=(16, 16, 1),
                        grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                        )

        cuda.Context.synchronize()

    def write_tp_VULCAN_temp(self, quant):
        """ writes the TP-profile to a file """
        try:
            with open("../tp_VULCAN.dat", "w") as file:
                file.writelines(
                    "{:<24}{:<18}".format("cent.press.[10^-6 bar]", "cent.temp.[K]")
                )
                for i in range(quant.nlayer):
                    if quant.p_lay[i] >= 1:
                        file.writelines(
                            "\n{:<24g}".format(quant.p_lay[i])
                            + "{:<18g}".format(quant.T_lay[i])
                        )
        except TypeError:
            print("VULCAN TP-file generation corrupted. You might want to look into it!")

    def write_tp_VULCAN(self, quant):
        """ writes the TP-profile to a file """
        try:
            with open("../" + str(self.V_iter_nr) + "_tp_VULCAN.dat", "w") as file:
                file.writelines(
                    "{:<24}{:<18}".format("cent.press.[10^-6 bar]", "cent.temp.[K]")
                )
                for i in range(quant.nlayer):
                    if quant.p_lay[i] >= 1:
                        file.writelines(
                            "\n{:<24g}".format(quant.p_lay[i])
                            + "{:<18g}".format(quant.T_lay[i])
                        )
        except TypeError:
            print("VULCAN TP-file generation corrupted. You might want to look into it!")

    def test_coupling_convergence(self, quant):
        """ test whether TP profile converged and ends coupled iteration """

        # read in temperatures from the last two iterations
        previous_temp = []
        current_temp = []

        with open("../" + str(self.V_iter_nr-1) + "_tp_VULCAN.dat", "r") as previous_file:
            next(previous_file)
            for line in previous_file:
                column = line.split()
                previous_temp.append(quant.fl_prec(column[1]))

        with open("../" + str(self.V_iter_nr) + "_tp_VULCAN.dat", "r") as current_file:
            next(current_file)
            for line in current_file:
                column = line.split()
                current_temp.append(quant.fl_prec(column[1]))

        # test for convergence
        limit = 1e-3

        converged_list = []

        for t in range(len(current_temp)):

            if abs(previous_temp[t] - current_temp[t]) / current_temp[t] < limit:

                converged_list.append(1)

        if len(converged_list) == len(current_temp):

            symbol_of_convergence = 1

        else:

            symbol_of_convergence = 0

        # write out result
        with open("../stop.dat", "w") as file:
            file.writelines(str(symbol_of_convergence))

    @staticmethod
    def Vrestart_file(read, quant):
        """ reads the restart temperatures from file """

        press_input = []

        try:
            with open(read.restart_path, "r") as restart_file:
                next(restart_file)
                for line in restart_file:
                    column = line.split()
                    quant.T_restart.append(column[0])
                    press_input.append(column[1])
        except IOError:
            print("ABORT - restart file not found!")
            raise SystemExit()

        # check if correct length
        if len(quant.T_restart) != quant.nlayer:
            print("ABORT - number of layers inconsistent between param file and restart file!")
            raise SystemExit()

        quant.p_boa = quant.fl_prec(press_input[0])
        quant.p_toa = quant.fl_prec(press_input[-1])


if __name__ == "__main__":
    print("This module provides the VULCAN coupling parameters and calculations. Please behave - VULCAN is usually ill-tempered.")
