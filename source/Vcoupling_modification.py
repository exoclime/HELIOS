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
        self.V_coupling = 0
        self.mol_opac_path = None
        self.species_file = None
        self.mix_file = None
        self.V_iter_nr = 0  # overwritten by read in
        self.species = []
        # PyCUDA variables
        self.kernel_file = open("./source/Vmod.cu")
        self.kernels = self.kernel_file.read()
        self.mod = SourceModule(self.kernels)

    @staticmethod
    def read_ray_scat_cross(name):

        h2_scat = []
        opac_wave = []
        ktemp = []
        kpress = []

        try:
            with h5py.File(name, "r") as opac_file:
                for k in opac_file["pure H2 Rayleigh cross-sections"][:]:
                    h2_scat.append(k)
                for x in opac_file["wavelengths"][:]:
                    opac_wave.append(x)
                for t in opac_file["temperatures"][:]:
                    ktemp.append(t)
                for p in opac_file["pressures"][:]:
                    kpress.append(p)
        except OSError:
            print("\nABORT - \" " + name + ", \" not found!")
            raise SystemExit()

        return h2_scat

    @staticmethod
    def read_ind_molecule_opac(quant, name, read_grid_parameters=False):
        """ reads the file with the opacity table """

        with h5py.File(name, "r") as opac_file:

            try:
                opac_k = [k for k in opac_file["kpoints"][:]]
            except KeyError:
                opac_k = [k for k in opac_file["opacities"][:]]

            if read_grid_parameters is True:

                # wavelength grid
                try:
                    quant.opac_wave = [x for x in opac_file["center wavelengths"][:]]
                except KeyError:
                    quant.opac_wave = [x for x in opac_file["wavelengths"][:]]
                quant.nbin = npy.int32(len(quant.opac_wave))

                # Gaussian y-points
                try:
                    quant.opac_y = [y for y in opac_file["ypoints"][:]]
                except KeyError:
                    quant.opac_y = [0]
                quant.ny = npy.int32(len(quant.opac_y))

                # interface positions of the wavelength bins
                try:
                    quant.opac_interwave = [i for i in opac_file["interface wavelengths"][:]]
                except KeyError:
                    # quick and dirty way to get the lamda interface values
                    quant.opac_interwave = []
                    quant.opac_interwave.append(quant.opac_wave[0] - (quant.opac_wave[1] - quant.opac_wave[0]) / 2)
                    for x in range(len(quant.opac_wave) - 1):
                        quant.opac_interwave.append((quant.opac_wave[x + 1] + quant.opac_wave[x]) / 2)
                    quant.opac_interwave.append(quant.opac_wave[-1] + (quant.opac_wave[-1] - quant.opac_wave[-2]) / 2)

                # widths of the wavelength bins
                try:
                    quant.opac_deltawave = [w for w in opac_file["wavelength width of bins"][:]]
                except KeyError:
                    quant.opac_deltawave = []
                    for x in range(len(quant.opac_interwave) - 1):
                        quant.opac_deltawave.append(quant.opac_interwave[x + 1] - quant.opac_interwave[x])

                # temperature grid
                quant.ktemp = [t for t in opac_file["temperatures"][:]]
                quant.ntemp = npy.int32(len(quant.ktemp))

                # pressure grid
                quant.kpress = [p for p in opac_file["pressures"][:]]
                quant.npress = npy.int32(len(quant.kpress))

            return opac_k

    @staticmethod
    def fill_array_with_zeros_if_empty(quant, array):

        if array is None:
            array = npy.zeros(quant.ny * quant.nbin * quant.npress * quant.ntemp)

        return array

    def read_species(self):

        if self.V_iter_nr > 0:

            with open(self.species_file, "r", encoding='utf-8') as species_file:

                for s in list(species_file):
                    s = s.strip('\n').strip()
                    self.species.append(s)

    def read_molecular_opacities(self, quant):
        """ reads the individual molecular opacities """

        if self.V_iter_nr > 0:

            if "H2O" in self.species:
                quant.opac_k_h2o = self.read_ind_molecule_opac(quant, self.mol_opac_path+"H2O_opac_ip.h5", read_grid_parameters=True)

            if "CO2" in self.species:
                quant.opac_k_co2 = self.read_ind_molecule_opac(quant, self.mol_opac_path+"CO2_opac_ip.h5")

            if "CO" in self.species:
                quant.opac_k_co = self.read_ind_molecule_opac(quant, self.mol_opac_path+"CO_opac_ip.h5")

            if "CH4" in self.species:
                quant.opac_k_ch4 = self.read_ind_molecule_opac(quant, self.mol_opac_path + "CH4_opac_ip.h5")

            if "NH3" in self.species:
                quant.opac_k_nh3 = self.read_ind_molecule_opac(quant, self.mol_opac_path + "NH3_opac_ip.h5")

            if "HCN" in self.species:
                quant.opac_k_hcn = self.read_ind_molecule_opac(quant, self.mol_opac_path + "HCN_opac_ip.h5")

            if "C2H2" in self.species:
                quant.opac_k_c2h2 = self.read_ind_molecule_opac(quant, self.mol_opac_path + "C2H2_opac_ip.h5")

            if "TiO" in self.species:
                quant.opac_k_tio = self.read_ind_molecule_opac(quant, self.mol_opac_path + "TiO_opac_ip.h5")

            if "VO" in self.species:
                quant.opac_k_vo = self.read_ind_molecule_opac(quant, self.mol_opac_path + "VO_opac_ip.h5")

            if "CIA_H2H2" in self.species:
                quant.opac_k_cia_h2h2 = self.read_ind_molecule_opac(quant, self.mol_opac_path + "CIA_H2H2_opac_ip.h5")

            if "CIA_H2He" in self.species:
                quant.opac_k_cia_h2he = self.read_ind_molecule_opac(quant, self.mol_opac_path + "CIA_H2He_opac_ip.h5")

            quant.opac_scat_cross = self.read_ray_scat_cross(self.mol_opac_path+"Rayleigh.h5")

            quant.opac_k_h2o = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_h2o)

            quant.opac_k_co2 = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_co2)
            quant.opac_k_co = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_co)
            quant.opac_k_ch4 = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_ch4)
            quant.opac_k_nh3 = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_nh3)
            quant.opac_k_hcn = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_hcn)
            quant.opac_k_c2h2 = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_c2h2)
            quant.opac_k_tio = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_tio)
            quant.opac_k_vo = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_vo)
            quant.opac_k_cia_h2h2 = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_cia_h2h2)
            quant.opac_k_cia_h2he = self.fill_array_with_zeros_if_empty(quant, quant.opac_k_cia_h2he)

    def try_read_and_interpolate_to_own_press_grid(self, file, name, old_press, new_press):

        try:
            old_array = file[name]

            new_array = interpolate.interp1d(old_press, old_array, bounds_error=False,
                                                     fill_value=(old_array[-1], old_array[0]))(new_press)
        except ValueError:
            if name == 'mu':
                print("\nFailed to read-in 'mu' data from " + file + ". Can't do without them, sorry! Aborting...")
                raise SystemExit()
            else:
                print("\nWarning: No chemistry data found for " + name + " in " + self.mix_file + ". Setting respective mixing ratios to zero.")
                new_array = npy.zeros(len(new_press))

        return new_array

    def read_layer_molecular_abundance(self, quant):
        """ reads the molecular abundance from the VULCAN output file """

        if self.V_iter_nr > 0:

            mixfile = npy.genfromtxt(self.mix_file,
                                    names=True, dtype=None, skip_header=1)

            chem_press = mixfile['Pressure']

            own_press = [quant.p_boa * npy.exp(npy.log(quant.p_toa/quant.p_boa) * p / (quant.nlayer - 1.0))
                         for p in range(quant.nlayer)]

            quant.meanmolmass_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "mu", chem_press, own_press)
            quant.meanmolmass_lay = [mu * pc.AMU for mu in quant.meanmolmass_lay]
            quant.f_h2o_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "H2O", chem_press, own_press)
            quant.f_co2_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "CO2", chem_press, own_press)
            quant.f_co_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "CO", chem_press, own_press)
            quant.f_ch4_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "CH4", chem_press, own_press)
            quant.f_nh3_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "NH3", chem_press, own_press)
            quant.f_hcn_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "HCN", chem_press, own_press)
            quant.f_c2h2_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "C2H2", chem_press, own_press)
            quant.f_tio_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "TiO", chem_press, own_press)
            quant.f_vo_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "VO", chem_press, own_press)
            quant.f_h2_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "H2", chem_press, own_press)
            quant.f_he_lay = self.try_read_and_interpolate_to_own_press_grid(mixfile, "He", chem_press, own_press)

    def interpolate_f_molecule_and_meanmolmass(self, quant):
        """ interpolates the layer mixing ratios and the mean molecular mass to the interfaces """

        if self.V_iter_nr > 0:

            if quant.iso == 0:

                fmol_inter = self.mod.get_function("f_mol_and_meanmass_inter")

                fmol_inter(quant.dev_f_h2o_lay,
                           quant.dev_f_co2_lay,
                           quant.dev_f_co_lay,
                           quant.dev_f_ch4_lay,
                           quant.dev_f_nh3_lay,
                           quant.dev_f_hcn_lay,
                           quant.dev_f_c2h2_lay,
                           quant.dev_f_h2_lay,
                           quant.dev_f_he_lay,
                           quant.dev_f_h2o_int,
                           quant.dev_f_co2_int,
                           quant.dev_f_co_int,
                           quant.dev_f_ch4_int,
                           quant.dev_f_nh3_int,
                           quant.dev_f_hcn_int,
                           quant.dev_f_c2h2_int,
                           quant.dev_f_tio_int,
                           quant.dev_f_vo_int,
                           quant.dev_f_h2_int,
                           quant.dev_f_he_int,
                           quant.dev_meanmolmass_lay,
                           quant.dev_meanmolmass_int,
                           quant.ninterface,
                           block=(16, 1, 1),
                           grid=((int(quant.ninterface)+15)//16, 1, 1)
                           )

                cuda.Context.synchronize()

    def interpolate_molecular_and_mixed_opac(self, quant):
        """ interpolates the molecular opacity tables to layer and interface values """

        if self.V_iter_nr > 0:

            opac_mol_mixed_interpol = self.mod.get_function("opac_mol_mixed_interpol")

            opac_mol_mixed_interpol(quant.dev_T_lay,
                              quant.dev_ktemp,
                              quant.dev_p_lay,
                              quant.dev_kpress,
                              quant.dev_opac_k,
                              quant.dev_opac_k_h2o,
                              quant.dev_opac_k_co2,
                              quant.dev_opac_k_co,
                              quant.dev_opac_k_ch4,
                              quant.dev_opac_k_nh3,
                              quant.dev_opac_k_hcn,
                              quant.dev_opac_k_c2h2,
                              quant.dev_opac_k_tio,
                              quant.dev_opac_k_vo,
                              quant.dev_opac_k_cia_h2h2,
                              quant.dev_opac_k_cia_h2he,
                              quant.dev_opac_h2o_wg_lay,
                              quant.dev_opac_co2_wg_lay,
                              quant.dev_opac_co_wg_lay,
                              quant.dev_opac_ch4_wg_lay,
                              quant.dev_opac_nh3_wg_lay,
                              quant.dev_opac_hcn_wg_lay,
                              quant.dev_opac_c2h2_wg_lay,
                              quant.dev_opac_tio_wg_lay,
                              quant.dev_opac_vo_wg_lay,
                              quant.dev_opac_cia_h2h2_wg_lay,
                              quant.dev_opac_cia_h2he_wg_lay,
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
                                  quant.dev_opac_k_h2o,
                                  quant.dev_opac_k_co2,
                                  quant.dev_opac_k_co,
                                  quant.dev_opac_k_ch4,
                                  quant.dev_opac_k_nh3,
                                  quant.dev_opac_k_hcn,
                                  quant.dev_opac_k_c2h2,
                                  quant.dev_opac_k_tio,
                                  quant.dev_opac_k_vo,
                                  quant.dev_opac_k_cia_h2h2,
                                  quant.dev_opac_k_cia_h2he,
                                  quant.dev_opac_wg_int,
                                  quant.dev_opac_h2o_wg_int,
                                  quant.dev_opac_co2_wg_int,
                                  quant.dev_opac_co_wg_int,
                                  quant.dev_opac_ch4_wg_int,
                                  quant.dev_opac_nh3_wg_int,
                                  quant.dev_opac_hcn_wg_int,
                                  quant.dev_opac_c2h2_wg_int,
                                  quant.dev_opac_tio_wg_int,
                                  quant.dev_opac_vo_wg_int,
                                  quant.dev_opac_cia_h2h2_wg_int,
                                  quant.dev_opac_cia_h2he_wg_int,
                                  quant.npress,
                                  quant.ntemp,
                                  quant.ny,
                                  quant.nbin,
                                  quant.ninterface,
                                  block=(16, 16, 1),
                                  grid=((int(quant.nbin) + 15) // 16, (int(quant.ninterface) + 15) // 16, 1)
                                  )

                cuda.Context.synchronize()

    def combine_to_mixed_opacities(self, quant):
        """ combine the individual molecular opacities to layer/interface opacities """

        if self.V_iter_nr > 0:

            comb_opacities = self.mod.get_function("comb_opac")

            comb_opacities(quant.dev_f_h2o_lay,
                           quant.dev_f_co2_lay,
                           quant.dev_f_co_lay,
                           quant.dev_f_ch4_lay,
                           quant.dev_f_nh3_lay,
                           quant.dev_f_hcn_lay,
                           quant.dev_f_c2h2_lay,
                           quant.dev_f_tio_lay,
                           quant.dev_f_vo_lay,
                           quant.dev_f_h2_lay,
                           quant.dev_f_he_lay,
                           quant.dev_opac_h2o_wg_lay,
                           quant.dev_opac_co2_wg_lay,
                           quant.dev_opac_co_wg_lay,
                           quant.dev_opac_ch4_wg_lay,
                           quant.dev_opac_nh3_wg_lay,
                           quant.dev_opac_hcn_wg_lay,
                           quant.dev_opac_c2h2_wg_lay,
                           quant.dev_opac_tio_wg_lay,
                           quant.dev_opac_vo_wg_lay,
                           quant.dev_opac_cia_h2h2_wg_lay,
                           quant.dev_opac_cia_h2he_wg_lay,
                           quant.dev_opac_wg_lay,
                           quant.dev_meanmolmass_lay,
                           quant.ny,
                           quant.nbin,
                           quant.nlayer,
                           block=(16, 16, 1),
                           grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                           )

            cuda.Context.synchronize()

            if quant.iso == 0:

                comb_opacities = self.mod.get_function("comb_opac")

                comb_opacities(quant.dev_f_h2o_int,
                               quant.dev_f_co2_int,
                               quant.dev_f_co_int,
                               quant.dev_f_ch4_int,
                               quant.dev_f_nh3_int,
                               quant.dev_f_hcn_int,
                               quant.dev_f_c2h2_int,
                               quant.dev_f_tio_int,
                               quant.dev_f_vo_int,
                               quant.dev_f_h2_int,
                               quant.dev_f_he_int,
                               quant.dev_opac_h2o_wg_int,
                               quant.dev_opac_co2_wg_int,
                               quant.dev_opac_co_wg_int,
                               quant.dev_opac_ch4_wg_int,
                               quant.dev_opac_nh3_wg_int,
                               quant.dev_opac_hcn_wg_int,
                               quant.dev_opac_c2h2_wg_int,
                               quant.dev_opac_tio_wg_int,
                               quant.dev_opac_vo_wg_int,
                               quant.dev_opac_cia_h2h2_wg_int,
                               quant.dev_opac_cia_h2he_wg_int,
                               quant.dev_opac_wg_int,
                               quant.dev_meanmolmass_int,
                               quant.ny,
                               quant.nbin,
                               quant.ninterface,
                               block=(16, 16, 1),
                               grid=((int(quant.nbin) + 15) // 16, (int(quant.ninterface) + 15) // 16, 1)
                               )

                cuda.Context.synchronize()

    def combine_to_scat_cross(self, quant):
        """ create layer scattering cross-sections """

        if self.V_iter_nr > 0:

            comb_scat_cross = self.mod.get_function("comb_scat_cross")

            comb_scat_cross(quant.dev_f_h2_lay,
                            quant.dev_opac_scat_cross,
                            quant.dev_scat_cross_lay,
                            quant.nbin,
                            quant.nlayer,
                            block=(16, 16, 1),
                            grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                            )

            if quant.iso == 0:

                comb_scat_cross = self.mod.get_function("comb_scat_cross")

                comb_scat_cross(quant.dev_f_h2_int,
                                quant.dev_opac_scat_cross,
                                quant.dev_scat_cross_int,
                                quant.nbin,
                                quant.ninterface,
                                block=(16, 16, 1),
                                grid=((int(quant.nbin) + 15) // 16, (int(quant.ninterface) + 15) // 16, 1)
                                )

                cuda.Context.synchronize()

    def read_or_create_iter_count(self):
        """ writes the TP-profile to a file """

        try:
            with open("../iter_count.txt", "r") as file:
                for line in file:
                    column = line.split()
                    if column:
                        self.V_iter_nr = float(column[0])
        except IOError:
            with open("../iter_count.txt", "w") as file:
                file.writelines("0")
                self.V_iter_nr = 0

    def write_tp_VULCAN(self, quant):
        """ writes the TP-profile to a file """
        try:
            with open("../tp_VULCAN_"+str(int(self.V_iter_nr))+".dat", "w") as file:
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

        if self.V_iter_nr > 0 and quant.singlewalk == 0:
            # read in temperatures from the last two iterations
            previous_temp = []
            current_temp = []

            with open("../tp_VULCAN_" + str(int(self.V_iter_nr-1)) + ".dat", "r") as previous_file:
                next(previous_file)
                for line in previous_file:
                    column = line.split()
                    previous_temp.append(quant.fl_prec(column[1]))

            with open("../tp_VULCAN_" + str(int(self.V_iter_nr)) + ".dat", "r") as current_file:
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

                convergence = 1

            else:

                convergence = 0

            # write out result
            with open("../stop.dat", "w") as file:
                file.writelines(str(convergence))


if __name__ == "__main__":
    print("This module provides the VULCAN coupling parameters and calculations. Please behave - VULCAN is usually ill-tempered.")
