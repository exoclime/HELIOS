# =============================================================================
# Module for combining the individual opacity sources
# Copyright (C) 2018 - 2022 Matej Malik
# =============================================================================
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
# =============================================================================

import os
import h5py
import numpy as npy
import time
from numba import jit
from source import tools as tls
from source import phys_const as pc
from source import species_database as sd


class Species(object):

    def __init__(self):

        self.name = None
        self.absorbing = None
        self.scattering = None
        self.mixing_ratio = None
        self.fc_name = None
        self.weight = None


class Comb(object):
    """ class responsible for combining and mixing the individual k-tables """
    
    def __init__(self):
        self.chem_press = None
        self.chem_temp = None
        self.chem_np = None
        self.chem_nt = None
        self.mu = None
        self.n_e = None
        self.k_y = None
        self.k_w = None
        self.k_i = None
        self.k_x = None
        self.molname_list = []
        self.fastchem_data = None
        self.fastchem_data_low = None
        self.fastchem_data_high = None
        self.nx = None
        self.ny = None
        self.final_press = None
        self.final_temp = None
        self.final_np = None
        self.final_nt = None
        self.combined_opacities = None
        self.combined_cross_sections = None

    # -------- generic methods -------- #

    @staticmethod
    def delete_duplicates(long_list):
        """ delete all duplicates in a list and return new list """
        short_list = []
        for item in long_list:
            if item not in short_list:
                short_list.append(item)
        return short_list

    @staticmethod
    def is_number(s):

        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def interpolate_vmr_to_final_grid(self, vmr):

        temp_old = self.chem_temp
        press_old = self.chem_press
        temp_new = self.final_temp
        press_new = self.final_press

        vmr_old = vmr

        vmr_new = npy.zeros(self.final_np * self.final_nt)

        old_nt = len(temp_old)
        old_np = len(press_old)

        print("\nInterpolating VMR...")
        print("chemistry temperature grid:\n", temp_old[:3], "...", temp_old[-3:])
        print("final table temperature grid:\n", temp_new[:3], "...", temp_new[-3:])
        print("chemistry pressure grid:\n", press_old[:3], "...", press_old[-3:])
        print("final table pressure grid:\n", press_new[:3], "...", press_new[-3:])

        print("final np:", self.final_np, "final nt:", self.final_nt)
        print("old np:", old_np, "old nt:", old_nt)

        for i in range(self.final_nt):

            for j in range(self.final_np):

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

                    vmr_new[j + self.final_np * i] = vmr_old[p_left + old_np * t_left]

                elif reduced_p != 1 and reduced_t == 1:

                    p_right = p_left + 1

                    vmr_new[j + self.final_np * i] = \
                            (vmr_old[p_right + old_np * t_left] * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                             + vmr_old[p_left + old_np * t_left] * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                             ) / (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left]))

                elif reduced_p == 1 and reduced_t != 1:

                    t_right = t_left + 1

                    vmr_new[j + self.final_np * i] = \
                        (vmr_old[p_left + old_np * t_right] * (temp_new[i] - temp_old[t_left]) \
                         + vmr_old[p_left + old_np * t_left] * (temp_old[t_right] - temp_new[i]) \
                         ) / (temp_old[t_right] - temp_old[t_left])

                elif reduced_p != 1 and reduced_t != 1:

                    p_right = p_left + 1
                    t_right = t_left + 1

                    vmr_new[j + self.final_np * i] = \
                        (
                            vmr_old[p_right + old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                            + vmr_old[p_left + old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                            + vmr_old[p_right + old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                            + vmr_old[p_left + old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                            ) / ((temp_old[t_right] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left])))

                if npy.isnan(vmr_new[j + self.final_np * i]):
                    print("NaN-Error at entry with indices:", "pressure:", j, "temperature:", i)
                    raise SystemExit()

        return vmr_new

    @staticmethod
    @jit(nopython=True)
    def interpolate_opacity_to_final_grid(press_old, temp_old, k_old, temp_new, press_new, final_nt, final_np, nx, ny):

        k_new = npy.zeros(ny * nx * final_np * final_nt)

        old_np = len(press_old)

        for i in range(final_nt):

            reduced_t = 0

            if temp_old[0] < temp_new[i]:

                for x in range(len(temp_old)):
                    if temp_old[x] <= temp_new[i]:
                        t_left = x
                    else:
                        break
            else:
                t_left = 0
                reduced_t = 1

            for j in range(final_np):

                reduced_p = 0

                if press_old[0] < press_new[j]:

                    for x in range(len(press_old)):
                        if press_old[x] <= press_new[j]:
                            p_left = x
                        else:
                            break
                else:
                    p_left = 0
                    reduced_p = 1

                if t_left == len(temp_old)-1:
                    reduced_t = 1

                if p_left == len(press_old)-1:
                    reduced_p = 1

                if reduced_p == 1 and reduced_t == 1:

                    for y in range(ny):
                        for x in range(nx):

                            k_new[y + ny * x + ny * nx * j + ny * nx * final_np * i] = \
                                k_old[y + ny*x + ny*nx*p_left + ny*nx*old_np*t_left]

                elif reduced_p != 1 and reduced_t == 1:

                    p_right = p_left + 1

                    for x in range(nx):
                        for y in range(ny):

                            k_new[y + ny*x + ny*nx*j + ny*nx*final_np*i] = \
                                (k_old[y + ny*x + ny*nx*p_right + ny*nx*old_np*t_left] * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                + k_old[y + ny*x + ny*nx*p_left + ny*nx*old_np*t_left] * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                                 ) / (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left]))

                elif reduced_p == 1 and reduced_t != 1:

                    t_right = t_left + 1

                    for x in range(nx):
                        for y in range(ny):

                            k_new[y + ny * x + ny * nx * j + ny * nx * final_np * i] = \
                                (k_old[y + ny * x + ny * nx * p_left + ny * nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) \
                                 + k_old[y + ny * x + ny * nx * p_left + ny * nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) \
                                 ) / (temp_old[t_right] - temp_old[t_left])

                elif reduced_p != 1 and reduced_t != 1:

                    p_right = p_left + 1
                    t_right = t_left + 1

                    for x in range(nx):
                        for y in range(ny):

                            k_new[y + ny * x + ny * nx * j + ny * nx * final_np * i] = \
                                (
                                 k_old[y + ny * x + ny * nx * p_right + ny * nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                 + k_old[y + ny * x + ny * nx * p_left + ny * nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j]))\
                                 + k_old[y + ny * x + ny * nx * p_right + ny * nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                 + k_old[y + ny * x + ny * nx * p_left + ny * nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                                 ) / ((temp_old[t_right] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left])))

        return k_new

    def write_h5(self, param, name, ktable):

        # create directory if necessary
        try:
            os.makedirs(param.individual_calc_path)
        except OSError:
            if not os.path.isdir(param.individual_calc_path):
                raise

        if param.format == "k-distribution":
            with h5py.File(param.individual_calc_path + name, "w") as mixed_file:
                mixed_file.create_dataset("pressures", data=self.final_press)
                mixed_file.create_dataset("temperatures", data=self.final_temp)
                mixed_file.create_dataset("interface wavelengths", data=self.k_i)
                mixed_file.create_dataset("center wavelengths", data=self.k_x)
                mixed_file.create_dataset("wavelength width of bins", data=self.k_w)
                mixed_file.create_dataset("ypoints", data=self.k_y)
                mixed_file.create_dataset("kpoints", data=ktable)

        elif param.format == "sampling":
            with h5py.File(param.individual_calc_path + name, "w") as mixed_file:
                mixed_file.create_dataset("pressures", data=self.final_press)
                mixed_file.create_dataset("temperatures", data=self.final_temp)
                mixed_file.create_dataset("wavelengths", data=self.k_x)
                mixed_file.create_dataset("opacities", data=ktable)

    def read_individual_ktable(self, param, name):
        """ read in individual ktables """

        with h5py.File(param.individual_calc_path + name + "_opac_kdistr.h5", "r") as ind_file:

            # is or should be universal
            self.k_y = [y for y in ind_file["ypoints"][:]]
            self.k_x = [x for x in ind_file["center wavelengths"][:]]
            try:
                self.k_w = [w for w in ind_file["wavelength width of bins"][:]]
                self.k_i = [i for i in ind_file["interface wavelengths"][:]]
            except KeyError:
                pass

            # species specific
            k_temp = [t for t in ind_file["temperatures"][:]]
            k_press = [p for p in ind_file["pressures"][:]]
            kpoints = [k for k in ind_file["kpoints"][:]]

            self.nx = len(self.k_x)
            self.ny = len(self.k_y)

        print("\nIncluding " + name + "_opac_kdistr.h5 !")
        self.molname_list.append(name.encode('utf8'))

        return k_press, k_temp, kpoints

    def read_individual_opacity_for_sampling(self, param, name):
        """ read in individual sampled opacities """

        try:
            with h5py.File(param.individual_calc_path + name + "_opac_sampling.h5", "r") as ind_file:

                # is or should be universal
                self.k_x = [x for x in ind_file["wavelengths"][:]]
                self.nx = len(self.k_x)
                self.ny = 1

                # species specific
                kpoints = [k for k in ind_file["opacities"][:]]
                k_temp = [t for t in ind_file["temperatures"][:]]
                k_press = [p for p in ind_file["pressures"][:]]

            print("\nIncluding " + name + "_opac_sampling.h5 !")
            self.molname_list.append(name.encode('utf8'))
        except(OSError):
            print("\nABORT - " + name + "_opac_sampling.h5 not found!")
            raise SystemExit()

        return k_press, k_temp, kpoints

    def load_fastchem_data(self, param):
        """ read in the fastchem mixing ratios"""

        try:

            self.fastchem_data = npy.genfromtxt(param.fastchem_path + 'chem.dat',
                                                    names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")
        except OSError:

            self.fastchem_data_low = npy.genfromtxt(param.fastchem_path + 'chem_low.dat',
                                                    names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

            self.fastchem_data_high = npy.genfromtxt(param.fastchem_path + 'chem_high.dat',
                                                     names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

        # temperature and pressure from the chemical grid
        if self.fastchem_data is not None:
            read_press = self.fastchem_data['Pbar']
            read_temp = self.fastchem_data['Tk']
            chem_mu = self.fastchem_data['mu']
        else:
            read_press = npy.concatenate((self.fastchem_data_low['Pbar'], self.fastchem_data_high['Pbar']))
            read_temp = npy.concatenate((self.fastchem_data_low['Tk'], self.fastchem_data_high['Tk']))
            chem_mu = npy.concatenate((self.fastchem_data_low['mu'], self.fastchem_data_high['mu']))

        read_press = self.delete_duplicates(read_press)
        self.chem_temp = self.delete_duplicates(read_temp)
        self.chem_press = [p * 1e6 for p in read_press]

        self.chem_nt = len(self.chem_temp)
        self.chem_np = len(self.chem_press)

        # mean molecular weight
        self.mu = self.interpolate_vmr_to_final_grid(chem_mu)

    @staticmethod
    def check_if_already_interpolated(param, name, ending):
        """ checks whether necessary to interpolate or file already exists """

        try:
            h5py.File(param.individual_calc_path + name + ending, "r")
            print("Interpolated file " + param.individual_calc_path + name + ending + " already exists. Skipping interpolation for this molecule...")
            return True

        except OSError:
            return False

    def interpolate_opacity(self, param, name, k_press, k_temp, kpoints):

        print("\nInterpolating...")

        if param.format == 'k-distribution':
            ending = "_opac_ip_kdistr.h5"
            database = "kpoints"
        elif param.format == 'sampling':
            ending = "_opac_ip_sampling.h5"
            database = "opacities"

        if self.check_if_already_interpolated(param, name, ending):

            interpolated_opacities = []
            with h5py.File(param.individual_calc_path + name + ending, "r") as ip_file:
                for k in ip_file[database][:]:
                    interpolated_opacities.append(k)

        else:

            start = time.time()

            # some user feedback on the P,T grid conversion
            print("opacity temperature grid:\n", k_temp[:3], "...", k_temp[-3:])
            print("final table temperature grid:\n", self.final_temp[:3], "...", self.final_temp[-3:])
            print("opacity pressure grid:\n", k_press[:3], "...", k_press[-3:])
            print("final table pressure grid:\n", self.final_press[:3], "...", self.final_press[-3:])
            print("ny:", self.ny, "nx:", self.nx, "np:", self.final_np, "nt:", self.final_nt)
            print("old np:", len(k_press))
            print("old nt:", len(k_temp))

            interpolated_opacities = self.interpolate_opacity_to_final_grid(k_press, k_temp, kpoints, self.final_temp, self.final_press, self.final_nt, self.final_np, self.nx, self.ny)

            end = time.time()
            print("Time for interpolating = {} sec.".format(end - start))

            self.write_h5(param, name + ending, interpolated_opacities)

        return interpolated_opacities

    def create_mixed_file(self, param):
        """ write to hdf5 file """

        # create directory if necessary
        try:
            os.makedirs(param.final_path)
        except OSError:
            if not os.path.isdir(param.final_path):
                raise

        if param.format == 'k-distribution':
            filename = "mixed_opac_kdistr.h5"
        elif param.format == 'sampling':
            filename = "mixed_opac_sampling.h5"

        # change units to MKS if chosen in param file
        if param.units == "MKS":
            self.final_press = [p * 1e-1 for p in self.final_press]
            self.combined_opacities = [c * 1e-1 for c in self.combined_opacities]
            self.combined_cross_sections = [c * 1e-4 for c in self.combined_cross_sections]
            self.k_x = [k * 1e-2 for k in self.k_x]

            if param.format == "k-distribution":
                self.k_i = [k * 1e-2 for k in self.k_i]
                self.k_w = [k * 1e-2 for k in self.k_w]

        with h5py.File(param.final_path + filename, "w") as mixed_file:
            mixed_file.create_dataset("pressures", data=self.final_press)
            mixed_file.create_dataset("temperatures", data=self.final_temp)
            mixed_file.create_dataset("meanmolmass", data=self.mu)
            mixed_file.create_dataset("kpoints", data=self.combined_opacities)
            mixed_file.create_dataset("weighted Rayleigh cross-sections", data=self.combined_cross_sections)
            mixed_file.create_dataset("included molecules", data=self.molname_list)
            mixed_file.create_dataset("wavelengths", data=self.k_x)
            mixed_file.create_dataset("FastChem path", data=param.fastchem_path)
            mixed_file.create_dataset("units", data=param.units)

            if param.format == 'k-distribution':
                mixed_file.create_dataset("center wavelengths", data=self.k_x)
                mixed_file.create_dataset("interface wavelengths",data=self.k_i)
                mixed_file.create_dataset("wavelength width of bins",data=self.k_w)
                mixed_file.create_dataset("ypoints",data=self.k_y)

    def add_to_scattering_file(self, param, name, ray_cross_this_species):

        with h5py.File(param.individual_calc_path + "scat_cross_sections.h5", "a") as scatfile:

            # P, T information not required at the moment since cross-sections independent of P,T.
            # Keep this code chunk for later versions though.
                # if "pressures" not in scatfile:
                #     scatfile.create_dataset("pressures", data=self.final_press)
                # if "temperatures" not in scatfile:
                #     scatfile.create_dataset("temperatures", data=self.final_temp)

            if "wavelengths" not in scatfile:
                scatfile.create_dataset("wavelengths", data=self.k_x)

            scatfile.create_dataset("rayleigh_"+name, data=ray_cross_this_species)

    def include_rayleigh_cross_section(self, param, ray, species, vol_mix_ratio):
        """ tabulates the rayleigh cross sections for various species """

        print("\nCalculating scattering cross sections ...")

        # references for the scattering constants:
        # H2: Cox 2000
        # He: Sneep & Ubachs 2005, Thalman et al. 2014
        # H: Lee & Kim 2004
        # H2O: Murphy 1977, Schiebener et al. 1990, Wagner & Kretzschmar 2008
        # CO: Sneep & Ubachs 2005
        # CO2: Sneep & Ubachs 2005, Thalman et al. 2014
        # O2: Sneep & Ubachs 2005, Thalman et al. 2014
        # N2: Sneep & Ubachs 2005, Thalman et al. 2014
        # e-: Thomson scattering cross-section from "astropy.constants" package.

        list_of_implemented_scattering_species = ['H', 'H2', 'He', 'H2O', 'CO2', 'CO', 'O2', 'N2', 'e-']

        if species.name in list_of_implemented_scattering_species:

            if species.name != 'H2O':

                try:
                    with h5py.File(param.individual_calc_path + "scat_cross_sections.h5", "r") as scatfile:
                        ray_cross_this_species = [r for r in scatfile["rayleigh_" + species.name][:]]

                    print("Scattering cross sections successfully read from file.")

                except (FileNotFoundError, OSError, KeyError):

                    print("Scattering cross sections file or data set for this species not found. --> Calculating cross sections ... ")
                    ray_cross_this_species = []

                    # calculate the Rayleigh cross sections that do not depend on P or T

                    for x in range(self.nx):

                        cross_section = 0

                        if (species.name != 'H') and (species.name != 'e-'):

                            if species.name == "CO2":

                                index = ray.index_co2(self.k_x[x])
                                n_ref = ray.n_ref_co2
                                King = ray.King_co2(self.k_x[x])
                                lamda_limit = self.k_x[-1]

                            elif species.name == "H2":

                                index = ray.index_h2(self.k_x[x])
                                n_ref = ray.n_ref_h2
                                King = ray.King_h2
                                lamda_limit = self.k_x[-1]

                            elif species.name == "He":

                                index = ray.index_he(self.k_x[x])
                                n_ref = ray.n_ref_he
                                King = ray.King_he
                                lamda_limit = self.k_x[-1]

                            elif species.name == "N2":

                                index = ray.index_n2(self.k_x[x])
                                n_ref = ray.n_ref_n2
                                King = ray.King_n2(self.k_x[x])
                                lamda_limit = self.k_x[-1]

                            elif species.name == "O2":

                                index = ray.index_o2(self.k_x[x])
                                n_ref = ray.n_ref_o2
                                King = ray.King_o2(self.k_x[x])
                                lamda_limit = self.k_x[-1]

                            elif species.name == "CO":

                                index = ray.index_co(self.k_x[x])
                                n_ref = ray.n_ref_co
                                King = ray.King_co
                                lamda_limit = self.k_x[-1]

                            cross_section = ray.cross_sect(self.k_x[x], index, n_ref, King, lamda_limit)

                        elif species.name == "e-":

                            cross_section = pc.SIGMA_T

                        elif species.name == "H":

                            cross_section = ray.cross_sect_h(self.k_x[x])

                        ray_cross_this_species.append(cross_section)

                    self.add_to_scattering_file(param, species.name, ray_cross_this_species)

                # add to weighted total cross section over all species
                for t in range(self.final_nt):

                    for p in range(self.final_np):

                        for x in range(self.nx):

                            self.combined_cross_sections[x + self.nx * p + self.nx * self.final_np * t] += vol_mix_ratio[p + self.final_np * t] * ray_cross_this_species[x]

            # H2O cannot be pre-tabulated individually as it depends on the water mixing ratio
            elif species.name == "H2O":

                ray_cross_this_species = []

                King = ray.King_h2o
                lamda_limit = 2.5e-4

                for t in range(self.final_nt):

                    for p in range(self.final_np):

                        n_ref = ray.n_ref_h2o(self.final_press[p], self.final_temp[t], vol_mix_ratio[p + self.final_np * t])

                        for x in range(self.nx):

                            index = ray.index_h2o(self.k_x[x], self.final_press[p], self.final_temp[t], vol_mix_ratio[p + self.final_np * t])

                            cross_section = ray.cross_sect(self.k_x[x], index, n_ref, King, lamda_limit)

                            ray_cross_this_species.append(cross_section)

                # add to weighted total cross section over all species -- for water everything is dependent on P and T
                for t in range(self.final_nt):

                    for p in range(self.final_np):

                        for x in range(self.nx):

                            self.combined_cross_sections[x + self.nx * p + self.nx * self.final_np * t] += vol_mix_ratio[p + self.final_np * t] * ray_cross_this_species[x + self.nx * p + self.nx * self.final_np * t]

        else:
            print("WARNING WARNING WARNING: Rayleigh scattering cross sections for species", species.name, "not found. Please double-check! Continuing without those... ")

    @staticmethod
    @jit(nopython=True)
    def weight_and_include_opacity(vol_mix_ratio, vol_mix_ratio2, weight, mu, opac, final_nt, final_np, nx, ny, combined_opacity):
        """ weights opacity and adds to total, combined opacity array """

        for t in range(final_nt):

            for p in range(final_np):

                mass_mix_ratio = vol_mix_ratio[p + final_np * t] * vol_mix_ratio2[p + final_np * t] * weight / mu[p + final_np * t]

                for x in range(nx):

                    for y in range(ny):

                        weighted_opac = mass_mix_ratio * opac[y + ny*x + ny*nx*p + ny*nx*final_np*t]

                        combined_opacity[y + ny*x + ny*nx*p + ny*nx*final_np*t] += weighted_opac

    def calc_h_minus_bf(self, conti, param):
        """ calculates the H- bound-free continuum opacity """

        print("\nCalculating H- bound-free ...")

        opac_h_min_bf = []

        # pressure and temperature dependent bound-free opacity
        for t in range(self.final_nt):

            for p in range(self.final_np):

                tls.percent_counter(t, self.final_nt, p, self.final_np)

                for x in range(self.nx):

                    opac_bf = conti.h_min_bf_cross_sect(self.k_x[x]) / (sd.species_lib["H"].weight * pc.AMU)

                    for y in range(self.ny):

                        opac_h_min_bf.append(opac_bf)

        # write to h5 file
        if param.format == 'k-distribution':
            ending = "_opac_ip_kdistr.h5"

        elif param.format == 'sampling':
            ending = "_opac_ip_sampling.h5"

        self.write_h5(param, "H-_bf" + ending, opac_h_min_bf)

        # add to mol list
        self.molname_list.append("H-_bf".encode('utf8'))

        print("\nH- bound-free calculation complete!")

        return opac_h_min_bf

    def calc_h_minus_ff(self, conti, param):
        """ calculates the H- free-free continuum opacity """

        print("\nCalculating H- free-free ...")

        opac_h_min_ff = []

        # pressure and temperature dependent free-free opacity
        for t in range(self.final_nt):

            for p in range(self.final_np):

                tls.percent_counter(t, self.final_nt, p, self.final_np)

                for x in range(self.nx):

                    opac_ff = conti.h_min_ff_cross_sect(self.k_x[x], self.final_temp[t], self.final_press[p]) / (sd.species_lib["H"].weight * pc.AMU)

                    for y in range(self.ny):

                        opac_h_min_ff.append(opac_ff)

        # write to h5 file
        if param.format == 'k-distribution':
            ending = "_opac_ip_kdistr.h5"

        elif param.format == 'sampling':
            ending = "_opac_ip_sampling.h5"

        self.write_h5(param, "H-_ff" + ending, opac_h_min_ff)

        # add to mol list
        self.molname_list.append("H-_ff".encode('utf8'))

        print("\nH- free-free calculation complete!")

        return opac_h_min_ff

    def calc_he_minus(self, conti, param):

        print("\nIncluding He- ...")

        opac_he_min_ff = []

        opac_func = conti.include_he_min_opacity()

        for t in range(self.final_nt):

            for p in range(self.final_np):

                for x in range(self.nx):

                    lamda = npy.log10(self.k_x[x] * 1e4) # because interpol. function requires log10 of wavelengths in micron

                    opac = 10**opac_func(self.final_temp[t], lamda)[0] * self.final_press[p] / (sd.species_lib["He"].weight * pc.AMU)

                    for y in range(self.ny):

                        opac_he_min_ff.append(opac)

        # write to h5 file
        if param.format == 'k-distribution':
            ending = "_opac_ip_kdistr.h5"

        elif param.format == 'sampling':
            ending = "_opac_ip_sampling.h5"

        self.write_h5(param, "He-" + ending, opac_he_min_ff)

        # add to mol list
        self.molname_list.append("He-".encode('utf8'))

        print("\n... is complete!")

        return opac_he_min_ff

    @staticmethod
    def read_species_data(path):

        species_list = []

        with open(path) as sfile:

            next(sfile)
            next(sfile)

            for line in sfile:

                column = line.split()

                if len(column) > 0:

                    species = Species()

                    species.name = column[0]
                    species.absorbing = column[1]
                    species.scattering = column[2]
                    species.mixing_ratio = column[3]

                    species_list.append(species)

        # we need an absorbing species as first entry so let's shuffle
        for s in range(len(species_list)):

            if species_list[s].absorbing == 'yes':

                species_list.insert(0, species_list[s])
                species_list.pop(s+1)

                break

            if s == len(species_list)-1:

                raise IOError("Whoops! At least one species needs to be absorbing. Please check your 'final species' file.")

        # obtain additional info from the species data base
        for s in range(len(species_list)):

            for key in sd.species_lib:

                if species_list[s].name == sd.species_lib[key].name:
                    species_list[s].weight = sd.species_lib[key].weight
                    species_list[s].fc_name = sd.species_lib[key].fc_name

        # check that each species was found in the data base
        for s in range(len(species_list)):

            if species_list[s].weight is None:
                raise IOError("Oops! Species '" + species_list[s].name +
                              "' was not found in the species data base. "
                              "Please check that the name is spelled correctly. "
                              "If so, add the relevant information to the file 'species_database.py' and try again. Aborting ..."
                              )

            if (species_list[s].fc_name is None) and (species_list[s].source_for_vmr == "FastChem"):
                raise IOError("Oops! FastChem name for species " + species_list[s].name +
                              "unknown."
                              "Please check that the species name is spelled correctly."
                              "If so, add the relevant information to the file 'species_database.py' and try again. Aborting ..."
                              )

        return species_list

    def use_hard_coded_PT_grid(self, species_list):

        self.final_temp = npy.arange(50, 6050, 50)

        press_list_p1 = [10 ** p for p in npy.arange(0, 10, 1)]
        press_list_p2 = [10 ** p for p in npy.arange(0.33333333, 9.33333333, 1)]
        press_list_p3 = [10 ** p for p in npy.arange(0.66666666, 9.66666666, 1)]
        press_list = npy.append(press_list_p1, npy.append(press_list_p2, press_list_p3))
        press_list.sort()
        self.final_press = press_list

        self.final_nt = len(self.final_temp)
        self.final_np = len(self.final_press)

        mu = 0
        mixing_ratio_tot = 0

        for n in range(len(species_list)):

            if self.is_number(species_list[n].mixing_ratio):  # checks if all characters are numeric. this excludes FastChem entries, CIA species, and H-

                mu += float(species_list[n].mixing_ratio) * species_list[n].weight
                mixing_ratio_tot += float(species_list[n].mixing_ratio)

        if mixing_ratio_tot > 0:
            self.mu = npy.ones(self.final_np * self.final_nt) * mu / mixing_ratio_tot  # normalizing with respect to the total VMR of all included species
            # Note that this mu calculation is overwritten if FastChem is used for the mixing ratios

    def add_one_species(self, param, ray, conti, species, iter):

        print("\n\n----------\nGenerating mixed opacity table. Including --> " + species.name + " <--")

        # needs to be assigned beforehand to make python happy
        interpol_opac = None

        # read in ktable for absorbing species
        if species.absorbing == 'yes':

            if species.name not in ["H-_bf", "H-_ff", "He-"]:

                if param.format == 'k-distribution':
                    k_press, k_temp, kpoints = self.read_individual_ktable(param, species.name)
                elif param.format == 'sampling':
                    k_press, k_temp, kpoints = self.read_individual_opacity_for_sampling(param, species.name)

                # interpolate
                interpol_opac = self.interpolate_opacity(param, species.name, k_press, k_temp, kpoints)

            elif species.name == "H-_bf":

                interpol_opac = self.calc_h_minus_bf(conti, param)

            elif species.name == "H-_ff":

                interpol_opac = self.calc_h_minus_ff(conti, param)

            elif species.name == "He-":

                interpol_opac = self.calc_he_minus(conti, param)

        # generate final arrays during first species iteration (cannot do that earlier because need to know ny and nx first)
        if iter == 0:
            self.combined_opacities = npy.zeros(self.ny * self.nx * self.final_np * self.final_nt)
            self.combined_cross_sections = npy.zeros(self.nx * self.final_np * self.final_nt)

        # get abundances
        # case 1: one mixing ratio
        if ("CIA" not in species.name) and species.name not in ["H-_ff", "He-"]:

            if species.mixing_ratio == "FastChem":

                if self.fastchem_data is not None:
                    chem_vmr = self.fastchem_data[species.fc_name]
                else:
                    chem_vmr = npy.concatenate((self.fastchem_data_low[species.fc_name], self.fastchem_data_high[species.fc_name]))

                final_vmr = self.interpolate_vmr_to_final_grid(chem_vmr)

            else:
                final_vmr = npy.ones(self.final_np * self.final_nt) * float(species.mixing_ratio)

            final_vmr_2 = npy.ones(self.final_np * self.final_nt)

        # case 2: two mixing ratios
        elif ("CIA" in species.name) or species.name in ["H-_ff", "He-"]:  # CIA, H-_ff and He- need two mixing ratios

            if species.mixing_ratio == "FastChem":

                two_fc_names = species.fc_name.split('&')

                if self.fastchem_data is not None:
                    chem_vmr = self.fastchem_data[two_fc_names[0]]
                    chem_vmr_2 = self.fastchem_data[two_fc_names[1]]
                else:
                    chem_vmr = npy.concatenate((self.fastchem_data_low[two_fc_names[0]], self.fastchem_data_high[two_fc_names[0]]))
                    chem_vmr_2 = npy.concatenate((self.fastchem_data_low[two_fc_names[1]], self.fastchem_data_high[two_fc_names[1]]))

                final_vmr = self.interpolate_vmr_to_final_grid(chem_vmr)
                final_vmr_2 = self.interpolate_vmr_to_final_grid(chem_vmr_2)

            else:
                two_vmrs = species.mixing_ratio.split('&')

                final_vmr = npy.ones(self.final_np * self.final_nt) * float(two_vmrs[0])
                final_vmr_2 = npy.ones(self.final_np * self.final_nt) * float(two_vmrs[1])

        if species.absorbing == 'yes':

            start = time.time()

            print("\nWeighting by mass mixing ratio...")

            # weight opacity by mixing ratio
            self.weight_and_include_opacity(final_vmr, final_vmr_2, species.weight, self.mu, interpol_opac, self.final_nt, self.final_np, self.nx, self.ny, self.combined_opacities)

            end = time.time()
            print("Time for weighting = {} sec.".format(end - start))

        # calculate Rayleigh scattering component
        if species.scattering == 'yes':

            self.include_rayleigh_cross_section(param, ray, species, final_vmr)

    def combine_all_species(self, param, ray, conti):

        # read in data from species file
        species_list = self.read_species_data(param.final_species_file_path)

        # generate P, T grid for final table
        # yes, this is hard-coded -- so what? it works the best so far!
        self.use_hard_coded_PT_grid(species_list)

        # get chemical abundances from the FastChem output if existing
        for n in range(len(species_list)):
            if species_list[n].mixing_ratio == 'FastChem':

                self.load_fastchem_data(param)
                break

        # add one species after another
        for n in range(len(species_list)):

            self.add_one_species(param, ray, conti, species_list[n], n)

        self.create_mixed_file(param)

    @staticmethod
    def success():
        """ prints success message """
        print("\n--- Production of mixed opacity table successful! ---")


if __name__ == "__main__":
    print("This module is for the combination of the individual molecular opacities. Yes, the class is a comb. It combs through the opacities.")
