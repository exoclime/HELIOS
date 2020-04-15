# =============================================================================
# Module for combining the individual opacity sources
# Copyright (C) 2018 Matej Malik
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

import sys
import os
import h5py
import numpy as npy
import scipy as sp
from source import tools as tls
from source import phys_const as pc


class Species(object):

    def __init__(self):

        self.name = None
        self.absorbing = None
        self.scattering = None
        self.mixing_ratio = None
        self.fc_name = None
        self.mass = None


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


    def interpolate_opacity_to_final_grid(self, k_press, k_temp, kpoints):

        temp_old = k_temp
        press_old = k_press
        temp_new = self.final_temp
        press_new = self.final_press

        k_old = kpoints

        print("opacity temperature grid:\n", temp_old[:3], "...", temp_old[-3:])
        print("final table temperature grid:\n", temp_new[:3], "...", temp_new[-3:])
        print("opacity pressure grid:\n", press_old[:3], "...", press_old[-3:])
        print("final table pressure grid:\n", press_new[:3], "...", press_new[-3:])

        print("ny:",self.ny, "nx:",self.nx, "np:",self.final_np, "nt:",self.final_nt)

        k_new = npy.zeros(self.ny * self.nx * self.final_np * self.final_nt)

        old_nt = len(temp_old)
        old_np = len(press_old)

        print("old np:", old_np)
        print("old nt:", old_nt)

        for i in range(self.final_nt):

            for j in range(self.final_np):

                tls.percent_counter(i, self.final_nt, j, self.final_np)

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

                if t_left == len(temp_old)-1:
                    reduced_t = 1

                if p_left == len(press_old)-1:
                    reduced_p = 1

                if reduced_p == 1 and reduced_t == 1:

                    for y in range(self.ny):
                        for x in range(self.nx):

                            k_new[y + self.ny * x + self.ny * self.nx * j + self.ny * self.nx * self.final_np * i] = \
                                k_old[y + self.ny*x + self.ny*self.nx*p_left + self.ny*self.nx*old_np*t_left]

                elif reduced_p != 1 and reduced_t == 1:

                    p_right = p_left + 1

                    for y in range(self.ny):
                        for x in range(self.nx):

                            k_new[y + self.ny*x + self.ny*self.nx*j + self.ny*self.nx*self.final_np*i] = \
                                (k_old[y + self.ny*x + self.ny*self.nx*p_right + self.ny*self.nx*old_np*t_left] * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                + k_old[y + self.ny*x + self.ny*self.nx*p_left + self.ny*self.nx*old_np*t_left] * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                                 ) / (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left]))

                elif reduced_p == 1 and reduced_t != 1:

                    t_right = t_left + 1

                    for y in range(self.ny):
                        for x in range(self.nx):

                            k_new[y + self.ny * x + self.ny * self.nx * j + self.ny * self.nx * self.final_np * i] = \
                                (k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) \
                                 + k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) \
                                 ) / (temp_old[t_right] - temp_old[t_left])

                elif reduced_p != 1 and reduced_t != 1:

                    p_right = p_left + 1
                    t_right = t_left + 1

                    for y in range(self.ny):
                        for x in range(self.nx):

                            k_new[y + self.ny * x + self.ny * self.nx * j + self.ny * self.nx * self.final_np * i] = \
                                (
                                 k_old[y + self.ny * x + self.ny * self.nx * p_right + self.ny * self.nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                 + k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j]))\
                                 + k_old[y + self.ny * x + self.ny * self.nx * p_right + self.ny * self.nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                 + k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                                 ) / ((temp_old[t_right] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left])))

                if npy.isnan(k_new[y + self.ny * x + self.ny * self.nx * j + self.ny * self.nx * self.final_np * i]):
                    print("NaN-Error at entry with indices:", y, x, j, i)
                    raise SystemExit()

        return k_new

    def write_h5(self, param, name, ktable):

        # create directory if necessary
        try:
            os.makedirs(param.resampling_path)
        except OSError:
            if not os.path.isdir(param.resampling_path):
                raise

        if param.format == "ktable":
            with h5py.File(param.resampling_path + name, "w") as mixed_file:
                mixed_file.create_dataset("pressures", data=self.final_press)
                mixed_file.create_dataset("temperatures", data=self.final_temp)
                mixed_file.create_dataset("interface wavelengths", data=self.k_i)
                mixed_file.create_dataset("center wavelengths", data=self.k_x)
                mixed_file.create_dataset("wavelength width of bins", data=self.k_w)
                mixed_file.create_dataset("ypoints", data=self.k_y)
                mixed_file.create_dataset("kpoints", data=ktable)
        elif param.format == "sampling":
            with h5py.File(param.resampling_path + name, "w") as mixed_file:
                mixed_file.create_dataset("pressures", data=self.final_press)
                mixed_file.create_dataset("temperatures", data=self.final_temp)
                mixed_file.create_dataset("wavelengths", data=self.k_x)
                mixed_file.create_dataset("opacities", data=ktable)

    def read_individual_ktable(self, param, name):
        """ read in individual ktables """

        with h5py.File(param.resampling_path + name + "_opacities.h5", "r") as ind_file:

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

        print("\nIncluding " + name + "_opacities.h5 !")
        self.molname_list.append(name.encode('utf8'))

        return k_press, k_temp, kpoints

    def read_individual_opacity_for_sampling(self, param, name):
        """ read in individual sampled opacities """

        try:
            with h5py.File(param.resampling_path + name + "_opac_sampling.h5", "r") as ind_file:

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

        self.fastchem_data_low = npy.genfromtxt(param.fastchem_path + 'chem_low.dat',
                                  names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

        self.fastchem_data_high = npy.genfromtxt(param.fastchem_path + 'chem_high.dat',
                                   names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

        # temperature and pressure from the chemical grid
        read_press = npy.concatenate((self.fastchem_data_low['Pbar'], self.fastchem_data_high['Pbar']))
        read_temp = npy.concatenate((self.fastchem_data_low['Tk'], self.fastchem_data_high['Tk']))

        read_press = self.delete_duplicates(read_press)
        self.chem_temp = self.delete_duplicates(read_temp)
        self.chem_press = [p * 1e6 for p in read_press]

        self.chem_nt = len(self.chem_temp)
        self.chem_np = len(self.chem_press)

        # mean molecular weight
        chem_mu = npy.concatenate((self.fastchem_data_low['mu'], self.fastchem_data_high['mu']))
        self.mu = self.interpolate_vmr_to_final_grid(chem_mu)

        # electrons
        try:
            chem_n_e = npy.concatenate((self.fastchem_data_low['e-'], self.fastchem_data_high['e-']))
        except ValueError:
            chem_n_e = npy.concatenate((self.fastchem_data_low['e_minus'], self.fastchem_data_high['e_minus']))
        self.n_e = self.interpolate_vmr_to_final_grid(chem_n_e)

    @staticmethod
    def check_if_already_interpolated(param, name, ending):
        """ checks whether necessary to interpolate or file already exists """

        try:
            h5py.File(param.resampling_path + name + ending, "r")
            print("Interpolated file " + param.resampling_path + name + ending + " already exists. Skipping interpolation for this molecule...")
            return True

        except OSError:
            return False

    def interpolate_opacity(self, param, name, k_press, k_temp, kpoints):

        print("\nInterpolating...")

        if param.format == 'ktable':
            ending = "_opac_ip.h5"
            database = "kpoints"
        elif param.format == 'sampling':
            ending = "_opac_ip_sampling.h5"
            database = "opacities"

        if self.check_if_already_interpolated(param, name, ending):

            interpolated_opacities = []
            with h5py.File(param.resampling_path + name + ending, "r") as ip_file:
                for k in ip_file[database][:]:
                    interpolated_opacities.append(k)

        else:
            interpolated_opacities = self.interpolate_opacity_to_final_grid(k_press, k_temp, kpoints)
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

        if param.format == 'ktable':
            filename = "mixed_opac_ktable.h5"
        elif param.format == 'sampling':
            filename = "mixed_opac_sampling.h5"

        with h5py.File(param.final_path + filename, "w") as mixed_file:
            mixed_file.create_dataset("pressures", data=self.final_press)
            mixed_file.create_dataset("temperatures", data=self.final_temp)
            mixed_file.create_dataset("meanmolmass", data=self.mu)
            mixed_file.create_dataset("kpoints", data=self.combined_opacities)
            mixed_file.create_dataset("weighted Rayleigh cross-sections", data=self.combined_cross_sections)
            mixed_file.create_dataset("included molecules", data=self.molname_list)
            mixed_file.create_dataset("wavelengths", data=self.k_x)
            mixed_file.create_dataset("FastChem path", data=param.fastchem_path)

            if param.format == 'ktable':
                mixed_file.create_dataset("center wavelengths", data=self.k_x)
                mixed_file.create_dataset("interface wavelengths",data=self.k_i)
                mixed_file.create_dataset("wavelength width of bins",data=self.k_w)
                mixed_file.create_dataset("ypoints",data=self.k_y)

    def include_rayleigh_cross_section(self, ray, species, vol_mix_ratio):
        """ tabulates the rayleigh cross sections for various species """

        print("\nCalculating scattering cross sections ...")

        # this is kind of hidden -- TODO move to a better location at some point
        # references for the scattering data:
        # H2: Cox 2000
        # He: Sneep & Ubachs 2005, Thalman et al. 2014
        # H: Lee & Kim 2004
        # H2O: Murphy 1977, Wagner & Kretzschmar 2008
        # CO: Sneep & Ubachs 2005
        # CO2: Sneep & Ubachs 2005, Thalman et al. 2014
        # O2: Sneep & Ubachs 2005, Thalman et al. 2014
        # N2: Sneep & Ubachs 2005, Thalman et al. 2014

        list_of_implemented_scattering_species = ['H', 'H2', 'He', 'H2O', 'CO2', 'CO', 'O2', 'N2']

        if species.name in list_of_implemented_scattering_species:

            for t in range(self.final_nt):
                for p in range(self.final_np):

                    tls.percent_counter(t, self.final_nt, p, self.final_np)

                    for x in range(self.nx):

                        cross_section = 0

                        if species.name != 'H':

                            if species.name == "H2O":

                                index = ray.index_h2o(self.k_x[x], self.final_press[p], self.final_temp[t], self.mu[p + self.final_np * t])
                                n_ref = ray.n_ref_h2o(self.final_press[p], self.final_temp[t])
                                King = ray.King_h2o
                                lamda_limit = 2.5e-4

                            elif species.name == "CO2":

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

                        elif species.name == "H":

                            cross_section = ray.cross_sect_h(self.k_x[x])

                        self.combined_cross_sections[x + self.nx * p + self.nx * self.final_np * t] += vol_mix_ratio[p + self.final_np * t] * cross_section

        else:
            print("WARNING WARNING WARNING: Rayleigh scattering cross sections for species", species.name, "not found. Please double-check! Continuing without those... ")

    def weight_opacities(self, vol_mix_ratio, vol_mix_ratio2, mass, opac):
        """ weights opacities """

        print("\nWeighting by mass mixing ratio...")

        for t in range(self.final_nt):

            for p in range(self.final_np):

                mass_mix_ratio = vol_mix_ratio[p + self.final_np * t] * vol_mix_ratio2[p + self.final_np * t] * mass / self.mu[p + self.final_np * t]

                for x in range(self.nx):

                    tls.percent_counter(t, self.final_nt, p, self.final_np, x, self.nx)

                    for y in range(self.ny):

                        weighted_opac = mass_mix_ratio * opac[y + self.ny*x + self.ny*self.nx*p + self.ny*self.nx*self.final_np*t]

                        self.combined_opacities[y + self.ny*x + self.ny*self.nx*p + self.ny*self.nx*self.final_np*t] += weighted_opac

    def condense_out_species(self, cond, mix, cond_path, species):
        """
            completely remove a particular species below its condensation (stability) temperature
        """

        stab = cond.calc_stability_curve(cond_path, species)

        for t in range(self.final_nt):

            for p in range(self.final_np):

                if self.final_temp[t] < stab(npy.log10(self.final_press[p])):
                    mix[p + self.final_np * t] = min(1e-30, mix[p + self.final_np * t])

    def exp_decay_species(self, cond, mix, cond_path, condensate):
        """
            exponentially decay a  species which is being removed due to condensation
        """

        grad = 2e-2  # increase in volume mixing ratio with temperature (roughly estimated from Sharp & Burrows 2007)

        stab = cond.calc_stability_curve(cond_path, condensate)

        for p in range(self.final_np):

            for t in range(self.final_nt):

                if self.final_temp[t] < stab(npy.log10(self.final_press[p])):

                    mix[p + self.final_np * t] = mix[p + self.final_np * t] * 10 ** (-grad * (stab(npy.log10(self.final_press[p])) - self.final_temp[t]))

                else:
                    break

    # Warning: the implementation of condensation is quite hand-wavy
    # --> better to wait until it is implemented self-consistently in FastChem
    def apply_condensation(self, cond, param, vol_mix_ratio, species):
        """ removes/attenuates the condensate species below their condensation temperature or if affected by mineral formation """

        condensates = {
            "SiO": "MgSiO3",
            "Mn": "MnS",
            "Cr": "Cr",
            "Na": "Na2S",
            "K": "KCl",
            "Fe": "Fe"
        }

        # manually enriching a certain species
        # if species == 'CO2':
        #     vol_mix_ratio = [v * 1e2 for v in vol_mix_ratio]

        if species in ["TiO", "VO", "H2O"]:

            print("\nApplying condensation for " + species + ".")

            self.condense_out_species(cond, vol_mix_ratio, param.cond_path, species)

        elif species in ["SiO", "Mn", "Cr", "Na", "K", "Fe"]:

            print("\nApplying condensation for "+species+". It is removed from the gas phase by " + condensates[species] + " condensates.")

            self.exp_decay_species(cond, vol_mix_ratio, param.cond_path, condensates[species])

        else:
            print("\nNo condensation data found for "+species+". Skipping condensational effects.")

        return vol_mix_ratio

    def calc_h_minus(self, conti, param):
        """ calculates the H- continuum opacity """

        print("\nCalculating H- ...")

        total_opac = []

        # pressure and temperature dependent bound-free and free-free opacities
        for t in range(self.final_nt):

            for p in range(self.final_np):

                tls.percent_counter(t, self.final_nt, p, self.final_np)

                for x in range(self.nx):

                    cross_bf_h_min = conti.bf_cross_sect_h_min(self.k_x[x], self.final_temp[t], self.final_press[p], self.n_e[p + self.final_np * t]) / (pc.M_H * pc.AMU)

                    cross_ff_h_min = conti.ff_cross_sect_h_min(self.k_x[x], self.final_temp[t], self.final_press[p], self.n_e[p + self.final_np * t]) / (pc.M_H * pc.AMU)

                    for y in range(self.ny):

                        total_opac.append(cross_bf_h_min + cross_ff_h_min)

        # write to h5 file to allow for quick visualization later
        if param.format == 'ktable':
            ending = "_opac_ip.h5"

        elif param.format == 'sampling':
            ending = "_opac_ip_sampling.h5"

        self.write_h5(param, "H-" + ending, total_opac)

        # add to mol list
        self.molname_list.append("H-".encode('utf8'))

        print("\nH- calculation complete!")

        return total_opac

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
                    species.fc_name = column[4]
                    species.mass = float(column[5])

                    species_list.append(species)

        # we need an absorbing species as first entry so let's shuffle
        for s in range(len(species_list)):

            if species_list[s].absorbing == 'yes':

                species_list.insert(0, species_list[s])
                species_list.pop(s+1)

                break

            if s == len(species_list)-1:

                print("Oops! At least one species needs to be absorbing. Please check your 'final species' file. Aborting ... ")
                raise SystemExit()

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

        for n in range(len(species_list)):

            if self.is_number(species_list[n].mixing_ratio):  # checks if all characters are numeric. this excludes FastChem entries and CIA species

                mu += float(species_list[n].mixing_ratio) * species_list[n].mass

        self.mu = npy.ones(self.final_np * self.final_nt) * mu

    def add_one_species(self, param, ray, cond, conti, species, iter):

        print("\n\n----------\nGenerating mixed opacity table. Including --> " + species.name + " <--")

        # needs to be assigned beforehand to make python happy
        interpol_opac = None

        if species.name != "H-":

            # read in ktable for absorbing species
            if species.absorbing == 'yes':

                if param.format == 'ktable':
                    k_press, k_temp, kpoints = self.read_individual_ktable(param, species.name)
                elif param.format == 'sampling':
                    k_press, k_temp, kpoints = self.read_individual_opacity_for_sampling(param, species.name)

                # interpolate
                interpol_opac = self.interpolate_opacity(param, species.name, k_press, k_temp, kpoints)

        elif species.name == "H-":

            h_min_opac = self.calc_h_minus(conti, param)

            # no interpolation required. values already generated with new grid
            interpol_opac = h_min_opac

        # generate final arrays on first walkthrough (cannot do that earlier because need to know ny and nx first)
        if iter == 0:
            self.combined_opacities = npy.zeros(self.ny * self.nx * self.final_np * self.final_nt)
            self.combined_cross_sections = npy.zeros(self.nx * self.final_np * self.final_nt)

        # get abundances
        if "CIA" not in species.name:

            if species.mixing_ratio == "FastChem":

                    chem_vmr = npy.concatenate((self.fastchem_data_low[species.fc_name], self.fastchem_data_high[species.fc_name]))

                    final_vmr = self.interpolate_vmr_to_final_grid(chem_vmr)

            else:
                final_vmr = npy.ones(self.final_np * self.final_nt) * float(species.mixing_ratio)

            final_vmr_2 = npy.ones(self.final_np * self.final_nt)

        elif "CIA" in species.name:

            if species.mixing_ratio == "FastChem":

                two_fc_names = species.fc_name.split('&')

                chem_vmr = npy.concatenate((self.fastchem_data_low[two_fc_names[0]], self.fastchem_data_high[two_fc_names[0]]))
                chem_vmr_2 = npy.concatenate((self.fastchem_data_low[two_fc_names[1]], self.fastchem_data_high[two_fc_names[1]]))

                final_vmr = self.interpolate_vmr_to_final_grid(chem_vmr)
                final_vmr_2 = self.interpolate_vmr_to_final_grid(chem_vmr_2)

            else:
                two_vmrs = species.mixing_ratio.split('&')

                final_vmr = npy.ones(self.final_np * self.final_nt) * float(two_vmrs[0])
                final_vmr_2 = npy.ones(self.final_np * self.final_nt) * float(two_vmrs[1])

        if param.condensation == "yes":
            self.apply_condensation(cond, param, final_vmr, species.name)

        if species.absorbing == 'yes':

            # weight with mixing ratio
            self.weight_opacities(final_vmr, final_vmr_2, species.mass, interpol_opac)

        # calculate Rayleigh scattering component
        if species.scattering == 'yes':

            self.include_rayleigh_cross_section(ray, species, final_vmr)

    def combine_all_species(self, param, ray, cond, conti):

        # read in data from species file
        species_list = self.read_species_data(param.species_path)

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

            self.add_one_species(param, ray, cond, conti, species_list[n], n)

        self.create_mixed_file(param)

    @staticmethod
    def success():
        """ prints success message """
        print("\nCombination of opacities --- Successful!")


if __name__ == "__main__":
    print("This module is for the combination of the individual molecular opacities. Yes, the class is a comb. It combs through the opacities.")
