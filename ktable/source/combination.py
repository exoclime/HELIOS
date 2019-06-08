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


class Comb(object):
    """ class responsible for combining and mixing the individual k-tables """
    
    def __init__(self):
        self.k_temp = []
        self.k_press = []
        self.chem_press = None
        self.chem_temp = None
        self.mu = []
        self.kpoints = []
        self.k_y = []
        self.k_w = []
        self.k_i = []
        self.k_x = []

        self.na_k = []
        self.k_k = []

        self.molname_list = []
        self.fastchem_data_low = None
        self.fastchem_data_high = None
        self.nx = None
        self.ny = None
        self.np = None
        self.nt = None
        self.n_e = None

        self.mixed_opacities = []

        # scattering cross - sections
        self.weighted_cross_ray_table = []
        self.pure_cross_ray_h2_table = []
        self.pure_cross_ray_he_table = []
        self.pure_cross_ray_h_table = []
        self.pure_cross_ray_h2o_table = []
        self.pure_cross_ray_co2_table = []

        # continuous opacities
        self.opac_h_minus_bf = []
        self.opac_h_minus_ff = []
        self.weighted_opac_h_minus = []

        # entropy, kappa, etc.
        self.c_p = []
        self.kappa = []
        self.entropy = []
        self.entr_temp = None
        self.entr_nt = None
        self.entr_press = None
        self.entr_np = None

        # masses of atoms in AMU (useful for later)
        self.m_c = 12.0096
        self.m_n = 14.007
        self.m_o = 15.999
        self.m_f = 18.9984
        self.m_ne = 20.1797
        self.m_mg = 24.305
        self.m_al = 26.9815385
        self.m_si = 28.085
        self.m_p = 30.973761998
        self.m_s = 32.06
        self.m_cl = 35.45
        self.m_ar = 39.948
        self.m_ca = 40.078
        self.m_ti = 47.867
        self.m_v = 50.9415
        self.m_cr = 51.9961
        self.m_mn = 54.938044
        self.m_fe = 55.845
        self.m_cob = 58.933194
        self.m_ni = 58.6934
        self.m_cu = 63.546
        self.m_zn = 65.38

    # -------- generic methods -------- #

    @staticmethod
    def delete_duplicates(long_list):
        """ delete all duplicates in a list and return new list """
        short_list = []
        for item in long_list:
            if item not in short_list:
                short_list.append(item)
        return short_list

    def interpolate_to_new_grid(self, temp_old, temp_new, press_old, press_new, k_old):

        print("opacity temperature grid:\n", temp_old[:3], "...", temp_old[-3:])
        print("final table temperature grid:\n", temp_new[:3], "...", temp_new[-3:])
        print("opacity pressure grid:\n", press_old[:3], "...", press_old[-3:])
        print("final table pressure grid:\n", press_new[:3], "...", press_new[-3:])

        print("ny:",self.ny, "nx:",self.nx, "np:",self.np, "nt:",self.nt)

        k_new = npy.zeros(self.ny * self.nx * self.np * self.nt)

        old_nt = len(temp_old)
        old_np = len(press_old)

        print("old np:", old_np)
        print("old nt:", old_nt)

        for i in range(self.nt):

            for j in range(self.np):

                tls.percent_counter(i, self.nt, j, self.np)

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

                            k_new[y + self.ny * x + self.ny * self.nx * j + self.ny * self.nx * self.np * i] = \
                                k_old[y + self.ny*x + self.ny*self.nx*p_left + self.ny*self.nx*old_np*t_left]

                elif reduced_p != 1 and reduced_t == 1:

                    p_right = p_left + 1

                    for y in range(self.ny):
                        for x in range(self.nx):

                            k_new[y + self.ny*x + self.ny*self.nx*j + self.ny*self.nx*self.np*i] = \
                                (k_old[y + self.ny*x + self.ny*self.nx*p_right + self.ny*self.nx*old_np*t_left] * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                + k_old[y + self.ny*x + self.ny*self.nx*p_left + self.ny*self.nx*old_np*t_left] * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                                 ) / (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left]))

                elif reduced_p == 1 and reduced_t != 1:

                    t_right = t_left + 1

                    for y in range(self.ny):
                        for x in range(self.nx):

                            k_new[y + self.ny * x + self.ny * self.nx * j + self.ny * self.nx * self.np * i] = \
                                (k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) \
                                 + k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) \
                                 ) / (temp_old[t_right] - temp_old[t_left])

                elif reduced_p != 1 and reduced_t != 1:

                    p_right = p_left + 1
                    t_right = t_left + 1

                    for y in range(self.ny):
                        for x in range(self.nx):
                            try:
                                k_new[y + self.ny * x + self.ny * self.nx * j + self.ny * self.nx * self.np * i] = \
                                    (
                                     k_old[y + self.ny * x + self.ny * self.nx * p_right + self.ny * self.nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                     + k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_right] * (temp_new[i] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j]))\
                                     + k_old[y + self.ny * x + self.ny * self.nx * p_right + self.ny * self.nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_new[j]) - npy.log10(press_old[p_left])) \
                                     + k_old[y + self.ny * x + self.ny * self.nx * p_left + self.ny * self.nx * old_np * t_left] * (temp_old[t_right] - temp_new[i]) * (npy.log10(press_old[p_right]) - npy.log10(press_new[j])) \
                                     ) / ((temp_old[t_right] - temp_old[t_left]) * (npy.log10(press_old[p_right]) - npy.log10(press_old[p_left])))
                            except IndexError:
                                print("IndexError at:", y, x, j, i, p_left, p_right, t_left, t_right, len(k_old), len(temp_new), len(press_new))

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
                mixed_file.create_dataset("pressures", data=self.chem_press)
                mixed_file.create_dataset("temperatures", data=self.chem_temp)
                mixed_file.create_dataset("interface wavelengths", data=self.k_i)
                mixed_file.create_dataset("center wavelengths", data=self.k_x)
                mixed_file.create_dataset("wavelength width of bins", data=self.k_w)
                mixed_file.create_dataset("ypoints", data=self.k_y)
                mixed_file.create_dataset("kpoints", data=ktable)
        elif param.format == "sampling":
            with h5py.File(param.resampling_path + name, "w") as mixed_file:
                mixed_file.create_dataset("pressures", data=self.chem_press)
                mixed_file.create_dataset("temperatures", data=self.chem_temp)
                mixed_file.create_dataset("wavelengths", data=self.k_x)
                mixed_file.create_dataset("opacities", data=ktable)

    def read_individual_ktable(self, param, name):
        """ read in individual ktables """

        try:
            with h5py.File(param.resampling_path + name + "_opacities.h5", "r") as ind_file:

                self.kpoints = [k for k in ind_file["kpoints"][:]]
                self.k_y = [y for y in ind_file["ypoints"][:]]
                self.k_x = [x for x in ind_file["center wavelengths"][:]]
                self.k_w = [w for w in ind_file["wavelength width of bins"][:]]
                self.k_i = [i for i in ind_file["interface wavelengths"][:]]
                self.k_temp = [t for t in ind_file["temperatures"][:]]
                self.k_press = [p for p in ind_file["pressures"][:]]

                self.nx = len(self.k_x)
                self.ny = len(self.k_y)
                self.np_mol = len(self.k_press)
                self.nt_mol = len(self.k_temp)

            print("\nIncluding " + name + "_opacities.h5 !")
            self.molname_list.append(name.encode('utf8'))
        except(OSError):
            print("\nABORT - " + name + "_opacities.h5 not found!")
            raise SystemExit()

    def read_individual_opacity_for_sampling(self, param, name):
        """ read in individual sampled opacities """

        try:
            with h5py.File(param.resampling_path + name + "_opac_sampling.h5", "r") as ind_file:

                self.kpoints = [k for k in ind_file["opacities"][:]]
                self.k_x = [x for x in ind_file["wavelengths"][:]]
                self.k_temp = [t for t in ind_file["temperatures"][:]]
                self.k_press = [p for p in ind_file["pressures"][:]]

                self.nx = len(self.k_x)
                self.ny = 1
                self.ny = 1

            print("\nIncluding " + name + "_opac_sampling.h5 !")
            self.molname_list.append(name.encode('utf8'))
        except(OSError):
            print("\nABORT - " + name + "_opacities.h5 not found!")
            raise SystemExit()

    def load_fastchem_data(self, param):
        """ read in the fastchem mixing ratios"""

        self.fastchem_data_low = npy.genfromtxt(param.fastchem_path + 'chem_low.dat',
                                  names=True, dtype=None, skip_header=0)

        self.fastchem_data_high = npy.genfromtxt(param.fastchem_path + 'chem_high.dat',
                                   names=True, dtype=None, skip_header=0)

        # parameters
        press = npy.concatenate((self.fastchem_data_low['Pbar'], self.fastchem_data_high['Pbar']))
        temp = npy.concatenate((self.fastchem_data_low['Tk'], self.fastchem_data_high['Tk']))
        self.mu = npy.concatenate((self.fastchem_data_low['mu'], self.fastchem_data_high['mu']))

        # electrons
        self.n_e = npy.concatenate((self.fastchem_data_low['e_minus'], self.fastchem_data_high['e_minus']))

        # overwrite pressure and temperature grid with the one from FastChem
        press = self.delete_duplicates(press)
        self.chem_press = [p * 1e6 for p in press]
        self.chem_temp = self.delete_duplicates(temp)

        self.nt = len(self.chem_temp)
        self.np = len(self.chem_press)

    @staticmethod
    def check_if_already_interpolated(param, name, ending):
        """ checks whether necessary to interpolate or file already exists """

        try:
            h5py.File(param.resampling_path + name + ending, "r")
            print("Interpolated file " + param.resampling_path + name + ending + " already exists. Skipping interpolation for this molecule...")
            return True

        except OSError:
            return False

    def interpolate_molecule(self, param, name, opac_list):

        print("\nInterpolating...")

        if param.format == 'ktable':
            ending = "_opac_ip.h5"
            database = "kpoints"
        elif param.format == 'sampling':
            ending = "_opac_ip_sampling.h5"
            database = "opacities"


        if self.check_if_already_interpolated(param, name, ending):

            opac_list = []
            with h5py.File(param.resampling_path + name + ending, "r") as ip_file:
                for k in ip_file[database][:]:
                    opac_list.append(k)

        else:
            opac_list = self.interpolate_to_new_grid(self.k_temp, self.chem_temp, self.k_press, self.chem_press, opac_list)
            self.write_h5(param, name + ending, opac_list)

        return opac_list

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
            mixed_file.create_dataset("pressures", data=self.chem_press)
            mixed_file.create_dataset("temperatures", data=self.chem_temp)
            mixed_file.create_dataset("meanmolmass", data=self.mu)
            mixed_file.create_dataset("kpoints", data=self.mixed_opacities)
            mixed_file.create_dataset("weighted Rayleigh cross-sections", data=self.weighted_cross_ray_table)
            mixed_file.create_dataset("pure H2 Rayleigh cross-sections", data=self.pure_cross_ray_h2_table)
            mixed_file.create_dataset("pure He Rayleigh cross-sections", data=self.pure_cross_ray_he_table)
            mixed_file.create_dataset("pure H2O Rayleigh cross-sections", data=self.pure_cross_ray_h2o_table)
            mixed_file.create_dataset("pure CO2 Rayleigh cross-sections", data=self.pure_cross_ray_co2_table)
            mixed_file.create_dataset("pure H Rayleigh cross-sections", data=self.pure_cross_ray_h_table)
            mixed_file.create_dataset("included molecules", data=self.molname_list)
            mixed_file.create_dataset("wavelengths", data=self.k_x)
            mixed_file.create_dataset("FastChem path", data=param.fastchem_path)

            if param.format == 'ktable':
                mixed_file.create_dataset("center wavelengths", data=self.k_x)
                mixed_file.create_dataset("interface wavelengths",data=self.k_i)
                mixed_file.create_dataset("wavelength width of bins",data=self.k_w)
                mixed_file.create_dataset("ypoints",data=self.k_y)

    def tabulate_rayleigh_cross_section(self, ray, param, f_h2o):
        """ tabulates the rayleigh cross sections for various species """

        print("Tabulating Rayleigh cross-sections...")

        try:
            with h5py.File(param.final_path + "Rayleigh.h5", "r") as ray_file:
                self.weighted_cross_ray_table = [o for o in ray_file["weighted Rayleigh cross-sections"][:]]
                self.pure_cross_ray_h2_table = [o for o in ray_file["pure H2 Rayleigh cross-sections"][:]]
                self.pure_cross_ray_he_table = [o for o in ray_file["pure He Rayleigh cross-sections"][:]]
                self.pure_cross_ray_h_table = [o for o in ray_file["pure H Rayleigh cross-sections"][:]]
                self.pure_cross_ray_co2_table = [o for o in ray_file["pure CO2 Rayleigh cross-sections"][:]]
                self.pure_cross_ray_h2o_table = [o for o in ray_file["pure H2O Rayleigh cross-sections"][:]]


            print("File " + param.final_path + "Rayleigh.h5 already exists. Reading of Rayleigh cross-sections successful.")

        except OSError or KeyError:

            # obtain relevant volume mixing ratios
            f_h2 = npy.concatenate((self.fastchem_data_low['H2'], self.fastchem_data_high['H2']))
            f_he = npy.concatenate((self.fastchem_data_low['He'], self.fastchem_data_high['He']))
            f_co2 = npy.concatenate((self.fastchem_data_low['C1O2'], self.fastchem_data_high['C1O2']))
            f_h = npy.concatenate((self.fastchem_data_low['H'], self.fastchem_data_high['H']))

            if param.special_abundance == 'pure_H2O':
                f_h2o = [1 for i in range(len(f_h2o))]
                f_co2 = [0 for i in range(len(f_co2))]
                f_h2 = [0 for i in range(len(f_h2))]
                f_h = [0 for i in range(len(f_h))]
                f_he = [0 for i in range(len(f_he))]

            elif param.special_abundance == 'pure_CO2':
                f_h2o = [0 for i in range(len(f_h2o))]
                f_co2 = [1 for i in range(len(f_co2))]
                f_h2 = [0 for i in range(len(f_h2))]
                f_h = [0 for i in range(len(f_h))]
                f_he = [0 for i in range(len(f_he))]

            elif param.special_abundance == 'venus':
                f_h2o = [3e-5 for i in range(len(f_h2o))]
                f_co2 = [1 for i in range(len(f_co2))]
                f_h2 = [0 for i in range(len(f_h2))]
                f_h = [0 for i in range(len(f_h))]
                f_he = [0 for i in range(len(f_he))]

            for t in range(self.nt):
                for p in range(self.np):

                    tls.percent_counter(t, self.nt, p, self.np)

                    for x in range(self.nx):

                        cross_ray_h2o = ray.cross_sect(self.k_x[x], ray.index_h2o(self.k_x[x], self.chem_press[p], self.chem_temp[t], self.mu[p + self.np * t]), ray.n_ref_h2o(self.chem_press[p], self.chem_temp[t]), ray.King_h2o, lamda_limit=2.5e-4)
                        cross_ray_h2 = ray.cross_sect(self.k_x[x], ray.index_h2(self.k_x[x]), ray.n_ref_h2, ray.King_h2)
                        cross_ray_he = ray.cross_sect(self.k_x[x], ray.index_he(self.k_x[x]), ray.n_ref_he, ray.King_he)
                        cross_ray_co2 = ray.cross_sect(self.k_x[x], ray.index_co2(self.k_x[x]), ray.n_ref_co2, ray.King_co2(self.k_x[x]))
                        cross_ray_h = ray.cross_sect_h(self.k_x[x])

                        self.pure_cross_ray_h2o_table.append(cross_ray_h2o)

                        # the following cross-sections depend only on the wavelength
                        if t == 0 and p == 0:
                            self.pure_cross_ray_h2_table.append(cross_ray_h2)
                            self.pure_cross_ray_he_table.append(cross_ray_he)
                            self.pure_cross_ray_co2_table.append(cross_ray_co2)
                            self.pure_cross_ray_h_table.append(cross_ray_h)

                        # mix everything according to their volume (number) mixing ratios
                        mix =     f_h2[p + self.np * t] * cross_ray_h2 \
                                + f_he[p + self.np * t] * cross_ray_he \
                                + f_h2o[p + self.np * t] * cross_ray_h2o \
                                + f_co2[p + self.np * t] * cross_ray_co2 \
                                + f_h[p + self.np * t] * cross_ray_h

                        self.weighted_cross_ray_table.append(mix)

            try:
                os.makedirs(param.final_path)
            except OSError:
                if not os.path.isdir(param.final_path):
                    raise

            with h5py.File(param.final_path + "Rayleigh.h5", "w") as ray_file:
                ray_file.create_dataset("pressures", data=self.chem_press)
                ray_file.create_dataset("temperatures", data=self.chem_temp)
                ray_file.create_dataset("wavelengths", data=self.k_x)
                ray_file.create_dataset("weighted Rayleigh cross-sections", data=self.weighted_cross_ray_table)
                ray_file.create_dataset("pure H2 Rayleigh cross-sections", data=self.pure_cross_ray_h2_table)
                ray_file.create_dataset("pure He Rayleigh cross-sections", data=self.pure_cross_ray_he_table)
                ray_file.create_dataset("pure H2O Rayleigh cross-sections", data=self.pure_cross_ray_h2o_table)
                ray_file.create_dataset("pure CO2 Rayleigh cross-sections", data=self.pure_cross_ray_co2_table)
                ray_file.create_dataset("pure H Rayleigh cross-sections", data=self.pure_cross_ray_h_table)
                ray_file.create_dataset("FastChem path", data=param.fastchem_path)

        ray.success()

    def weight_opacities(self, param, species, vol_mix_ratio, mass, kpoints, cia='no'):
        """ weights opacities """

        try:
            with h5py.File(param.resampling_path + species + ".h5", "r") as weightfile:

                self.mixed_opacities = [k for k in weightfile["weighted kpoints"][:]]

            print("\nWeighted file " + param.resampling_path + species + ".h5 already exists. Reading in weighted opacities for this molecule...")

            if len(self.mixed_opacities) == self.nt * self.np * self.nx * self.ny:

                print("Dimensions test passed\n")

            else:
                print("Dimensions test failed. Aborting...")
                raise SystemExit()

        except OSError:

            if cia == 'yes':
                vol_mix_ratio2 = npy.concatenate((self.fastchem_data_low['H2'], self.fastchem_data_high['H2']))

            print("\nWeighting by mass mixing ratio...")

            for t in range(self.nt):

                for p in range(self.np):

                    mass_mix_ratio = vol_mix_ratio[p + self.np * t] * mass / self.mu[p + self.np * t]

                    if param.special_abundance == 'pure_H2O':
                        mass_mix_ratio = 1
                        self.mu[p + self.np * t] = 18.0153

                    elif param.special_abundance == 'pure_CO2':
                        mass_mix_ratio = 1
                        self.mu[p + self.np * t] = 44.01

                    if param.special_abundance == 'venus' and species == 'H2O_weighted':
                        mass_mix_ratio = 1.227e-5
                        self.mu[p + self.np * t] = 44.01

                    if param.special_abundance == 'venus' and species == 'CO2_weighted':
                        mass_mix_ratio = 1
                        self.mu[p + self.np * t] = 44.01

                    if cia =='yes':

                        mass_mix_ratio *= vol_mix_ratio2[p + self.np * t]

                    for x in range(self.nx):

                        tls.percent_counter(t, self.nt, p, self.np, x, self.nx)

                        for y in range(self.ny):

                            mixed = mass_mix_ratio * kpoints[y + self.ny * x + self.ny * self.nx * p + self.ny * self.nx * self.np * t]

                            self.mixed_opacities.append(mixed)

            with h5py.File(param.resampling_path + species + ".h5", "w") as weightfile:

                weightfile.create_dataset("weighted kpoints", data=self.mixed_opacities)

    def add_to_mixed_file(self, param):
        """ add molecular opacities to the mixed opacity file """

        if param.format == 'ktable':
            filename = "mixed_opac_ktable.h5"
        elif param.format == 'sampling':
            filename = "mixed_opac_sampling.h5"

        old_kpoints = []
        new_kpoints = []

        try:
            with h5py.File(param.final_path + filename, "r") as mixed_file:

                for k in mixed_file["kpoints"][:]:
                    old_kpoints.append(k)

        except:
            print("ABORT - something wrong with reading the mixed_opac_*.h5 file!")
            raise SystemExit()

        for k in range(len(old_kpoints)):

            new_kpoints.append(old_kpoints[k] + self.mixed_opacities[k])

        with h5py.File(param.final_path + filename, "a") as mixed_file:

            del mixed_file["kpoints"]
            del mixed_file["included molecules"]

            mixed_file.create_dataset("kpoints",data=new_kpoints)

            mixed_file.create_dataset("included molecules", data=self.molname_list)

    def clear_memory(self):
        """ clears the opacity arrays so they can be filled again with new data """

        self.kpoints = []
        self.mixed_opacities = []

    def condense_out_species(self, cond, mix, ele_abund, species):
        """
            completely remove a particular species below its condensation (stability) temperature
        """

        stab = cond.calc_stability_curve(ele_abund, species)

        for t in range(self.nt):

            for p in range(self.np):

                if self.chem_temp[t] < stab(npy.log10(self.chem_press[p])):
                    mix[p + self.np * t] = min(1e-30, mix[p + self.np * t])

    def exp_decay_species(self, cond, mix, ele_abund, condensate):
        """
            exponentially decay a  species which is being removed due to condensation
        """

        grad = 2e-2  # increase in volume mixing ratio with temperature (roughly estimated from Sharp & Burrows 2007)

        stab = cond.calc_stability_curve(ele_abund, condensate)

        for p in range(self.np):

            for t in range(self.nt):

                if self.chem_temp[t] < stab(npy.log10(self.chem_press[p])):

                    mix[p + self.np * t] = mix[p + self.np * t] * 10 ** (-grad * (stab(npy.log10(self.chem_press[p])) - self.chem_temp[t]))

                else:
                    break

    # by default condensation is switched off (to activate set override to False) as the implementation is quite hand-wavy
    # --> better to wait until it is implemented self-consistently in FastChem
    def apply_condensation(self, cond, param, fc_name, species, override=False):
        """ removes/attenuates the condensate species below their condensation temperature or if affected by mineral formation """

        condensates = {
            "SiO": "MgSiO3",
            "Mn": "MnS",
            "Cr": "Cr",
            "Na": "Na2S",
            "K": "KCl",
            "Fe": "Fe"
        }

        # read in the eq. chem. gas mixing ratios
        vol_mix_ratio = npy.concatenate((self.fastchem_data_low[fc_name], self.fastchem_data_high[fc_name]))

        if override is False:

            if species in ["TiO", "VO", "H2O"]:

                print("\nApplying condensation for " + species + ".")

                self.condense_out_species(cond, vol_mix_ratio, param.ele_abund, species)

            elif species in ["SiO", "Mn", "Cr", "Na", "K", "Fe"]:

                print("\nApplying condensation for "+species+". It is removed from the gas phase by " + condensates[species] + " condensates.")

                self.exp_decay_species(cond, vol_mix_ratio, param.ele_abund, condensates[species])

            else:
                print("\nNo condensation data found for "+species+". Skipping condensational effects.")

        elif override is True:

            print("\nCondensation for " + species + " overridden.")

        return vol_mix_ratio

    # -------- procedural methods -------- #

    def do_water(self, param, ray, cond):

        print("\n\n----------\nStarting opacity table by including water...")

        if param.format == 'ktable':
            self.read_individual_ktable(param, "H2O")
        elif param.format == 'sampling':
            self.read_individual_opacity_for_sampling(param, "H2O")

        self.load_fastchem_data(param)

        self.kpoints = self.interpolate_molecule(param, "H2O", self.kpoints)

        # read in data from species file
        species_data = npy.genfromtxt(param.species_path, names=True, dtype=None)
        species = npy.array(species_data['your_species_name'], dtype='U')
        fastchem_name = npy.array(species_data['name_in_FastChem'], dtype='U')
        mass = species_data['mass_in_AMU']

        try:
            for n in range(len(species)):

                if species[n] == 'H2O':

                    vol_mix_ratio_h2o = self.apply_condensation(cond, param, fastchem_name[n], species[n], override=True)

                    self.weight_opacities(param, "H2O_weighted", vol_mix_ratio_h2o, mass[n], self.kpoints)

                    if param.special_abundance == 'no_H2O': # use this option for a model with no water content
                        self.weight_opacities(param, "H2O_zero", vol_mix_ratio_h2o, mass[n], npy.zeros(len(self.kpoints)))
        except TypeError:

                    vol_mix_ratio_h2o = self.apply_condensation(cond, param, 'H2O1', 'H2O', override=True)

                    self.weight_opacities(param, "H2O_weighted", vol_mix_ratio_h2o, mass, self.kpoints)

        # Rayleigh scattering inclusion
        self.tabulate_rayleigh_cross_section(ray, param, vol_mix_ratio_h2o)

        self.create_mixed_file(param)

        self.clear_memory()

    def add_other_species(self, param, cond, species, fastchem_name, mass, cia='no'):

        if param.format == 'ktable':
            self.read_individual_ktable(param, species)
        elif param.format == 'sampling':
            self.read_individual_opacity_for_sampling(param, species)

        self.kpoints = self.interpolate_molecule(param, species, self.kpoints)

        vol_mix_ratio = self.apply_condensation(cond, param, fastchem_name, species)

        # ### plotting check ###
        # import matplotlib.pyplot as plt
        #
        # p = 18
        # print(self.chem_press[p])
        # vmr_plot = [vol_mix_ratio[p + self.np * t] for t in range(self.nt)]
        # color_list = ['blue', 'green', 'red', 'magenta', 'cyan', 'orange']
        # plt.plot(self.chem_temp, vmr_plot, c=color_list[npy.random.randint(len(color_list))], label=species)
        # plt.yscale('log')
        # plt.show()
        # ###

        self.weight_opacities(param, species+"_weighted", vol_mix_ratio, mass, self.kpoints, cia)

        self.add_to_mixed_file(param)

        self.clear_memory()

    def calc_h_minus(self, conti, mass):
        """ calculates the H- continuum opacity """

        print("\nCalculating H- ...")

        # pressure and temperature dependent bound-free and free-free opacities
        for t in range(self.nt):

            for p in range(self.np):

                for x in range(self.nx):

                    cross_bf_h_min = conti.bf_cross_sect_h_min(self.k_x[x], self.chem_temp[t], self.chem_press[p], self.n_e[p + self.np * t])

                    cross_ff_h_min = conti.ff_cross_sect_h_min(self.k_x[x], self.chem_temp[t], self.chem_press[p], self.n_e[p + self.np * t])

                    self.opac_h_minus_bf.append(cross_bf_h_min / (mass * pc.AMU))

                    self.opac_h_minus_ff.append(cross_ff_h_min / (mass * pc.AMU))

        vol_mix_ratio = npy.concatenate((self.fastchem_data_low['H'], self.fastchem_data_high['H']))

        for t in range(self.nt):

            for p in range(self.np):

                mass_mix_ratio = vol_mix_ratio[p + self.np * t] * mass / self.mu[p + self.np * t]

                for x in range(self.nx):

                    whole_weighted_thingy = mass_mix_ratio * (self.opac_h_minus_bf[x + self.nx * p + self.nx * self.np * t] + self.opac_h_minus_ff[x + self.nx * p + self.nx * self.np * t])

                    self.weighted_opac_h_minus.append(whole_weighted_thingy)

        print("\nH- calculation complete!")

    @staticmethod
    def h5_read(param, name):
        """
            checks if h5 container exists
        """

        with h5py.File(param.final_path + name + "_opacities.h5", "r") as file:
            weighted_opac = [o for o in file["weighted opacities"][:]]
            print("File " + param.final_path + name + "_opacities.h5 already exists. Reading of opacities successful.")

        return weighted_opac

    def add_continuous_to_mixed(self, param, weighted_continuous, name):
        """
            adds weighted continuous opacity to the mixed file
        """

        if param.format == 'ktable':
            filename = "mixed_opac_ktable.h5"
        elif param.format == 'sampling':
            filename = "mixed_opac_sampling.h5"

        old_kpoints = []
        new_kpoints = []

        try:
            with h5py.File(param.final_path + filename, "r") as mixed_file:

                for k in mixed_file["kpoints"][:]:
                    old_kpoints.append(k)

        except:
            print("ABORT - something wrong with reading the mixed_opac_*.h5 file!")
            raise SystemExit()

        for t in range(self.nt):

            for p in range(self.np):

                for x in range(self.nx):

                    for y in range(self.ny):

                        new_kpoints.append(old_kpoints[y + self.ny * x + self.ny * self.nx * p + self.ny * self.nx * self.np * t] + weighted_continuous[x + self.nx * p + self.nx * self.np * t])

        self.molname_list.append(name.encode('utf8'))

        with h5py.File(param.final_path + filename, "a") as mixed_file:

            del mixed_file["kpoints"]
            del mixed_file["included molecules"]

            mixed_file.create_dataset("kpoints",data=new_kpoints)

            mixed_file.create_dataset("included molecules", data=self.molname_list)

    def add_h_minus(self, param, conti, name, mass):
        """
            calculates and adds H- opacities if not calculated yet
        """

        try:
            self.weighted_opac_h_minus = self.h5_read(param, name)

        except OSError or KeyError:

            self.calc_h_minus(conti, mass)

            with h5py.File(param.final_path + name + "_opacities.h5", "w") as f:

                f.create_dataset("pressures", data=self.chem_press)
                f.create_dataset("temperatures", data=self.chem_temp)
                f.create_dataset("wavelengths", data=self.k_x)
                f.create_dataset("bf opacities", data=self.opac_h_minus_bf)
                f.create_dataset("ff opacities", data=self.opac_h_minus_ff)
                f.create_dataset("weighted opacities", data=self.weighted_opac_h_minus)

        self.add_continuous_to_mixed(param, self.weighted_opac_h_minus, name)

    def add_cia_cross_sections(self, param, cond, species, fc_name):

        # reads in cross-sections
        with h5py.File(param.resampling_path + species + ".h5", "r") as file:
            sigma_pre = [s for s in file["cross-sections"][:]]

        # reads in mixing ratios (only works for twin-CIA at the moment --> improve in future!)
        vol_mix_ratio = self.apply_condensation(cond, param, fc_name, species)

        weighted_cia_opac = []

        for t in range(self.nt):
            for p in range(self.np):
                for x in range(self.nx):

                    mix = sigma_pre[x + self.nx*p + self.nx*self.np*t] * vol_mix_ratio[p + self.np * t]**2 / (self.mu[p + self.np * t] * pc.AMU)

                    if param.special_abundance == 'pure_CO2':
                        mix = sigma_pre[x + self.nx * p + self.nx * self.np * t] / (44.01 * pc.AMU)

                    if param.special_abundance == 'venus':
                        mix = sigma_pre[x + self.nx * p + self.nx * self.np * t] / (44.01 * pc.AMU)

                    weighted_cia_opac.append(mix)

        self.add_continuous_to_mixed(param, weighted_cia_opac, species)


    def include_the_other_species(self, param, cond, conti):

        # read in data from species file
        species_data = npy.genfromtxt(param.species_path, names=True, dtype=None)
        species = npy.array(species_data['your_species_name'], dtype='U')
        fastchem_name = npy.array(species_data['name_in_FastChem'], dtype='U')
        mass = species_data['mass_in_AMU']

        try:
            for n in range(len(species)):

                if species[n] == 'H2O':
                    continue  # water is already included

                elif species[n] == "H-":
                    self.add_h_minus(param, conti, species[n], mass[n])
                    print("\n--------\nAdding species: H-")

                elif ("CIA" in species[n]) and ("cross" in species[n]):
                    print("\n--------\nAdding species:", species[n])
                    self.add_cia_cross_sections(param, cond, species[n], fastchem_name[n])

                elif "CIA" in species[n] and ("cross" not in species[n]):
                    print("\n--------\nAdding species:", species[n])
                    self.add_other_species(param, cond, species[n], fastchem_name[n], mass[n], cia='yes')
                else:
                    print("\n--------\nAdding species:", species[n], ", FastChem name:", fastchem_name[n])
                    self.add_other_species(param, cond, species[n], fastchem_name[n], mass[n])

        except TypeError:
            if species == 'H2O':
                pass  # water is already included

            else:
                print("\nERROR: Water must be included first into final table. Aborting process...")

            # elif species == "H-":
            #     self.add_h_minus(param, conti, species, mass)
            #     print("\n--------\nAdding species: H-")
            #
            # elif "CIA" in species:
            #     print("\n--------\nAdding species:", species)
            #     self.add_other_species(param, cond, species, fastchem_name, mass, cia='yes')
            # else:
            #     print("\n--------\nAdding species:", species, ", FastChem name:", fastchem_name)
            #     self.add_other_species(param, cond, species, fastchem_name, mass)

    @staticmethod
    def success():
        """ prints success message """
        print("\nCombination of opacities --- Successful!")


if __name__ == "__main__":
    print("This module is for the combination of the individual molecular opacities. Yes, the class is a comb. It combs through the opacities.")
