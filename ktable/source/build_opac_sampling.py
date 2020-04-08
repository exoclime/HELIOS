# ==============================================================================
# This program generates the resampled opacities from the HELIOS-K output
# Copyright (C) 2018 Matej Malik
#
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
import h5py
import sys
from source import tools as tls


class Production(object):
    """ class to produce the molecular k-distribution functions """

    def __init__(self):
        self.species_name = []
        self.species_path = []
        self.rt_lamda = []
        self.rt_nu = []

    def read_param_sampling(self, param):

        with open(param.sampling_param_path, 'r')as sfile:
            next(sfile)
            for line in sfile:
                column = line.split()
                if column:
                    self.species_name.append(column[0])
                    self.species_path.append(column[1])

    @staticmethod
    def read_dat_file(path):

        opac_list = []

        with open(path, 'r')as kfile:

            for line in kfile:

                column = line.split()
                if len(column) > 0:

                    opac_list.append(column[1])

        return opac_list

    def initialize_wavelength_grid(self, param):

        if param.special_limits is None:

            # set maximum at 30 micron
            top_limit = 30 * 1e-4

            # starting from 0.34 micron in order to avoid conflict with lower opacity boundary at 0.33
            l_point = 0.34 * 1e-4

        else:

            top_limit = param.special_limits[1] * 1e-4
            l_point = param.special_limits[0] * 1e-4

        while l_point < top_limit:
            self.rt_lamda.append(l_point)

            l_point *= (param.resolution + 1) / param.resolution

        self.rt_nu = [1 / l for l in self.rt_lamda]
        self.rt_nu.reverse()
        self.rt_nu = [round(n, 2) for n in self.rt_nu]
        print(self.rt_nu[:5], self.rt_nu[-5:])
        raise SystemExit()

    def big_loop(self, param):

        for m in range(len(self.species_name)):

            if not self.species_path[m].endswith("/"):
                self.species_path[m] += "/"

            # get molecular parameter ranges
            files = os.listdir(self.species_path[m])
            file_list = [f for f in files if "Out_" in f]
            temp_list = []
            press_list = []
            numin_list = []
            numax_list = []
            press_exp_list = []
            species_nr = 'not provided'

            for f in file_list:

                if 'Opacity_Atoms' in self.species_path[m]:
                    # numin_list.append(int(f[9:14]))  # taking hardcoded wavenumber limits (see below), because some molecules have lower boundaries
                    # numax_list.append(int(f[15:20]))
                    temp_list.append(int(f[21:26]))
                    if species_nr == 'not provided':
                        species_nr = int(f[4:8])
                elif 'Opacity3' in self.species_path[m]:
                    # numin_list.append(int(f[4:9]))
                    # numax_list.append(int(f[10:15]))
                    temp_list.append(int(f[16:21]))
                elif 'h2h2o_' in self.species_path[m]:
                    temp_list.append(int(f[28:33]))
                else:
                    # numin_list.append(int(f[7:12]))
                    # numax_list.append(int(f[13:18]))
                    temp_list.append(int(f[19:24]))
                    if species_nr == 'not provided':
                        species_nr = int(f[4:6])

            # hardcoded wavenumber limits -- improve this at some point in future
            numin_list = npy.arange(0, 30000, 1000)
            numax_list = npy.arange(1000, 31000, 1000)

            if 'h2h2o_' in self.species_path[m]:
                numin_list = npy.arange(0, 41000, 1000)
                numax_list = npy.arange(1000, 42000, 1000)

            # delete duplicate entries in the lists and sort in ascending order
            temp_list = list(set(temp_list))
            temp_list.sort()

            temp_min = min(temp_list)
            temp_max = max(temp_list)
            numin = min(numin_list)
            numax = max(numax_list)

            if 'h2h2o_' in self.species_path[m]:

                press_list_p1 = [10 ** p for p in npy.arange(0, 10, 1)]
                press_list_p2 = [10 ** p for p in npy.arange(0.5, 9.5, 1)]
                press_list = npy.append(press_list_p1, press_list_p2)
                press_list.sort()

                press_exp_list = ['n600', 'n550',
                                  'n500', 'n450',
                                  'n400', 'n350',
                                  'n300', 'n250',
                                  'n200', 'n150',
                                  'n100', 'n050',
                                  'p000', 'p050',
                                  'p100', 'p150',
                                  'p200', 'p250',
                                  'p300']

            elif 'Opacity_Atoms' in self.species_path[m]:

                press_list = [1e-2]

                press_exp_list = ['n800']

            # hardcoded pressures -- improve this at some point in future
            else:
                press_list_p1 = [10 ** p for p in npy.arange(0, 10, 1)]
                press_list_p2 = [10**p for p in npy.arange(0.33333333, 9.33333333, 1)]
                press_list_p3 = [10**p for p in npy.arange(0.66666666, 9.66666666, 1)]
                press_list = npy.append(press_list_p1,npy.append(press_list_p2,press_list_p3))
                press_list.sort()

                press_exp_list = ['n600','n566','n533',
                                  'n500','n466','n433',
                                  'n400','n366','n333',
                                  'n300','n266','n233',
                                  'n200','n166','n133',
                                  'n100','n066','n033',
                                  'p000','p033','p066',
                                  'p100','p133','p166',
                                  'p200','p233','p266',
                                  'p300']

            # some user feedback to check whether all is fine
            print("\n--- working on ---")
            print("molecule or atom: ", self.species_name[m])
            print("Hitran ID: ", species_nr)
            print("wavenumber range: ", numin, numax)
            print("temperature range: ", temp_min, temp_max)
            print("number of wavelength bins:", len(self.rt_nu),"\n")

            opac_array = []

            # read files
            for t in range(len(temp_list)):

                for p in range(len(press_exp_list)):

                    opac_array_temp = []

                    for n in range(len(numin_list)):

                        exist = 1

                        tls.percent_counter(t, len(temp_list), p, len(press_exp_list), n, len(numin_list))

                        if 'Opacity_Atoms' in self.species_path[m]:

                            file = self.species_path[m] + "Out_{:04d}_{:05d}_{:05d}_{:05d}_".format(species_nr, numin_list[n], numax_list[n], temp_list[t]) + press_exp_list[p] + ".bin"

                            try:

                                content = npy.fromfile(file, npy.float32, -1, "")

                            except IOError:

                                print("WARNING: File '" + file + "' not found. Using value 1e-15 for opacity in this regime.")
                                exist = 0

                        elif 'Opacity2' in self.species_path[m]:

                            file = self.species_path[m] + "Out_{:02d}_{:05d}_{:05d}_{:05d}_".format(species_nr, numin_list[n], numax_list[n], temp_list[t]) + press_exp_list[p] + ".bin"

                            try:

                                content = npy.fromfile(file,npy.float32, -1, "")

                            except IOError:

                                print("WARNING: File '" + file + "' not found. Using value 1e-15 for opacity in this regime.")
                                exist = 0

                        elif 'Opacity3' in self.species_path[m]:

                            file = self.species_path[m] + "Out_{:05d}_{:05d}_{:05d}_".format(numin_list[n], numax_list[n], temp_list[t]) + press_exp_list[p] + ".bin"

                            try:

                                content = npy.fromfile(file,npy.float32, -1, "")

                            except IOError:

                                print("WARNING: File '" + file + "' not found. Using value 1e-15 for opacity in this regime.")
                                exist = 0

                        elif 'h2h2o_' in self.species_path[m]:

                            file = self.species_path[m] + "Out_{}_{:05d}_{:05d}_{:05d}_".format(self.species_name[m], numin_list[n], numax_list[n], temp_list[t]) + press_exp_list[p] + ".dat"

                            try:

                                content = self.read_dat_file(file)

                            except IOError:

                                print("WARNING: File '" + file + "' not found. Using value 1e-15 for opacity in this regime.")
                                exist = 0

                        else:

                            file = self.species_path[m] + "Out_{:02d}_{:05d}_{:05d}_{:05d}_".format(species_nr, numin_list[n], numax_list[n], temp_list[t]) + press_exp_list[p] + ".dat"

                            try:

                                content = self.read_dat_file(file)


                            except IOError:

                                print("WARNING: File '" + file + "' not found. Using value 1e-15 for opacity in this regime.")
                                exist = 0

                        for i in range(len(self.rt_nu)):

                            if self.rt_nu[i] < numin_list[n]:
                                continue

                            elif numin_list[n] <= self.rt_nu[i] < numax_list[n]:

                                if exist == 1:

                                    index = round(self.rt_nu[i] * 100) - int(numin_list[n] * 100)
                                    opac_array_temp.append(float(content[index]))

                                elif exist == 0:

                                    opac_array_temp.append(1e-15)

                            elif self.rt_nu[i] >= numax_list[n]:
                                break

                    # reverse array
                    opac_array_temp.reverse()
                    opac_array.extend(opac_array_temp)

            try:  # create directory if necessary
                os.makedirs(param.resampling_path)
            except OSError:
                if not os.path.isdir(param.resampling_path):
                    raise

            if 'h2h2o_' in self.species_path[m]:

                # save to hdf5
                with h5py.File(param.resampling_path + "H2O_opac_sampling.h5", "w") as f:

                    f.create_dataset("pressures", data=press_list)
                    f.create_dataset("temperatures", data=temp_list)
                    f.create_dataset("wavelengths", data=self.rt_lamda)
                    f.create_dataset("opacities", data=opac_array)

            else:

                # save to hdf5
                with h5py.File(param.resampling_path + self.species_name[m] + "_opac_sampling.h5", "w") as f:

                    f.create_dataset("pressures", data=press_list)
                    f.create_dataset("temperatures", data=temp_list)
                    f.create_dataset("wavelengths", data=self.rt_lamda)
                    f.create_dataset("opacities", data=opac_array)

            print("\nSuccessfully completed -->", self.species_name[m], "<-- !\n---------------------")



    @staticmethod
    def success():
        print("\nOpacity building -- DONE!")
