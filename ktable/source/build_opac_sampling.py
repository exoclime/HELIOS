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
        self.press_dict = {}

    def read_param_sampling(self, param):

        with open(param.sampling_param_path, 'r')as sfile:
            next(sfile)
            for line in sfile:
                column = line.split()
                if column:
                    self.species_name.append(column[0])
                    self.species_path.append(column[1])

    def set_up_press_dict(self):

        self.press_dict['n800'] = 1e-2
        self.press_dict['n766'] = 10 ** -1.66666666
        self.press_dict['n750'] = 10 ** -1.5
        self.press_dict['n733'] = 10 ** -1.33333333
        self.press_dict['n700'] = 1e-1
        self.press_dict['n666'] = 10 ** -0.66666666
        self.press_dict['n650'] = 10 ** -0.5
        self.press_dict['n633'] = 10 ** -0.33333333
        self.press_dict['n600'] = 1e0
        self.press_dict['n566'] = 10 ** 0.33333333
        self.press_dict['n550'] = 10 ** 0.5
        self.press_dict['n533'] = 10 ** 0.66666666
        self.press_dict['n500'] = 1e1
        self.press_dict['n466'] = 10 ** 1.33333333
        self.press_dict['n450'] = 10 ** 1.5
        self.press_dict['n433'] = 10 ** 1.66666666
        self.press_dict['n400'] = 1e2
        self.press_dict['n366'] = 10 ** 2.33333333
        self.press_dict['n350'] = 10 ** 2.5
        self.press_dict['n333'] = 10 ** 2.66666666
        self.press_dict['n300'] = 1e3
        self.press_dict['n266'] = 10 ** 3.33333333
        self.press_dict['n250'] = 10 ** 3.5
        self.press_dict['n233'] = 10 ** 3.66666666
        self.press_dict['n200'] = 1e4
        self.press_dict['n166'] = 10 ** 4.33333333
        self.press_dict['n150'] = 10 ** 4.5
        self.press_dict['n133'] = 10 ** 4.66666666
        self.press_dict['n100'] = 1e5
        self.press_dict['n066'] = 10 ** 5.33333333
        self.press_dict['n050'] = 10 ** 5.5
        self.press_dict['n033'] = 10 ** 5.66666666
        self.press_dict['p000'] = 1e6
        self.press_dict['p033'] = 10 ** 6.33333333
        self.press_dict['p050'] = 10 ** 6.5
        self.press_dict['p066'] = 10 ** 6.66666666
        self.press_dict['p100'] = 1e7
        self.press_dict['p133'] = 10 ** 7.33333333
        self.press_dict['p150'] = 10 ** 7.5
        self.press_dict['p166'] = 10 ** 7.66666666
        self.press_dict['p200'] = 1e8
        self.press_dict['p233'] = 10 ** 8.33333333
        self.press_dict['p250'] = 10 ** 8.5
        self.press_dict['p266'] = 10 ** 8.66666666
        self.press_dict['p300'] = 1e9
        self.press_dict['p333'] = 10 ** 9.33333333
        self.press_dict['p350'] = 10 ** 9.5
        self.press_dict['p366'] = 10 ** 9.66666666
        self.press_dict['p400'] = 1e10

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

    def big_loop(self, param):

        for m in range(len(self.species_name)):

            if not self.species_path[m].endswith("/"):
                self.species_path[m] += "/"

            # get molecular parameter ranges
            files = os.listdir(self.species_path[m])
            file_list = [f for f in files if ("Out_" in f) and ("_cbin" not in f)]
            file_name = None


            # determine filename structure with one example file in the directory
            example_file = file_list[0]

            nr_underscore = example_file.count("_")

            indices = []

            for i in range(len(example_file)):

                if not example_file.find("_", i) in indices:
                    indices.append(example_file.find("_", i))

            indices.pop(-1)  # remove "-1" entry

            if nr_underscore > 4:

                # Aha! This is the case with species number
                start_file_name = indices[0] + 1
                end_file_name = indices[-4]

                file_name = example_file[start_file_name:end_file_name]

            start_numin = indices[-4] + 1
            end_numin = indices[-3]

            start_numax = indices[-3] + 1
            end_numax = indices[-2]

            start_temp = indices[-2] + 1
            end_temp = indices[-1]

            start_press = indices[-1] + 1
            end_press = indices[-1] + 5

            temp_list = []
            numin_list = []
            numax_list = []
            press_exp_list = []

            for f in file_list:

                numin_list.append(int(f[start_numin:end_numin]))
                numax_list.append(int(f[start_numax:end_numax]))
                temp_list.append(int(f[start_temp:end_temp]))
                press_exp_list.append(f[start_press:end_press])

            # delete duplicate entries in the lists and sort in ascending order
            temp_list = list(set(temp_list))
            temp_list.sort()

            numin_list = list(set(numin_list))
            numin_list.sort()

            numax_list = list(set(numax_list))
            numax_list.sort()

            press_list = [self.press_dict[press_exp_list[p]] for p in range(len(press_exp_list))]
            press_list = list(set(press_list))
            press_list.sort()

            # getting back the press_exp list in ascending order
            press_exp_list_ordered = []

            for p in press_list:
                for k in self.press_dict.keys():

                    if self.press_dict[k] == p:
                        press_exp_list_ordered.append(k)
                        break

            temp_min = min(temp_list)
            temp_max = max(temp_list)
            numin = min(numin_list)
            numax = max(numax_list)
            press_min = min(press_list)
            press_max = max(press_list)

            # some user feedback to check whether all is fine
            print("\n--- working on ---")
            print("molecule or atom: ", self.species_name[m])
            if not file_name is None:
                print("Files named as:", file_name)
            else:
                print("Files not specifically named.")
            print("wavenumber range: ", numin, numax)
            print("temperature range: ", temp_min, temp_max)
            print("pressure range: {:g} {:g}".format(press_min, press_max))
            print("number of wavelength bins:", len(self.rt_nu),"\n")

            opac_array = []

            # read files
            for t in range(len(temp_list)):

                for p in range(len(press_exp_list_ordered)):

                    opac_array_temp = []

                    for n in range(len(numin_list)):

                        exist = 1

                        tls.percent_counter(t, len(temp_list), p, len(press_exp_list_ordered))

                        if param.heliosk_format == "binary":

                            if file_name is None:

                                file=self.species_path[m]+"Out_{:05d}_{:05d}_{:05d}_".format(numin_list[n],numax_list[n],temp_list[t])+press_exp_list_ordered[p]+".bin"

                            else:

                                file=self.species_path[m]+"Out_{}_{:05d}_{:05d}_{:05d}_".format(file_name, numin_list[n],numax_list[n],temp_list[t])+press_exp_list_ordered[p]+".bin"

                            try:
                                content = npy.fromfile(file, npy.float32, -1, "")

                            except IOError:

                                print("WARNING: File '" + file + "' not found. Using value 1e-15 for opacity in this regime.")
                                exist = 0

                        elif param.heliosk_format == "text":

                            if file_name is None:

                                file=self.species_path[m]+"Out_{:05d}_{:05d}_{:05d}_".format(numin_list[n],numax_list[n],temp_list[t])+press_exp_list_ordered[p]+".dat"

                            else:

                                file=self.species_path[m]+"Out_{}_{:05d}_{:05d}_{:05d}_".format(file_name, numin_list[n],numax_list[n],temp_list[t])+press_exp_list_ordered[p]+".dat"

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

                                    index = round(self.rt_nu[i] * 100) - int(numin_list[n] * 100)  # WARNING: Assumes a HELIOS-K resolution of 0.01 cm^-1
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
