# ==============================================================================
# This program generates the resampled opacities from the HELIOS-K output
# Copyright (C) 2018 - 2022 Matej Malik
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
from scipy import interpolate as itp
from numpy.polynomial.legendre import leggauss as G
from source import tools as tls


class Production(object):
    """ class to produce the individual molecular opacity files """

    def __init__(self):
        self.species_name = []
        self.species_path = []
        self.lamda_hk = []
        self.lamda = []
        self.lamda_int = None
        self.delta_lamda = []
        self.y_gauss = None
        self.nu_hk = None
        self.nu = None
        self.nu_int = None
        self.press_dict = {}

    def read_individual_species_file(self, param):

        with open(param.individual_species_file_path, 'r')as ifile:
            next(ifile)
            for line in ifile:
                column = line.split()
                if column:
                    self.species_name.append(column[0])
                    self.species_path.append(column[1])

    def set_up_press_dict(self):
        """ converts the pressure info in the Helios-K file name to a pressure value. Note the SI to cgs unit change """

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
    def read_text_file(path):

        opac_list = []

        with open(path, 'r')as kfile:

            for line in kfile:

                column = line.split()
                if len(column) > 0:

                    opac_list.append(float(column[1]))

        return opac_list

    @staticmethod
    def gen_fixed_res_grid(bot_limit, top_limit, resolution):

        l_array = []

        l_point = bot_limit

        while l_point < top_limit:

            l_array.append(l_point)

            l_point *= (resolution + 1) / resolution

        return l_array

    @staticmethod
    def read_grid_file(file):

        l_array = []

        with open(file, 'r') as infile:
            for line in infile:
                column = line.split()
                l_array.append(float(column[0]))

        return l_array

    def initialize_wavelength_grid(self, param):

        if param.grid_format == 'fixed_resolution':

            bot_limit = param.grid_limits[0] * 1e-4
            top_limit = param.grid_limits[1] * 1e-4

            if param.format == 'sampling':

                self.lamda = self.gen_fixed_res_grid(bot_limit, top_limit, param.resolution)

            elif param.format == 'k-distribution':

                self.lamda_int = self.gen_fixed_res_grid(bot_limit, top_limit, param.resolution)

        elif param.grid_format == "file":

            if param.format == 'sampling':

                self.lamda = self.read_grid_file(param.grid_file_path)

            elif param.format == 'k-distribution':

                self.lamda_int = self.read_grid_file(param.grid_file_path)

        # Including the option to use the native resolution of the Helios-K calculation
        # Note that in this case the grid is constant in delta_wavenumber and not constant in R = lamda / delta_lamda.
        elif param.grid_format == "native_helios-k":

            if param.format == 'k-distribution':

                raise IOError("ERROR: The Native Helios-K resolution setting only works in combination "
                              "with the sampling method and not the k-distribution method. Please choose 'sampling' and start again.")

            elif param.format == 'sampling':

                bot_limit = 0.01
                top_limit = 41000 # consistent with water opacity range

                self.nu = npy.arange(bot_limit, top_limit+0.01, 0.01)

        if param.format == 'sampling':

            if param.grid_format != "native_helios-k":

                self.nu = [1 / l for l in self.lamda]
                self.nu.reverse()

                self.nu = [round(n, 2) for n in self.nu]
                # Comment: This rearranges the wavenumber values to match a 0.01 cm-1 resolution, which is commonly the default for Helios-K calculations.
                # Now, one could make this flexible and dependent on the actual Helios-K resolution.
                # However, since different species may have been calculated at different resolutions,
                # this would lead to inconsistencies in the final wavelength grid, since the latter needs to be the same for all species.
                # Hence, taking fixed 2 decimal places is probably a reasonably compromised approach (compromised = compromise?).

            self.lamda = [1 / n for n in self.nu]  # matching lamda values to the rearranged nu values
            self.lamda.reverse()

        elif param.format == 'k-distribution':

            for l in range(len(self.lamda_int) - 1):

                self.lamda.append((self.lamda_int[l] + self.lamda_int[l + 1]) / 2)

                self.delta_lamda.append(self.lamda_int[l + 1] - self.lamda_int[l])

            # construct Gaussian x-axis within a bin (also called y-points for some reason)
            y_gauss_original = G(param.n_gauss)[0]

            self.y_gauss = [0.5 * y + 0.5 for y in y_gauss_original]

    def big_loop(self, param):

        for m in range(len(self.species_name)):

            if not self.species_path[m].endswith("/"):
                self.species_path[m] += "/"

            # get molecular parameter ranges
            files = os.listdir(self.species_path[m])

            file_list = [f for f in files if ("Out_" in f) and ("_cbin" not in f)]

            if param.heliosk_format in ['binary', 'bin']:  # because I never remember whether to write 'bin' or 'binary'

                file_list = [f for f in file_list if ".bin" in f]

                file_ending = ".bin"

            elif param.heliosk_format in ['text', 'dat']:

                file_list = [f for f in file_list if ".dat" in f]

                file_ending = ".dat"

            # check that chosen Helios-K format is indeed correct
            if file_list == []:

                raise TypeError ("No files with the correct format found in the chosen directory. Please double-check that the Helios-K format is correct.")

            # determine filename structure with one example file in the directory
            file_name = None
            example_file = file_list[0]

            nr_underscore = example_file.count("_")

            indices = []

            for i in range(len(example_file)):

                if not example_file.find("_", i) in indices:
                    indices.append(example_file.find("_", i))

            indices.pop(-1)  # remove "-1" entry

            if nr_underscore > 4:  # Aha! This is the case with species number or other kind of file name

                start_file_name = indices[0] + 1
                end_file_name = indices[-4]

                file_name = example_file[start_file_name:end_file_name]

            # apart from the potential species name the file naming should be unique and thus all nu, T, P info is retrievable
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

            # getting back the press_exp list in sorted, ascending order
            press_exp_list_ordered = []

            for p in press_list:
                for k in self.press_dict.keys():

                    if self.press_dict[k] == p:
                        press_exp_list_ordered.append(k)
                        break

            # display some parameter feedback for the user
            print("\n--- working on ---")
            print("molecule or atom: ", self.species_name[m])
            if file_name is not None:
                print("Files named as:", file_name)
            else:
                print("Files not specifically named.")
            print("Helios-K wavenumber range: ", min(numin_list), max(numax_list), "cm-1")
            print("Helios-K temperature range: ", min(temp_list), max(temp_list), "K")
            print("Helios-K pressure range: {:g} {:g} dyne/cm2".format(min(press_list), max(press_list)))
            print("# wavelength bins/points for table:", len(self.lamda))

            opac_helios_total = []

            # read files
            for t in range(len(temp_list)):

                for p in range(len(press_exp_list_ordered)):

                    tls.percent_counter(t, len(temp_list), p, len(press_exp_list_ordered))

                    opac_hk_for_one_TP = []
                    opac_helios_for_one_TP = []

                    # read Helios-K data for one P,T point but the whole spectral range
                    for n in range(len(numin_list)):

                        if file_name is None:

                            file = self.species_path[m]+"Out_{:05d}_{:05d}_{:05d}_".format(numin_list[n],numax_list[n],temp_list[t]) + press_exp_list_ordered[p] + file_ending

                        else:

                            file = self.species_path[m]+"Out_{}_{:05d}_{:05d}_{:05d}_".format(file_name, numin_list[n],numax_list[n],temp_list[t]) + press_exp_list_ordered[p] + file_ending

                        # opening files using the aforespecified format
                        if param.heliosk_format in ['binary', 'bin']:

                            opac_from_file = npy.fromfile(file, npy.float32, -1, "")

                        elif param.heliosk_format == "text":

                            opac_from_file = self.read_text_file(file)

                        # determine Helios-K wavenumber resolution by counting opacity values in one file. done only once at the very beginning.
                        if t == 0 and p == 0 and n == 0:

                            hk_resolution = (numax_list[n] - numin_list[n]) / len(opac_from_file)
                            print("Resolution of Helios-K: {:g} \n".format(hk_resolution))

                            if param.format == 'k-distribution':

                                self.nu_hk = npy.arange(numin_list[0], numax_list[-1], hk_resolution)

                                self.lamda_hk = [1/n for n in self.nu_hk if n > 0]
                                self.lamda_hk.insert(0, 10000) # inserts 10000 cm at index 0 because nu = 0 does not have a wavelength equivalent
                                self.lamda_hk.reverse()

                        opac_hk_for_one_TP.extend(opac_from_file)

                    # calculate Helios opacity for one P,T point using the sampling method (i.e., picking opacity values point-wise)
                    if param.format == 'sampling':

                        # read out the entries that match with the chosen wavenumber grid values
                        for i in range(len(self.nu)):

                            if self.nu[i] < numin_list[0]:  # filling up opacity array if HK wavenumber grid starts later than the Helios grid

                                opac_helios_for_one_TP.append(1e-15)

                            elif numin_list[0] <= self.nu[i] < numax_list[-1]:

                                index = round((self.nu[i] - numin_list[0]) / hk_resolution)
                                # Comment: In theory the 'round' should be redundant as the nu grid matches a 0.01 resolution.
                                # However, we need to convert from 'float' to 'int' anyway, so rounding is probably the safest way to do it.

                                opac_helios_for_one_TP.append(float(opac_hk_for_one_TP[index]))

                            elif self.nu[i] >= numax_list[-1]:

                                opac_helios_for_one_TP.append(1e-15)

                        opac_helios_for_one_TP.reverse()

                    # calculate Helios opacity for one P,T point using the k-distribution method
                    elif param.format == 'k-distribution':

                        opac_hk_for_one_TP.reverse()

                        l_start = 0
                        l_end = 0

                        for x in range(len(self.lamda)):

                            lamda_hk_within_bin = []
                            opac_hk_within_bin = []

                            for l in range(l_start, len(self.lamda_hk)):

                                if self.lamda_int[x] <= self.lamda_hk[l] < self.lamda_int[x + 1]:

                                    lamda_hk_within_bin.append(self.lamda_hk[l])
                                    opac_hk_within_bin.append(max(1e-15, opac_hk_for_one_TP[l]))
                                    # Comment: Taking 1e-15 as minimum opacity to prevent issues with log10(k) in case there are vanishing opacity values.

                                    l_end = l

                                elif self.lamda_hk[l] >= self.lamda_int[x + 1]:

                                    l_start = l

                                    break

                            n_ppb = len(lamda_hk_within_bin)

                            if n_ppb > 1:

                                k_g_unsorted = npy.zeros(n_ppb, dtype=[('y_g', 'float64'), ('w_g', 'float64'), ('log k_g', 'float64')])

                                k_g_unsorted[:]['log k_g'] = npy.log10(opac_hk_within_bin[:])

                                # constructing w_g
                                k_g_unsorted[0]['w_g'] = (lamda_hk_within_bin[0] - self.lamda_int[x]) + (lamda_hk_within_bin[1] - lamda_hk_within_bin[0]) / 2

                                for i in range(1, n_ppb - 1):

                                    k_g_unsorted[i]['w_g'] = (lamda_hk_within_bin[i + 1] - lamda_hk_within_bin[i - 1]) / 2

                                k_g_unsorted[n_ppb - 1]['w_g'] = (self.lamda_int[x+1] - lamda_hk_within_bin[n_ppb - 1]) + (lamda_hk_within_bin[n_ppb - 1] - lamda_hk_within_bin[n_ppb - 2]) / 2

                                k_g_unsorted['w_g'] /= self.delta_lamda[x]

                                k_g_sorted = npy.sort(k_g_unsorted, order='log k_g')

                                # constructing new x_axis
                                k_g_sorted[0]['y_g'] = 0.5 * k_g_sorted[0]['w_g']

                                for y in range(1, n_ppb):

                                    k_g_sorted[y]['y_g'] = k_g_sorted[y - 1]['y_g'] + 0.5 * (k_g_sorted[y - 1]['w_g'] + k_g_sorted[y]['w_g'])

                                # interpolation
                                itp_func = itp.interp1d(k_g_sorted['y_g'], k_g_sorted['log k_g'],
                                                        kind="linear",
                                                        bounds_error=False,
                                                        assume_sorted=True,
                                                        fill_value=(k_g_sorted[0]['log k_g'], k_g_sorted[n_ppb - 1]['log k_g'])
                                                        )

                                opac_helios_within_bin = list(10**itp_func(self.y_gauss))

                            elif n_ppb == 1:

                                opac_helios_within_bin = list(npy.ones(len(self.y_gauss)) * opac_hk_within_bin[0])

                            else: # case that n_ppb = 0

                                # if HELIOS grid starts before HK grid or if HELIOS grid extends further than HK grid, just fill with minimum opacity value
                                if (l_start == 0) or l_end == len(self.lamda_hk) - 1:

                                    opac_helios_within_bin = list(npy.ones(len(self.y_gauss)) * 1e-15)

                                else:

                                    raise IndexError(
                                        "ERROR: k-distribution construction failed. Original wavelength grid must be finer than Helios wavelength grid. "
                                        "In each new wavelength bin there must be at least 1 opacity point of the original grid."
                                        "Please use finer original grid or coarser HELIOS grid.")

                            opac_helios_for_one_TP.extend(opac_helios_within_bin)

                        opac_helios_for_one_TP.reverse()

                    opac_helios_total.extend(opac_helios_for_one_TP)

            try:  # create directory if necessary
                os.makedirs(param.individual_calc_path)
            except OSError:
                if not os.path.isdir(param.individual_calc_path):
                    raise

            if param.format == 'sampling':

                # save to hdf5
                with h5py.File(param.individual_calc_path + self.species_name[m] + "_opac_sampling.h5", "w") as f:

                    f.create_dataset("pressures", data=press_list)
                    f.create_dataset("temperatures", data=temp_list)
                    f.create_dataset("wavelengths", data=self.lamda)
                    f.create_dataset("opacities", data=opac_helios_total)

            elif param.format == 'k-distribution':

                with h5py.File(param.individual_calc_path + self.species_name[m] + "_opac_kdistr.h5", "w") as f:

                    f.create_dataset("pressures", data=press_list)
                    f.create_dataset("temperatures", data=temp_list)
                    f.create_dataset("interface wavelengths", data=self.lamda_int)
                    f.create_dataset("center wavelengths", data=self.lamda)
                    f.create_dataset("wavelength width of bins", data=self.delta_lamda)
                    f.create_dataset("ypoints", data=self.y_gauss)
                    f.create_dataset("kpoints", data=opac_helios_total)

            print("\nSuccessfully completed -->", self.species_name[m], "<-- !\n---------------------")

    @staticmethod
    def success():
        print("\nOpacity building -- DONE!")
