# ==============================================================================
# Module for reading in data
# Copyright (C) 2018 - 2022 Matej Malik
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

import os
import datetime
import pkg_resources
import h5py
import numpy as npy
from scipy import interpolate
import argparse
from helios import phys_const as pc
from helios import planet_database as pd
from helios import species_database as sdb
from helios import host_functions as hsfunc

KERNEL_PATH = pkg_resources.resource_filename(__name__, "kernels.cu")

class Species(object):
    """ class that sets properties of an atmospheric species """

    def __init__(self):

        self.name = None
        self.absorbing = None
        self.scattering = None
        self.source_vmr = None
        self.fc_name = None
        self.weight = None
        self.vmr_layer = []
        self.vmr_interface = []
        self.vmr_pretab = []
        self.opacity_pretab = []
        self.scat_cross_sect_layer = []
        self.scat_cross_sect_interface = []


class Read(object):
    """ class that reads in parameters, which are to be used in the HELIOS code"""

    def __init__(self):
        self.param_file = None
        self.output_path = None
        self.ktable_path = None
        self.ind_mol_opac_path = None
        self.temp_path = None
        self.stellar_model = None
        self.stellar_path = None
        self.stellar_data_set = None
        self.entr_kappa_path = None
        self.temp_format = None
        self.temp_pressure_unit = None
        self.albedo_file = None
        self.albedo_file_header_lines = None
        self.albedo_file_wavelength_name = None
        self.albedo_file_wavelength_unit = None
        self.albedo_file_surface_name = None
        self.surface_type = None
        self.input_surf_albedo = None
        self.species_file = None
        self.vertical_vmr_file = None
        self.vertical_vmr_file_header_lines = None
        self.vertical_vmr_file_press_name = None
        self.vertical_vmr_file_press_units = None
        self.opacity_path = None
        self.fastchem_path = None
        self.force_eq_chem = None
        self.fastchem_data = None
        self.fastchem_data_low = None
        self.fastchem_data_high = None
        self.fastchem_temp = None
        self.fastchem_press = None
        self.fastchem_n_t = None
        self.fastchem_n_p = None

    @staticmethod
    def delete_duplicates(long_list):
        """ deletes all duplicates in a list and returns new list """
        short_list = []
        for item in long_list:
            if item not in short_list:
                short_list.append(item)
        return short_list

    @staticmethod
    def __read_yes_no__(var):
        """ transforms yes to 1 and no to zero """
        if var == "yes":
            value = npy.int32(1)
        elif var == "no":
            value = npy.int32(0)
        elif var == "special":
            value = npy.int32(2)
        else:
            print("\nWARNING: Weird value found in input file. "
                  "\nCheck that all (yes/no) parameters do have \"yes\" or \"no\" as value. "
                  "\nThis input has the form", var,
                  "\nAborting...")
            raise SystemExit()
        return value

    @staticmethod
    def set_realtime_plotting(var):
        """ sets the realtime plotting parameters """

        if var == "yes":
            real_plot = npy.int32(1)
            n_plot = npy.int32(10)
        elif var == "no":
            real_plot = npy.int32(0)
            n_plot = npy.int32(10)
        else:
            try:
                if float(var) >= 1:
                    real_plot = npy.int32(1)
                    n_plot = npy.int32(var)
                else:
                    real_plot = npy.int32(0)
                    n_plot = npy.int32(10)
            except:
                print("\nInvalid choice for realtime plotting. Aborting...")
                raise SystemExit()
        return real_plot, n_plot

    @staticmethod
    def read_physical_timestep(input):
        """ sets the realtime plotting parameters """

        if input == "no":
            output = npy.float64(0)

        else:
            output = npy.float64(input)

            if output < 0:
                raise ValueError("ERROR: Timestep must be larger than zero. Please double-check your input.")

        return output

    @staticmethod
    def set_precision(quant):
        """ sets the correct precision for floating point numbers """

        if quant.prec == "single":
            quant.fl_prec = npy.float32
            quant.nr_bytes = 4
        elif quant.prec == "double":
            quant.fl_prec = npy.float64
            quant.nr_bytes = 8
        else:
            print("\nInvalid choice of precision. Aborting...")
            raise SystemExit()

    @staticmethod
    def set_prec_in_cudafile(quant):

        yes_to_change = 0

        with open(KERNEL_PATH, "r") as cudafile:

            contents = cudafile.readlines()

            if quant.prec == "single":

                if "/***\n" in contents:

                    contents.remove("/***\n")
                    contents.remove("***/\n")
                    yes_to_change = 1
                    print("\nRewriting Cuda-sourcefile for single precision.")
                    print("Restarting program...\n")

            if quant.prec == "double":

                if "/***\n" not in contents:

                    ind = contents.index("#define USE_SINGLE\n")
                    contents.insert(ind, "/***\n")
                    contents.insert(ind+2, "***/\n")
                    yes_to_change = 1
                    print("\nRewriting Cuda-sourcefile for double precision.")
                    print("Restarting program...\n")

        if yes_to_change == 1:

            os.rename(KERNEL_PATH, "./backup/kernels_backup/kernels.cu.backup.{:.0f}".format(datetime.datetime.now().timestamp()))

            with open(KERNEL_PATH, "w") as cudafile:
                contents = "".join(contents)
                cudafile.write(contents)

            os.system("python3 helios.py")  # recompile and restart program
            raise SystemExit()  # prevent old program from resuming at the end

    def read_param_file_and_command_line(self, param_file, quant, cloud):
        """ reads the input file and command line options """

        # setting up command line options
        parser = argparse.ArgumentParser(description=
                                         "The following are the possible command-line parameters for HELIOS")

        parser.add_argument('-parameter_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        ######################
        ### basic settings ###
        ######################

        # general
        parser.add_argument('-name', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-output_directory', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-realtime_plotting', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-planet_type', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # grid
        parser.add_argument('-toa_pressure', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-boa_pressure', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # iteration
        parser.add_argument('-run_type', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_temperature_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # radiation
        parser.add_argument('-scattering', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-direct_irradiation_beam', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-f_factor', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-stellar_zenith_angle', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-internal_temperature', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-surface_albedo', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_albedo_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-surface_name', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-use_f_approximation_formula', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # opacity mixing
        parser.add_argument('-opacity_mixing', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_opacity_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_species_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-file_with_vertical_mixing_ratios', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-directory_with_fastchem_files', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-directory_with_opacity_files', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # convective adjustment
        parser.add_argument('-convective_adjustment', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-kappa_value', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-kappa_file_path', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # stellar and planetary parameters
        parser.add_argument('-stellar_spectral_model', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_stellar_spectrum_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-dataset_in_stellar_spectrum_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-planet', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-surface_gravity', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-orbital_distance', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-radius_planet', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-radius_star', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-temperature_star', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # clouds
        parser.add_argument('-number_of_cloud_decks', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_mie_files', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-aerosol_radius_mode', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-aerosol_radius_geometric_std_dev', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-cloud_mixing_ratio', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_file_with_cloud_data', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-aerosol_name', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-cloud_bottom_pressure', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-cloud_bottom_mixing_ratio', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-cloud_to_gas_scale_height_ratio', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # photochemical kinetics coupling
        parser.add_argument('-coupling_mode', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-full_output_each_iteration_step', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-force_eq_chem_for_first_iteration', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-coupling_speed_up', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-coupling_iteration_step', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        #########################
        ### advanced settings ###
        #########################

        parser.add_argument('-debugging_feedback', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-precision', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-number_of_layers', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-isothermal_layers', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-adaptive_interval', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-tp_profile_smoothing', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-improved_two_stream_correction', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-i2s_transition_point', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-asymmetry_factor_g_0', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-diffusivity_factor', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-second_eddington_coefficient', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-geometric_zenith_angle_correction', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-flux_calculation_method', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-k_coefficients_mixing_method', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-energy_budget_correction', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-convective_damping_parameter', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-maximum_number_of_iterations', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-radiative_equilibrium_criterion', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-number_of_prerun_timesteps', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-physical_timestep', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-runtime_limit', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-start_from_provided_tp_profile', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-include_additional_heating', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_heating_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-write_tp_profile_during_run', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-convergence_criterion', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        args = parser.parse_args()

        # read parameter file name. If none specified, use standard name.
        if args.parameter_file:
            self.param_file = args.parameter_file
        else:
            self.param_file = param_file

        # first, reading parameter file
        with open(self.param_file, "r", encoding='utf-8') as pfile:

            for line in pfile:
                column = line.split()
                if column:

                    ######################
                    ### basic settings ###
                    ######################

                    # general
                    if column[0] == "name":
                        quant.name = column[2]

                    elif column[0] == "output" and column[1] == "directory":
                        self.output_path = column[3]

                    elif column[0] == "realtime":
                        quant.realtime_plot, quant.n_plot = self.set_realtime_plotting(column[3])

                    elif column[0] == "planet" and column[1] == "type":
                        quant.planet_type = column[3]

                    # grid
                    elif column[0] == "TOA" and column[1] == "pressure":
                        quant.p_toa = npy.float64(column[5])

                    elif column[0] == "BOA" and column[1] == "pressure":
                        quant.p_boa = npy.float64(column[5])

                    # iteration
                    elif column[0] == "run" and column[1] == "type":
                        quant.run_type = column[3]

                    elif column[2] == "path" and column[4] == "temperature":
                        self.temp_path = column[7]

                    elif column[2] == "temperature" and column[4] == "format":
                        self.temp_format = column[6]
                        self.temp_pressure_unit = column[7]

                    # radiation
                    elif column[0] == "scattering":
                        quant.scat = self.__read_yes_no__(column[2])

                    elif column[0] == "direct" and column[2] == "beam":
                        quant.dir_beam = self.__read_yes_no__(column[4])

                    elif column[2] == "f" and column[3] == "factor":
                        quant.f_factor = npy.float64(column[5])

                    elif column[2] == "stellar" and column[3] == "zenith":
                        zenith_angle = npy.float64(column[7])

                    elif column[0] == "internal" and column[1] == "temperature":
                        quant.T_intern = npy.float64(column[4])

                    elif column[0] == "surface" and column[1] == "albedo":
                        self.input_surf_albedo = column[3]

                    elif column[2] == "path" and column[4] == "albedo":
                        self.albedo_file = column[7]

                    elif column[2] == "albedo" and column[4] == "format":
                        self.albedo_file_header_lines = int(column[6])
                        self.albedo_file_wavelength_name = column[7]
                        self.albedo_file_wavelength_unit = column[8]

                    elif column[2] == "surface" and column[3] == "name":
                        self.albedo_file_surface_name = column[5]

                    elif column[3] == "use" and column[4] == "f" and column[5] == "approximation":
                        quant.approx_f = self.__read_yes_no__(column[8])

                    # opacity mixing
                    elif column[0] == "opacity" and column[1] == "mixing":
                        quant.opacity_mixing = column[3]

                    elif column[2] == "path" and column[4] == "opacity" and column[5] == "file":
                        self.ktable_path = column[7]

                    elif column[2] == "path" and column[4] == "species" and column[5] == "file":
                        self.species_file = column[7]

                    elif column[2] == "file" and column[4] == "vertical":
                        self.vertical_vmr_file = column[8]

                    elif column[2] == "vertical" and column[5] == "format":
                        self.vertical_vmr_file_header_lines = int(column[7])
                        self.vertical_vmr_file_press_name = column[8]
                        self.vertical_vmr_file_press_units = column[9]

                    elif column[2] == "directory" and column[4] == "FastChem":
                        self.fastchem_path = column[7]

                    elif column[2] == "directory" and column[4] == "opacity":
                        self.opacity_path = column[7]

                    # convective adjustment
                    elif column[0] == "convective" and column[1] == "adjustment":
                        quant.convection = self.__read_yes_no__(column[3])

                    elif column[0] == "kappa" and column[1] == "value":
                        quant.input_kappa_value = column[3]

                    elif column[2] == "kappa" and column[4] == "path":
                        self.entr_kappa_path = column[6]

                    # stellar and planetary parameters
                    elif column[0] == "stellar" and column[2] == "model":
                        self.stellar_model = column[4]

                    elif column[2] == "path" and column[4] == "stellar":
                        self.stellar_path = column[8]

                    elif column[2] == "dataset" and column[4] == "stellar":
                        self.stellar_data_set = column[8]

                    elif column[0] == "planet" and column[1] == "=":
                        quant.planet = column[2]

                    elif column[2] == "surface" and column[3] == "gravity":
                        quant.g = npy.float64(column[7])

                    elif column[2] == "orbital" and column[3] == "distance":
                        quant.a = npy.float64(column[6])

                    elif column[2] == "radius" and column[3] == "planet":
                        quant.R_planet = npy.float64(column[6])

                    elif column[2] == "radius" and column[3] == "star":
                        quant.R_star = npy.float64(column[6])

                    elif column[2] == "temperature" and column[3] == "star":
                        quant.T_star = npy.float64(column[6])

                    # clouds
                    elif column[0] == "number" and column[3] == "decks":
                        cloud.nr_cloud_decks = npy.int32(column[5])

                    elif column[0] == "path" and column[2] == "Mie":
                        cloud.mie_path = []
                        for i in range(cloud.nr_cloud_decks):
                            cloud.mie_path.append(column[5 + i])

                    elif column[0] == "aerosol" and column[2] == "mode":
                        cloud.cloud_r_mode = []
                        for i in range(cloud.nr_cloud_decks):
                            cloud.cloud_r_mode.append(npy.float64(column[5 + i]))

                    elif column[0] == "aerosol" and column[3] == "std":
                        cloud.cloud_r_std_dev = []
                        for i in range(cloud.nr_cloud_decks):
                            cloud.cloud_r_std_dev.append(npy.float64(column[6 + i]))

                    elif column[0] == "cloud" and column[1] == "mixing":
                        cloud.cloud_mixing_ratio_setting = column[4]

                    elif column[2] == "path" and column[6] == "cloud" and column[7] == "data":
                        cloud.cloud_vmr_file = column[9]

                    elif column[2] == "cloud" and column[3] == "file" and column[4] == "format":
                        cloud.cloud_vmr_file_header_lines = int(column[6])
                        cloud.cloud_file_press_name = column[7]
                        cloud.cloud_file_press_units = column[8]

                    elif column[2] == "aerosol" and column[3] == "name":
                        cloud.cloud_file_species_name = []
                        if cloud.cloud_mixing_ratio_setting == "file":
                            for i in range(cloud.nr_cloud_decks):
                                cloud.cloud_file_species_name.append(column[5 + i])

                    elif column[2] == "cloud" and column[3] == "bottom" and column[4] == "pressure":
                        cloud.p_cloud_bot = []
                        if cloud.cloud_mixing_ratio_setting == "manual":
                            for i in range(cloud.nr_cloud_decks):
                                cloud.p_cloud_bot.append(npy.float64(column[8 + i]))

                    elif column[2] == "cloud" and column[3] == "bottom" and column[4] == "mixing":
                        cloud.f_cloud_bot = []
                        if cloud.cloud_mixing_ratio_setting == "manual":
                            for i in range(cloud.nr_cloud_decks):
                                cloud.f_cloud_bot.append(npy.float64(column[7 + i]))

                    elif column[2] == "cloud" and column[7] == "ratio":
                        cloud.cloud_to_gas_scale_height = []
                        if cloud.cloud_mixing_ratio_setting == "manual":
                            for i in range(cloud.nr_cloud_decks):
                                cloud.cloud_to_gas_scale_height.append(npy.float64(column[9 + i]))

                    # photochemical kinetics coupling
                    elif column[0] == "coupling" and column[1] == "mode":
                        quant.coupling = self.__read_yes_no__(column[3])

                    elif column[2] == "full" and column[3] == "output":
                        quant.coupling_full_output = self.__read_yes_no__(column[8])

                    elif column[2] == "force" and column[3] == "eq" and column[4] == "chem":
                        self.force_eq_chem = column[9]

                    elif column[2] == "coupling" and column[3] == "speed":
                        quant.coupling_speed_up = self.__read_yes_no__(column[6])

                    elif column[2] == "coupling" and column[4] == "step":
                        quant.coupling_iter_nr = npy.int32(column[6])

                    #########################
                    ### advanced settings ###
                    #########################

                    elif column[0] == "debugging":
                        quant.debug = self.__read_yes_no__(column[3])

                    elif column[0] == "precision":
                        quant.prec = column[2]

                    elif column[0] == "number" and column[2] == "layers":
                        quant.nlayer = column[4]

                    elif column[0] == "isothermal":
                        iso_input = column[3]

                    elif column[0] == "adaptive" and column[1] == "interval":
                        quant.adapt_interval = npy.int32(column[3])

                    elif column[0] == "TP" and column[2] == "smoothing":
                        quant.smooth = self.__read_yes_no__(column[4])

                    elif column[0] == "improved" and column[1] == "two" and column[2] == "stream":
                        quant.scat_corr = self.__read_yes_no__(column[5])

                    elif column[2] == "I2S" and column[3] == "transition":
                        quant.i2s_transition = npy.float64(column[6])

                    elif column[0] == "asymmetry":
                        quant.g_0 = npy.float64(column[4])

                    elif column[0] == "diffusivity":
                        quant.diffusivity = npy.float64(column[3])

                    elif column[0] == "second" and column[1] == "Eddington":
                        quant.epsi2 = npy.float64(column[4])

                    elif column[0] == "geometric" and column[1] == "zenith":
                        zenith_correction = column[5]

                    elif column[0] == "flux" and column[2] == "method":
                        quant.flux_calc_method = column[4]

                    elif column[3] == "coefficients" and column[4] == "mixing":
                        quant.kcoeff_mixing = column[7]

                    elif column[0] == "energy" and column[2] == "correction":
                        energy_corr = column[4]

                    elif column[1] == "damping" and column[2] == "parameter":
                        quant.input_dampara = column[4]

                    elif column[0] == "plancktable" and column[1] == "dimension":
                        quant.plancktable_dim = npy.int32(column[5])
                        quant.plancktable_step = npy.int32(column[6])

                    elif column[0] == "maximum" and column[3] == "iterations":
                        quant.max_nr_iterations = npy.int32(column[5])

                    elif column[0] == "radiative" and column[2] == "criterion":
                        quant.rad_convergence_limit = npy.float64(column[4])

                    elif column[0] == "relax" and column[2] == "criterion":
                        quant.crit_relaxation_numbers = []
                        i = 5
                        while column[i] != "[two":
                            quant.crit_relaxation_numbers.append(npy.int32(float(column[i])))
                            i += 1

                    elif column[0] == "number" and column[2] == "prerun":
                        quant.foreplay = npy.int32(column[5])

                    elif column[0] == "physical" and column[1] == "timestep":
                        quant.physical_tstep = self.read_physical_timestep(column[4])

                    elif column[2] == "runtime" and column[3] == "limit":
                        quant.runtime_limit = npy.float64(column[6])

                    elif column[2] == "start" and column[5] == "TP":
                        quant.force_start_tp_from_file = self.__read_yes_no__(column[8])

                    elif column[0] == "include" and column[1] == "additional" and column[2] == "heating":
                        quant.add_heating = self.__read_yes_no__(column[4])

                    elif column[2] == "path" and column[4] == "heating":
                        quant.add_heating_path = column[7]

                    elif column[2] == "heating" and column[3] == "file" and column[4] == "format":
                        quant.add_heating_file_header_lines = int(column[6])
                        quant.add_heating_file_press_name = column[7]
                        quant.add_heating_file_press_unit = column[8]
                        quant.add_heating_file_data_name = column[9]
                        quant.add_heating_file_data_conv_factor = npy.float64(column[10])

                    elif column[0] == "coupling" and column[2] == "write" and column[3] == "TP":
                        write_tp_during_run = column[8]

                    elif column[0] == "coupling" and column[2] == "convergence":
                        quant.coupl_convergence_limit = npy.float64(column[5])

        # second, reading command-line options (note: will overwrite settings in param.dat because they are read later)

        ######################
        ### basic settings ###
        ######################

        # general
        if args.name:
            quant.name = args.name

        if args.output_directory:
            self.output_path = args.output_directory

        if args.realtime_plotting:
            quant.realtime_plot, quant.n_plot = self.set_realtime_plotting(args.realtime_plotting)

        if args.planet_type:
            quant.planet_type = args.planet_type

        if args.planet_type:
            quant.planet_type = args.planet_type

        # grid
        if args.toa_pressure:
            quant.p_toa = npy.float64(args.toa_pressure)

        if args.boa_pressure:
            quant.p_boa = npy.float64(args.boa_pressure)

        # iteration
        if args.run_type:
            quant.run_type = args.run_type

        if args.path_to_temperature_file:
            self.temp_path = args.path_to_temperature_file

        # radiation
        if args.scattering:
            quant.scat = self.__read_yes_no__(args.scattering)

        if args.direct_irradiation_beam:
            quant.dir_beam = self.__read_yes_no__(args.direct_irradiation_beam)

        if args.f_factor:
            quant.f_factor = npy.float64(args.f_factor)

        if args.stellar_zenith_angle:
            zenith_angle = npy.float64(args.stellar_zenith_angle)

        if args.internal_temperature:
            quant.T_intern = npy.float64(args.internal_temperature)

        if args.surface_albedo:
            self.input_surf_albedo = args.surface_albedo

        if args.path_to_albedo_file:
            self.albedo_file = args.path_to_albedo_file

        if args.surface_name:
            self.albedo_file_surface_name = args.surface_name

        if args.use_f_approximation_formula:
            quant.approx_f = self.__read_yes_no__(args.use_f_approximation_formula)

        # opacity mixing
        if args.opacity_mixing:
            quant.opacity_mixing = args.opacity_mixing

        if args.path_to_opacity_file:
            self.ktable_path = args.path_to_opacity_file

        if args.path_to_species_file:
            self.species_file = args.path_to_species_file

        if args.file_with_vertical_mixing_ratios:
            self.vertical_vmr_file = args.file_with_vertical_mixing_ratios

        if args.directory_with_fastchem_files:
            self.fastchem_path = args.directory_with_fastchem_files

        if args.directory_with_opacity_files:
            self.opacity_path = args.directory_with_opacity_files

        # convective adjustment
        if args.convective_adjustment:
            quant.convection = self.__read_yes_no__(args.convective_adjustment)

        if args.kappa_value:
            quant.input_kappa_value = args.kappa_value

        if args.kappa_file_path:
            self.entr_kappa_path = args.kappa_file_path

        # stellar and planetary parameters
        if args.stellar_spectral_model:
            self.stellar_model = args.stellar_spectral_model

        if args.path_to_stellar_spectrum_file:
            self.stellar_path = args.path_to_stellar_spectrum_file

        if args.dataset_in_stellar_spectrum_file:
            self.stellar_data_set = args.dataset_in_stellar_spectrum_file

        if args.planet:
            quant.planet = args.planet

        if args.surface_gravity:
            quant.g = npy.float64(args.surface_gravity)

        if args.orbital_distance:
            quant.a = npy.float64(args.orbital_distance)

        if args.radius_planet:
            quant.R_planet = npy.float64(args.radius_planet)

        if args.radius_star:
            quant.R_star = npy.float64(args.radius_star)

        if args.temperature_star:
            quant.T_star = npy.float64(args.temperature_star)

        # clouds
        if args.number_of_cloud_decks:
            cloud.nr_cloud_decks = npy.int32(args.number_of_cloud_decks)

        if args.path_to_mie_files:
            cloud.mie_path = [args.path_to_mie_files]

        if args.aerosol_radius_mode:
            cloud.cloud_r_mode = [npy.float64(args.aerosol_radius_mode)]

        if args.aerosol_radius_geometric_std_dev:
            cloud.cloud_r_std_dev = [npy.float64(args.aerosol_radius_geometric_std_dev)]

        if args.cloud_mixing_ratio:
            cloud.cloud_mixing_ratio_setting = args.cloud_mixing_ratio

        if args.path_to_file_with_cloud_data:
            cloud.cloud_vmr_file = args.path_to_file_with_cloud_data

        if args.aerosol_name:
            cloud.cloud_file_species_name = [args.aerosol_name]

        if args.cloud_bottom_pressure:
            cloud.p_cloud_bot = [npy.float64(args.cloud_bottom_pressure)]

        if args.cloud_bottom_mixing_ratio:
            cloud.f_cloud_bot = [npy.float64(args.cloud_bottom_mixing_ratio)]

        if args.cloud_to_gas_scale_height_ratio:
            cloud.cloud_to_gas_scale_height = [npy.float64(args.cloud_to_gas_scale_height_ratio)]

        # photochemical kinetics coupling
        if args.coupling_mode:
            quant.coupling = self.__read_yes_no__(args.coupling_mode)

        if args.full_output_each_iteration_step:
            quant.coupling_full_output = self.__read_yes_no__(args.full_output_each_iteration_step)

        if args.force_eq_chem_for_first_iteration:
            self.force_eq_chem = args.force_eq_chem_for_first_iteration

        if args.coupling_speed_up:
            quant.coupling_speed_up = self.__read_yes_no__(args.coupling_speed_up)

        if args.coupling_iteration_step:
            quant.coupling_iter_nr = npy.int32(args.coupling_iteration_step)

        #########################
        ### advanced settings ###
        #########################

        if args.debugging_feedback:
            quant.debug = self.__read_yes_no__(args.debugging_feedback)

        if args.number_of_layers:
            quant.nlayer = args.number_of_layers

        if args.isothermal_layers:
            iso_input = args.isothermal_layers

        if args.adaptive_interval:
            quant.adapt_interval = npy.int32(args.adaptive_interval)

        if args.tp_profile_smoothing:
            quant.smooth = self.__read_yes_no__(args.tp_profile_smoothing)

        if args.improved_two_stream_correction:
            quant.scat_corr = self.__read_yes_no__(args.improved_two_stream_correction)

        if args.i2s_transition_point:
            quant.i2s_transition = npy.float64(args.i2s_transition_point)

        if args.asymmetry_factor_g_0:
            quant.g_0 = npy.float64(args.asymmetry_factor_g_0)

        if args.diffusivity_factor:
            quant.diffusivity = npy.float64(args.diffusivity_factor)

        if args.second_eddington_coefficient:
            quant.epsi2 = npy.float64(args.second_eddington_coefficient)

        if args.geometric_zenith_angle_correction:
            zenith_correction = args.geometric_zenith_angle_correction

        if args.flux_calculation_method:
            quant.flux_calc_method = args.flux_calculation_method

        if args.k_coefficients_mixing_method:
            quant.kcoeff_mixing = args.k_coefficients_mixing_method

        if args.energy_budget_correction:
            energy_corr = args.energy_budget_correction

        if args.convective_damping_parameter:
            quant.input_dampara = args.convective_damping_parameter

        if args.maximum_number_of_iterations:
            quant.max_nr_iterations = npy.int32(args.maximum_number_of_iterations)

        if args.radiative_equilibrium_criterion:
            quant.rad_convergence_limit = npy.float64(args.radiative_equilibrium_criterion)

        if args.number_of_prerun_timesteps:
            quant.foreplay = npy.int32(args.number_of_prerun_timesteps)

        if args.physical_timestep:
            quant.physical_tstep = self.read_physical_timestep(args.physical_timestep)

        if args.runtime_limit:
            quant.runtime_limit = npy.float64(args.runtime_limit)

        if args.start_from_provided_tp_profile:
            quant.force_start_tp_from_file = self.__read_yes_no__(args.start_from_provided_tp_profile)

        if args.include_additional_heating:
            quant.add_heating = self.__read_yes_no__(args.include_additional_heating)

        if args.path_to_heating_file:
            quant.add_heating_path = args.path_to_heating_file

        if args.write_tp_profile_during_run:
            write_tp_during_run = args.write_tp_profile_during_run

        if args.convergence_criterion:
            quant.coupl_convergence_limit = npy.float64(args.convergence_criterion)

        # now that both the param.dat and the command-line inputs are known,
        # we can process the inputs and configure "automatic" and input dependent parameters

        # set run type automatization
        if quant.run_type == "iterative":
            quant.singlewalk = npy.int32(0)
            quant.iso = npy.int32(0)
            quant.energy_correction = npy.int32(1)
        elif quant.run_type == "post-processing":
            quant.singlewalk = npy.int32(1)
            quant.iso = npy.int32(1)
            quant.energy_correction = npy.int32(0)

        # zenith angle conversion
        quant.dir_angle = npy.float64((180 - zenith_angle) * npy.pi / 180.0)
        quant.mu_star = npy.float64(npy.cos(quant.dir_angle))

        # activating clouds
        if cloud.nr_cloud_decks > 0:
            quant.clouds = npy.int32(1)

        elif cloud.nr_cloud_decks == 0:
            quant.clouds = npy.int32(0)
        else:
            raise IOError("\nParameter Error: Number of cloud decks must be >=0. Please correct input value.")

        # coupling self-check
        if quant.coupling == 1 and quant.opacity_mixing == "premixed":
            raise IOError("ERROR: Coupling mode cannot be set when a premixed opacity table is used.")

        # coupling full output name changes to include the iteration number
        if quant.coupling == 1 and quant.coupling_full_output == 1:
            quant.name = quant.name + "_" + str(quant.coupling_iter_nr)

        # set precision
        self.set_precision(quant)
        self.set_prec_in_cudafile(quant)

        # set number of layers and interfaces
        if quant.nlayer == "automatic":
            quant.nlayer = npy.int32(npy.ceil(10.5 * npy.log10(quant.p_boa / quant.p_toa)))
        else:
            quant.nlayer = npy.int32(quant.nlayer)

        # decide log or no-log for surface gravity
        if quant.g < 10:
            quant.g = npy.float64(10 ** quant.g)

        # process isothermal layers input
        if iso_input != "automatic":
            quant.iso = self.__read_yes_no__(iso_input)

        # get 1st Eddington coefficient from diffusivity parameter
        quant.epsi = npy.float64(1.0/quant.diffusivity)

        # set zenith angle correction
        if zenith_correction != "automatic":
            quant.geom_zenith_corr = self.__read_yes_no__(zenith_correction)
        elif zenith_correction == "automatic":
            if zenith_angle > 70:
                quant.geom_zenith_corr = npy.int32(1)
            else:
                quant.geom_zenith_corr = npy.int32(0)

        # this is really just for people like me
        if quant.flux_calc_method == "iterative":  # because I always keep writing 'iterative' instead of 'iteration'
            quant.flux_calc_method = "iteration"

        # process energy correction manual input
        if energy_corr != "automatic":
            quant.energy_correction = self.__read_yes_no__(energy_corr)

        # process TP writing during run
        if write_tp_during_run == "no":
            quant.coupl_tp_write_interval = 0
        else:
            quant.coupl_tp_write_interval = npy.int32(write_tp_during_run)

        # physical timestepping needs convective adjustment because it needs the c_p from there
        if quant.physical_tstep > 0 and quant.convection == 0:
            raise IOError("ERROR: Physical timesteppings needs convective adjustment switched on (because the c_p value that is calculated from kappa). "
                          "Please either activate or deactivate both settings.")

        # set no-atmosphere special mode (setting it last so it overwrites all previous settings)
        if quant.planet_type == "no_atmosphere":
            print("\n\tWARNING: Running 'no atmosphere' special case. "
                  "\n\tAll opacities are discarded. "
                  "\n\tAtmospheric temperatures are set to zero. "
                  "\n\tConvective adjustment is disabled. "
                  "\n\tScattering is disabled. "
                  "\n\tAtmospheric pressure set to very low. "
                  "\n\tNumber of layers set to two. ")

            quant.no_atmo_mode = npy.int32(1) # used in rad_temp_iter kernel
            quant.p_toa = 1e-3
            quant.p_boa = 2e-3
            quant.scat = npy.int32(0)
            quant.convection = npy.int32(0)
            quant.nlayer = npy.int32(2)

        # set number of interfaces
        quant.ninterface = npy.int32(quant.nlayer + 1)

        # finally, since we have been introduced by now, let's do some pleasantries
        print("\n### Welcome to HELIOS! This run has the name: " + quant.name + ". Enjoy the ride! ###")

    @staticmethod
    def read_planet_database(quant):

        print("Looking up parameters of", quant.planet, "in 'planet_database.py'.")

        try:
            quant.R_planet = quant.fl_prec(pd.planet_lib[quant.planet].R_p)
            quant.g = quant.fl_prec(pd.planet_lib[quant.planet].g_p)
            quant.a = quant.fl_prec(pd.planet_lib[quant.planet].a)
            quant.R_star = quant.fl_prec(pd.planet_lib[quant.planet].R_star)
            quant.T_star = quant.fl_prec(pd.planet_lib[quant.planet].T_star)

        except KeyError:
            print("ERROR: No such planet found! Either there is a typo in the name or the entry may not exist yet. "
                  "Please feel free to add more planets in 'planet_database.py'.")
            print("Aborting for now ...")
            raise SystemExit()

    def load_premixed_opacity_table(self, quant):
        """ loads the premixed opacity table and applies some tweaks to the opacity if necessary """

        quant.opac_k = self.read_opac_file(quant, self.ktable_path, type="premixed")

        # no atmosphere special case: if 'no_atmosphere' found in name, all opacities are discarded to correctly model the absence of an atmosphere
        if quant.no_atmo_mode == 1:
            nump = len(quant.kpress)
            nt = len(quant.ktemp)
            nx = len(quant.opac_wave)
            ny = len(quant.gauss_y)
            for t in range(nt):
                for p in range(nump):
                    for x in range(nx):
                        for y in range(ny):
                            quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * t] = 1e-30

        # uncomment this chunk to play around with opacity. for debugging purposes only -- obviously :)
        # nump = len(quant.kpress)
        # nt = len(quant.ktemp)
        # nx = len(quant.opac_wave)
        # ny = len(quant.gauss_y)
        # for t in range(nt):
        #     for p in range(nump):
        #         for x in range(nx):
        #             for y in range(ny):
        #                 #if temp 14 fixed at 800K, 22 1200K
        #                 # quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * t] = quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * 10]
        #                 #
        #                 # if press fixed at 1bar
        #                 # quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * t] = quant.opac_k[y + ny * x + ny * nx * 18 + ny * nx * nump * t]

    @staticmethod
    def read_opac_file(quant, name, type="premixed", read_grid_parameters=False):
        """ reads the opacity table file for an individual species """

        with h5py.File(name, "r") as opac_file:

            print("\nReading opacity file:", name)

            try:
                opac_k = [k for k in opac_file["kpoints"][:]]
            except KeyError:
                opac_k = [k for k in opac_file["opacities"][:]]

            if type == "premixed":
                # Rayleigh scattering cross-sections
                quant.opac_scat_cross = [c for c in opac_file["weighted Rayleigh cross-sections"][:]]

                # pre-tabulated mean molecular mass values (& convert from mu to mean mass)
                quant.opac_meanmass = [m * pc.AMU for m in opac_file["meanmolmass"][:]]

            if type == "premixed" or read_grid_parameters is True:

                # wavelength grid
                try:
                    quant.opac_wave = [x for x in opac_file["center wavelengths"][:]]
                except KeyError:
                    quant.opac_wave = [x for x in opac_file["wavelengths"][:]]
                quant.nbin = npy.int32(len(quant.opac_wave))

                # Gaussian y-points
                try:
                    quant.gauss_y = [y for y in opac_file["ypoints"][:]]
                except KeyError:
                    quant.gauss_y = [0]
                quant.ny = npy.int32(len(quant.gauss_y))

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

    def read_kappa_table_or_use_constant_kappa(self, quant):
        """ reads in entropy and kappa (for the stellar community: delad) values from ASCII table """

        if quant.convection == 1:

            # a constant kappa/delad value has been set manually
            try:
                quant.input_kappa_value = npy.float64(quant.input_kappa_value)

            except ValueError:

                pass

            if quant.input_kappa_value == str(quant.input_kappa_value):

                # kappa/delad is being read from file
                if quant.input_kappa_value == "file":

                    print("\nReading kappa/delad values from file (standard format).")

                    with open(self.entr_kappa_path, "r") as entr_file:

                        next(entr_file)
                        next(entr_file)

                        for line in entr_file:
                            column = line.split()
                            if column:
                                quant.entr_temp.append(quant.fl_prec(column[0]))
                                quant.entr_press.append(quant.fl_prec(column[1]))
                                quant.entr_kappa.append(quant.fl_prec(column[2]))
                                quant.entr_c_p.append(quant.fl_prec(column[3]))
                                try:
                                    quant.entr_entropy.append(10**quant.fl_prec(column[4]))
                                except IndexError:
                                    quant.entr_entropy.append(0)

                elif quant.input_kappa_value == "water_atmo":

                    print("\nReading kappa/delad values from file (water atmospheres format).")

                    with open(self.entr_kappa_path, "r") as entr_file:

                        next(entr_file)
                        next(entr_file)
                        next(entr_file)
                        next(entr_file)
                        next(entr_file)

                        for line in entr_file:
                            column = line.split()
                            if column:
                                quant.entr_temp.append(quant.fl_prec(column[0]))
                                quant.entr_press.append(quant.fl_prec(column[1]))
                                quant.entr_kappa.append(quant.fl_prec(column[2]))
                                quant.entr_c_p.append(quant.fl_prec(column[3]))
                                quant.entr_entropy.append(10**quant.fl_prec(column[4]))
                                quant.entr_phase_number.append(quant.fl_prec(column[7]))

                quant.entr_press = npy.sort(list(set(quant.entr_press)))
                quant.entr_temp = npy.sort(list(set(quant.entr_temp)))
                quant.entr_npress = npy.int32(len(quant.entr_press))
                quant.entr_ntemp = npy.int32(len(quant.entr_temp))

                # layer quantities will be filled (=interpolated to) later during iteration
                quant.kappa_lay = npy.zeros(quant.nlayer, quant.fl_prec)
                quant.c_p_lay = npy.zeros(quant.nlayer, quant.fl_prec)

                if quant.iso == 0:
                    quant.kappa_int = npy.zeros(quant.ninterface, quant.fl_prec)

            else:

                quant.kappa_lay = npy.ones(quant.nlayer, quant.fl_prec) * npy.float64(quant.input_kappa_value)

                c_p_value = pc.R_UNIV / float(quant.input_kappa_value)

                quant.c_p_lay = npy.ones(quant.nlayer, quant.fl_prec) * c_p_value

                if quant.iso == 0:
                    quant.kappa_int = npy.ones(quant.ninterface, quant.fl_prec) * float(quant.input_kappa_value)


        # no convection -- need to prefill c_p_lay with something anyway
        else:
            quant.c_p_lay = npy.zeros(quant.nlayer, quant.fl_prec)
            quant.kappa_lay = npy.zeros(quant.nlayer, quant.fl_prec)
            if quant.iso == 0:
                quant.kappa_int = npy.zeros(quant.ninterface, quant.fl_prec)

    def read_star(self, quant):
        """ reads the correct stellar spectrum from the corresponding file """

        if self.stellar_model == "file":

            try:

                with h5py.File(self.stellar_path, "r") as starfile:
                    quant.starflux = [f for f in starfile[self.stellar_data_set][:]]

                quant.real_star = npy.int32(1)
                print("\nReading", self.stellar_path + self.stellar_data_set, "as spectral model of the host star.")

            except KeyError:

                print("\nThere is no such stellar spectrum found. Please check file path and data set.")
                inp = None
                while inp != "yes" and inp != "no":
                    inp = input("\n\tProceed with blackbody flux? (yes/no) \n\n\t")
                    if inp == "no":
                        print("\nAborting...")
                        raise SystemExit()
                    elif inp == "yes":
                        self.stellar_model = "blackbody"
                        self.read_star(quant)
                    else:
                        print("\nInvalid input. Try again with \"yes\" or \"no\".")

            # test that stellar spectrum is on the same wavelength grid as the opacities
            if len(quant.starflux) != quant.nbin:

                print("length wavelength grid of star:", len(quant.starflux))
                print("length wavelength grid of opacities:", quant.nbin)

                raise OverflowError("Stellar spectrum and opacity files have different lengths. Please double-check your input files.")

        elif self.stellar_model == "blackbody":
            quant.starflux = npy.zeros(quant.nbin, quant.fl_prec)
            print("\nUsing blackbody flux for the stellar irradiation.")

        else:
            raise IOError("Unknown Stellar model. Please check your input.")

    def read_or_fill_surf_albedo_array(self, quant):
        """ reads the albedo data from a text file """

        if self.input_surf_albedo == "file":

            albedo_file = npy.genfromtxt(self.albedo_file, names=True, dtype=None, skip_header=self.albedo_file_header_lines)

            lamda_orig = albedo_file[self.albedo_file_wavelength_name]

            if self.albedo_file_wavelength_unit == "micron":

                lamda_orig *= 1e-4

            elif self.albedo_file_wavelength_unit == "m":

                lamda_orig *= 1e2

            albedo_orig = albedo_file[self.albedo_file_surface_name]

            # convert to Helios wavelength grid
            quant.surf_albedo = interpolate.interp1d(lamda_orig, albedo_orig, bounds_error=False, fill_value=(albedo_orig[0], albedo_orig[-1]))(quant.opac_wave)
        else:
            self.input_surf_albedo = quant.fl_prec(self.input_surf_albedo)
            self.input_surf_albedo = max(1e-8, min(0.999, self.input_surf_albedo))
            # everything above 0.999 albedo is not physical. fullstop. lower boundary is for matrix method to work.

            quant.surf_albedo = npy.ones(quant.nbin) * self.input_surf_albedo

    @staticmethod
    def interpolate_to_own_press(old_press, old_array, new_press):

        new_array = interpolate.interp1d(npy.log10(old_press), old_array, bounds_error=False,
                                         fill_value=(old_array[-1], old_array[0]))(npy.log10(new_press))

        return new_array

    def read_temperature_file(self, quant):
        """ reads the temperatures from a file """

        file_temp = []
        file_press = []

        if self.temp_format == 'helios':
            try:
                with open(self.temp_path, "r") as temp_file:
                    next(temp_file)
                    next(temp_file)
                    for line in temp_file:
                        column = line.split()
                        file_temp.append(quant.fl_prec(column[1]))
                        file_press.append(quant.fl_prec(column[2]))

            except IOError:
                print("ABORT - TP file not found!")
                raise SystemExit()

        elif self.temp_format == 'TP' or 'PT':
            try:
                with open(self.temp_path, "r") as temp_file:
                    for line in temp_file:
                        column = line.split()
                        try:
                            float(column[0])
                        except ValueError:
                            continue
                        if self.temp_format == 'TP':
                            file_temp.append(quant.fl_prec(column[0]))
                            file_press.append(quant.fl_prec(column[1]))
                        elif self.temp_format == 'PT':
                            file_press.append(quant.fl_prec(column[0]))
                            file_temp.append(quant.fl_prec(column[1]))
            except IOError:
                print("ABORT - TP file not found!")
                raise SystemExit()

            if self.temp_pressure_unit == 'bar':
                file_press = [p * 1e6 for p in file_press]

        else:
            print("Wrong format for TP-file. Aborting...")
            raise SystemExit()

        new_press = [quant.p_int[0]] + quant.p_lay

        quant.T_restart = self.interpolate_to_own_press(file_press, file_temp, new_press)

    def read_species_file(self, quant):
        with open(self.species_file) as sfile:

            next(sfile)

            for line in sfile:

                column = line.split()

                if len(column) > 0:

                    species = Species()

                    species.name = column[0]

                    if species.name != "H-":
                        species.absorbing = column[1]
                        species.scattering = column[2]
                        species.source_for_vmr = column[3]

                        quant.species_list.append(species)

                    elif species.name == "H-":

                        species.name = "H-_bf"
                        species.absorbing = column[1]
                        species.scattering = column[2]
                        species.source_for_vmr = column[3]

                        quant.species_list.append(species)

                        species2 = Species()
                        species2.name = "H-_ff"
                        species2.absorbing = column[1]
                        species2.scattering = column[2]
                        species2.source_for_vmr = column[3]

                        quant.species_list.append(species2)

        # if selected, forcing equilibrium chemistry for the first iteration step (i.e., converts vertical setting into FastChem setting)
        if quant.coupling == 1 and self.force_eq_chem == "yes":

            if quant.coupling_iter_nr == 0:

                for s in range(len(quant.species_list)):

                    if quant.species_list[s].source_for_vmr == "file":
                        quant.species_list[s].source_for_vmr = "FastChem"

        # we need an absorbing species as first entry, so let's reshuffle if necessary.
        # (This is important! It is also used to set the correlated-k procedure instead of Random Overlap for 1st species)
        for s in range(len(quant.species_list)):

            if quant.species_list[s].absorbing == 'yes':
                quant.species_list.insert(0, quant.species_list[s])
                quant.species_list.pop(s + 1)
                break

            if s == len(quant.species_list) - 1:
                raise IOError("Oops! At least one species needs to be absorbing. Please double-check your included species file. "
                              "\nAborting ... ")

        # obtain additional info from the species data base
        for s in range(len(quant.species_list)):

            for key in sdb.species_lib:

                if quant.species_list[s].name == sdb.species_lib[key].name:
                    quant.species_list[s].weight = sdb.species_lib[key].weight
                    quant.species_list[s].fc_name = sdb.species_lib[key].fc_name

        # check that each species was found in the data base
        for s in range(len(quant.species_list)):

            if quant.species_list[s].weight is None:
                raise IOError("Oops! Species '" + quant.species_list[s].name + "' was not found in the species data base. "
                                                                              "Please check that the name is spelled correctly. "
                                                                              "If so, add the relevant information to the file 'species_database.py' and try again. Aborting ..."
                              )

            if (quant.species_list[s].fc_name is None) and (quant.species_list[s].source_for_vmr == "FastChem"):
                raise IOError("Oops! FastChem name for species " + quant.species_list[s].name + "unknown."
                                                                                               "Please check that the species name is spelled correctly."
                                                                                               "If so, add the relevant information to the file 'species_database.py' and try again. Aborting ..."
                              )

    def load_fastchem_data(self):
        """ read in the fastchem mixing ratios"""

        try:

            self.fastchem_data = npy.genfromtxt(self.fastchem_path + 'chem.dat',
                                                names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")
        except OSError:

            self.fastchem_data_low = npy.genfromtxt(self.fastchem_path + 'chem_low.dat',
                                                    names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

            self.fastchem_data_high = npy.genfromtxt(self.fastchem_path + 'chem_high.dat',
                                                     names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

        # temperature and pressure from the chemical grid
        if self.fastchem_data is not None:
            read_press = self.fastchem_data['Pbar']
            read_temp = self.fastchem_data['Tk']
        else:
            read_press = npy.concatenate((self.fastchem_data_low['Pbar'], self.fastchem_data_high['Pbar']))
            read_temp = npy.concatenate((self.fastchem_data_low['Tk'], self.fastchem_data_high['Tk']))

        read_press = list(set(read_press))
        read_press.sort()

        self.fastchem_temp = list(set(read_temp))
        self.fastchem_temp.sort()

        self.fastchem_press = [p * 1e6 for p in read_press]

        self.fastchem_n_t = len(self.fastchem_temp)
        self.fastchem_n_p = len(self.fastchem_press)

    def read_species_mixing_ratios(self, quant):
        """ reads the mixing ratios for all of the atmospheric species """

        # first loop is here just to read in VMR file and FastChem if demanded
        for s in range(len(quant.species_list)):

            if quant.species_list[s].source_for_vmr == "file":

                vertical_vmr = npy.genfromtxt(self.vertical_vmr_file, names=True, dtype=None, skip_header=self.vertical_vmr_file_header_lines)
                file_press_grid = vertical_vmr[self.vertical_vmr_file_press_name]

                if self.vertical_vmr_file_press_units == "Pa":

                    file_press_grid *= 10

                elif self.vertical_vmr_file_press_units == "bar":

                    file_press_grid *= 1e6

                helios_press_layer, helios_press_interface = hsfunc.calculate_pressure_levels(quant)
                break

        for s in range(len(quant.species_list)):

            if quant.species_list[s].source_for_vmr == "FastChem":

                # read FastChem file if at least one species requires FastChem abundances
                self.load_fastchem_data()
                break

        # 2nd loop is here to read the actual mixing ratio for each species from the correct source
        for s in range(len(quant.species_list)):

            # case (i): vertical VMR profile is read in from file
            if quant.species_list[s].source_for_vmr == "file":

                quant.species_list[s].vmr_layer = self.read_vertical_vmr_and_interpolate_to_helios_press_grid(vertical_vmr,
                                                                                                              quant.species_list[s],
                                                                                                              file_press_grid,
                                                                                                              helios_press_layer)
                if quant.iso == 0:
                    quant.species_list[s].vmr_interface = self.read_vertical_vmr_and_interpolate_to_helios_press_grid(vertical_vmr,
                                                                                                                      quant.species_list[s],
                                                                                                                      file_press_grid,
                                                                                                                      helios_press_interface)

                # convert to numpy arrays so they have the correct format for copying to GPU
                quant.species_list[s].vmr_layer = npy.array(quant.species_list[s].vmr_layer, quant.fl_prec)
                quant.species_list[s].vmr_interface = npy.array(quant.species_list[s].vmr_interface, quant.fl_prec)

            # case (ii): pre-tabulated VMR is read in from FastChem. Note, this VMR is still pre-tabulated format, for the TP grid of FastChem.
            # So we are really using a pre-tabulated chemistry but interpolate on-the-fly during the Helios run
            # The vertical VMR profile will be interpolated later during the HELIOS run.
            elif quant.species_list[s].source_for_vmr == "FastChem":

                quant.species_list[s].vmr_pretab = self.read_fastchem_vmr_and_interpolate_to_opacity_PT_grid(quant, quant.species_list[s])

            # case (iii): constant VMR value is read in directly from the species input file
            else:

                if "CIA" not in quant.species_list[s].name:

                    quant.species_list[s].vmr_layer = npy.array(npy.ones(quant.nlayer) * float(quant.species_list[s].source_for_vmr), quant.fl_prec)

                    if quant.iso == 0:
                        quant.species_list[s].vmr_interface = npy.array(npy.ones(quant.ninterface) * float(quant.species_list[s].source_for_vmr), quant.fl_prec)

                elif "CIA" in quant.species_list[s].name:

                    two_mixing_ratios = quant.species_list[s].source_for_vmr.split('&')

                    quant.species_list[s].vmr_layer = npy.array(npy.ones(quant.nlayer) * float(two_mixing_ratios[0]) * float(two_mixing_ratios[1]), quant.fl_prec)

                    if quant.iso == 0:
                        quant.species_list[s].vmr_interface = npy.array(npy.ones(quant.ninterface) * float(two_mixing_ratios[0]) * float(two_mixing_ratios[1]), quant.fl_prec)

    @staticmethod
    def read_vertical_vmr_and_interpolate_to_helios_press_grid(vmr_file, species, file_press, helios_press):

        if ("CIA" not in species.name) and ("H-" not in species.name) and (species.name != "He-"):

            vertical_vmr = vmr_file[species.name]

        elif ("CIA" in species.name):

            two_fc_names = species.fc_name.split('&')

            for key in sdb.species_lib:

                if two_fc_names[0] == sdb.species_lib[key].fc_name:
                    name_1 = key

                if two_fc_names[1] == sdb.species_lib[key].fc_name:
                    name_2 = key

            vertical_vmr_1 = vmr_file[name_1]
            vertical_vmr_2 = vmr_file[name_2]

            vertical_vmr = [vertical_vmr_1[i] * vertical_vmr_2[i] for i in range(len(vertical_vmr_1))]

        elif species.name == "H-_bf":

            vertical_vmr = vmr_file["H-"]

        elif species.name == "H-_ff":

            vertical_vmr_1 = vmr_file["H"]
            vertical_vmr_2 = vmr_file["e-"]

            vertical_vmr = [vertical_vmr_1[i] * vertical_vmr_2[i] for i in range(len(vertical_vmr_1))]

        elif species.name == "He-":

            vertical_vmr_1 = vmr_file["He"]
            vertical_vmr_2 = vmr_file["e-"]

            vertical_vmr = [vertical_vmr_1[i] * vertical_vmr_2[i] for i in range(len(vertical_vmr_1))]

        # get logarithm of pressure because makes more sense for interpolation
        log_file_press = [npy.log10(p) for p in file_press]
        log_helios_press = [npy.log10(p) for p in helios_press]

        helios_vmr = interpolate.interp1d(log_file_press, vertical_vmr, kind='linear', bounds_error=False,
                                          fill_value=(vertical_vmr[-1], vertical_vmr[0]))(log_helios_press)

        return helios_vmr

    def read_fastchem_vmr_and_interpolate_to_opacity_PT_grid(self, quant, species):

        # get abundances
        if ("CIA" not in species.name) and (species.name != "H-_ff") and (species.name != "He-"):

            if self.fastchem_data is not None:
                chem_vmr = self.fastchem_data[species.fc_name]
            else:
                chem_vmr = npy.concatenate((self.fastchem_data_low[species.fc_name], self.fastchem_data_high[species.fc_name]))

        elif ("CIA" in species.name) or (species.name == "H-_ff") or (species.name == "He-"):

            two_fc_names = species.fc_name.split('&')

            if self.fastchem_data is not None:
                chem_vmr_1 = self.fastchem_data[two_fc_names[0]]
                chem_vmr_2 = self.fastchem_data[two_fc_names[1]]
            else:
                chem_vmr_1 = npy.concatenate((self.fastchem_data_low[two_fc_names[0]], self.fastchem_data_high[two_fc_names[0]]))
                chem_vmr_2 = npy.concatenate((self.fastchem_data_low[two_fc_names[1]], self.fastchem_data_high[two_fc_names[1]]))

            chem_vmr = [chem_vmr_1[c] * chem_vmr_2[c] for c in range(len(chem_vmr_1))]

        helios_vmr = hsfunc.interpolate_vmr_to_opacity_grid(self, quant, chem_vmr)

        return helios_vmr

    def read_species_opacities(self, quant):

        for s in range(len(quant.species_list)):

            if s == 0:
                read_grid_params = True
            else:
                read_grid_params = False

            if quant.species_list[s].absorbing == "yes":

                try:
                    quant.species_list[s].opacity_pretab = self.read_opac_file(quant,
                                                                              self.opacity_path + quant.species_list[s].name + "_opac_ip_kdistr.h5",
                                                                              type="species",
                                                                              read_grid_parameters=read_grid_params)

                except IOError:

                    try:
                        quant.species_list[s].opacity_pretab = self.read_opac_file(quant,
                                                                                  self.opacity_path + quant.species_list[s].name + "_opac_ip.h5",
                                                                                  type="species",
                                                                                  read_grid_parameters=read_grid_params)

                    except IOError:
                        quant.species_list[s].opacity_pretab = self.read_opac_file(quant,
                                                                                  self.opacity_path + quant.species_list[s].name + "_opac_ip_sampling.h5",
                                                                                  type="species",
                                                                                  read_grid_parameters=read_grid_params)

                # convert to numpy array (necessary for GPU copying)
                quant.species_list[s].opacity_pretab = npy.array(quant.species_list[s].opacity_pretab, quant.fl_prec)

    def read_species_scat_cross_sections(self, quant):
        for s in range(len(quant.species_list)):

            if quant.species_list[s].scattering == "yes":

                if quant.species_list[s].name != "H2O":

                    with h5py.File(self.opacity_path + "scat_cross_sections.h5", "r") as scatfile:
                        quant.species_list[s].scat_cross_sect_pretab = [r for r in scatfile["rayleigh_" + quant.species_list[s].name][:]]

                    quant.species_list[s].scat_cross_sect_layer = npy.array(quant.species_list[s].scat_cross_sect_pretab * quant.nlayer, quant.fl_prec)

                    if quant.iso == 0:
                        quant.species_list[s].scat_cross_sect_interface = npy.array(quant.species_list[s].scat_cross_sect_pretab * quant.ninterface, quant.fl_prec)


if __name__ == "__main__":
    print("This module is for reading stuff. "
          "...stuff like the input file, or the opacity container, or the 'Lord of the Rings' by J. R. R. Tolkien.")
