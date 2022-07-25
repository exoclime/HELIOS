# ==============================================================================
# Mini module to read parameter file
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

import argparse


class Param(object):
    """ class to read in the input parameters """

    def __init__(self):

        self.building = None
        self.format = None
        self.heliosk_format = None
        self.individual_species_file_path = None
        self.grid_format = None
        self.resolution = None
        self.grid_limits = None
        self.grid_file_path = None
        self.n_gauss = None
        self.mixing = None
        self.individual_calc_path = None
        self.final_species_file_path = None
        self.fastchem_path = None
        self.final_path = None
        self.units = None

    def read_param_file_and_command_line(self):
        """ reads the param file """

        # set up command line options.
        parser = argparse.ArgumentParser(description=
                                         "The following are the possible command-line parameters for the ktable program")

        parser.add_argument('-parameter_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        ### 1st stage ###

        parser.add_argument('-individual_species_calculation', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # general properties
        parser.add_argument('-format', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-helios_k_output_format', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_individual_species_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # grid
        parser.add_argument('-grid_format', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-wavelength_grid', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_grid_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-number_of_gaussian_points', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # output
        parser.add_argument('-directory_with_individual_files', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        ### 2nd stage ###

        parser.add_argument('-mixed_table_production', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        # combining / weighing
        parser.add_argument('-path_to_final_species_file', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-path_to_fastchem_output', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-mixed_table_output_directory', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)
        parser.add_argument('-units_of_mixed_opacity_table', help='see documentation (https://heliosexo.readthedocs.io/en/latest/)', required=False)

        args = parser.parse_args()

        # read parameter file name. If none specified, use standard name.
        if args.parameter_file:
            param_file = args.parameter_file
        else:
            param_file = "param_ktable.dat"

        with open(param_file, "r", encoding='utf-8') as param_file:

            for line in param_file:
                column = line.split()
                if column:

                    ### 1st stage ###

                    if column[0] == "individual" and column[2] == "calculation":
                        self.building = column[4]

                    # general properties
                    elif column[0] == "format":
                        self.format = column[2]

                    elif column[0] == "HELIOS-K" and column[2] == "format":
                        self.heliosk_format = column[4]

                    elif column[0] == "path" and column[2] == "individual":
                        self.individual_species_file_path = column[6]

                    # grid
                    elif column[0] == "grid" and column[1] == "format":
                        self.grid_format = column[3]

                    elif column[2] == "wavelength" and column[3] == "grid":
                        self.resolution = float(column[5])
                        self.grid_limits = [float(column[6]), float(column[7])]

                    elif column[2] == "path" and column[4] == "grid" and column[5] == "file":
                        self.grid_file_path = column[7]

                    elif column[2] == "number" and column[4] == "Gaussian":
                        self.n_gauss = int(column[7])

                    ### output
                    elif column[0] == "directory" and column[2] == "individual":
                        self.individual_calc_path = column[5]

                    ### 2nd stage ###

                    elif column[0] == "mixed" and column[2] == "production":
                        self.mixing = column[4]

                    # interpolating / combining / weighting
                    elif column[0] == "path" and column[2] == "final" and column[3] == "species":
                        self.final_species_file_path = column[6]

                    elif column[0] == "path" and column[2] == "FastChem":
                        self.fastchem_path = column[5]

                    elif column[0] == "mixed" and column[2] == "output":
                        self.final_path = column[5]

                    elif column[0] == "units" and column[4] == "table":
                        self.units = column[6]

        # read command line options

        ### 1st stage ###

        if args.individual_species_calculation:
            self.building = args.individual_species_calculation

        # general properties
        if args.format:
            self.format = args.format

        if args.helios_k_output_format:
            self.heliosk_format = args.helios_k_output_format

        if args.path_to_individual_species_file:
            self.individual_species_file_path = args.path_to_individual_species_file

        # grid
        if args.grid_format:
            self.grid_format = args.grid_format

        if args.path_to_grid_file:
            self.grid_file_path = args.path_to_grid_file

        if args.number_of_gaussian_points:
            self.n_gauss = args.number_of_gaussian_points

        # output
        if args.directory_with_individual_files:
            self.individual_calc_path = args.directory_with_individual_files

        ### 2nd stage ###

        if args.mixed_table_production:
            self.mixing = args.mixed_table_production

        # interpolating / combining / weighing
        if args.path_to_final_species_file:
            self.final_species_file_path = args.path_to_final_species_file

        if args.path_to_fastchem_output:
            self.fastchem_path = args.path_to_fastchem_output

        if args.mixed_table_output_directory:
            self.final_path = args.mixed_table_output_directory

        if args.units_of_mixed_opacity_table:
            self.units = args.units_of_mixed_opacity_table

        # check that chosen units exist
        if self.units not in ["MKS", "CGS"]:
            raise ValueError("Chosen units for the opacity table unknown. Please double-check entry in the parameter file.")

if __name__ == "__main__":
    print("This is a mini module to read in the k-generator parameters."
          "There used to be many parameters, but now there are only a few left. This is a good sign...or is it not?")
