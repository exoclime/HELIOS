# ==============================================================================
# Mini module to read parameter file
# Copyright (C) 2018 Matej Malik
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


class Param(object):
    """ class to read in the input parameters """

    def __init__(self):
        self.format = None
        self.building = None
        self.heliosk_path = None
        self.resampling_path = None
        self.sampling_param_path = None
        self.resolution = None
        self.special_limits = None
        self.fastchem_path = None
        self.ele_abund = None
        self.species_path = None
        self.final_path = None
        self.special_abundance = None

    @staticmethod
    def __read_yes_no__(var):
        """ transforms yes to 1 and no to zero """
        if var == "yes":
            value = npy.int32(1)
        elif var == "no":
            value = npy.int32(0)
        else:
            print("\nWARNING: Weird value found in input file. "
                  "\nCheck that all (yes/no) parameters do have \"yes\" or \"no\" as value. "
                  "\nThis input has the form", var,
                  "\nAborting...")
            raise SystemExit()
        return value

    def read_param_file(self):
        """ reads the param file """
        try:
            with open("param_ktable.dat", "r", encoding='utf-8') as param_file:

                for line in param_file:
                    column = line.split()
                    if column:

                        # OPACITY FORMAT
                        if column[0] == "format":
                            self.format = column[2]
                        elif column[0] == "individual":
                            self.building = self.__read_yes_no__(column[4])
                        elif column[0] == "path" and column[2] == "HELIOS-K":
                            self.heliosk_path = column[5]
                        elif column[0] == "path" and column[2] == "sampling" and column[3] == "output":
                            self.resampling_path = column[5]
                        elif column[0] == "path" and column[2] == "sampling" and column[3] == "param":
                            self.sampling_param_path = column[6]
                        elif column[0] == "sampling" and column[1] == "wavelength":
                            self.resolution = float(column[4])
                            try:
                                self.special_limits = [float(column[5]), float(column[6])]
                            except ValueError:
                                pass


                        # COMBINING / WEIGHTING
                        elif column[0] == "path" and column[2] == "FastChem":
                            self.fastchem_path = column[5]
                        elif column[0] == "path" and column[2] == "condensation":
                            self.ele_abund = column[5]
                        elif column[0] == "path" and column[2] == "species":
                            self.species_path = column[5]
                        elif column[0] == "path" and column[2] == "final":
                            self.final_path = column[7]

                        # EXPERIMENTAL
                        elif column[0] == "special" and column[1] == "abundance":
                            self.special_abundance = column[3]


        except IOError:
            print("ABORT - Param file not found!")
            raise SystemExit()


if __name__ == "__main__":
    print("This is a mini module to read in the k-generator parameters. "
          "There used to be many parameters, but now there are only a few left. This is a good sign...or is it?")
