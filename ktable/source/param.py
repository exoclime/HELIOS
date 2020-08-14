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
        self.heliosk_format = None
        self.resolution = None
        self.special_limits = None
        self.fastchem_path = None
        self.cond_path = None
        self.species_path = None
        self.final_path = None
        self.units = None
        self.condensation = None

    def read_param_file(self):
        """ reads the param file """
        try:
            with open("param_ktable.dat", "r", encoding='utf-8') as param_file:

                for line in param_file:
                    column = line.split()
                    if column:

                        # 1st stage
                        if column[0] == "format":
                            self.format = column[2]
                        elif column[0] == "individual" and column[2] == "calculation":
                            self.building = column[4]
                        elif column[0] == "path" and column[2] == "HELIOS-K":
                            self.heliosk_path = column[5]
                        elif column[0] == "path" and column[2] == "sampling" and column[3] == "species":
                            self.sampling_param_path = column[6]
                        elif column[0] == "HELIOS-K" and column[2] == "format":
                            self.heliosk_format = column[4]
                        elif column[0] == "sampling" and column[1] == "wavelength":
                            self.resolution = float(column[4])
                            try:
                                self.special_limits = [float(column[5]), float(column[6])]
                            except ValueError:
                                pass

                        # 2nd stage
                        elif column[0] == "directory" and column[2] == "individual":
                            self.resampling_path = column[5]
                        elif column[0] == "path" and column[2] == "final" and column[3] == "species":
                            self.species_path = column[6]
                        elif column[0] == "path" and column[2] == "FastChem":
                            self.fastchem_path = column[5]
                        elif column[0] == "final" and column[1] == "output":
                            self.final_path = column[6]
                        elif column[0] == "units" and column[4] == "table":
                            self.units = column[6]
                            if self.units not in ["MKS", "CGS"]:
                                raise ValueError("Chosen units for the opacity table unknown. Please double-check entry in the parameter file.")

                        # experimental
                        elif column[0] == "include" and column[1] == "condensation":
                            self.condensation = column[3]
                        elif column[0] == "path" and column[2] == "condensation":
                            self.cond_path = column[5]


        except IOError:
            print("ABORT - Param file not found!")
            raise SystemExit()


if __name__ == "__main__":
    print("This is a mini module to read in the k-generator parameters. "
          "There used to be many parameters, but now there are only a few left. This is a good sign...or is it?")
