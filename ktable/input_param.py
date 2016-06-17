# ==============================================================================
# Mini module to read input parameter file
# Copyright (C) 2016 Matej Malik
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

class Read_param(object):
    """ class to read in the input parameters file """    
    
    def __init__(self):
        self.dir = None
        self.form = None
        self.n_o = None
        self.n_c = None
        self.mu = None
    
    def read(self):
        """ reads in parameters from the input file """
        
        try:
            with open("input_ktable.dat","r") as param:
                for line in param:
                        column = line.split()
                        if column:               
                            if column[0] == "HELIOS-K" and column[1] == "output":
                                self.dir = column[4]
                            elif column[0] == "output" and column[1] == "format":
                                self.form = int(column[3])
                            elif column[0] == "elemental" and column[1] == "oxygen":
                                self.n_o = float(column[4])
                            elif column[0] == "elemental" and column[1] == "carbon":
                                self.n_c = float(column[4])
                            elif column[0] == "mean" and column[2] == "weight":
                                self.mu = float(column[5])
        except(FileNotFoundError):
            print("Parameter file not found. Aborting...")
            raise SystemExit()  
    
if __name__ == "__main__":
    print("This is a mini module to read the correct HELIOS-K output format.")