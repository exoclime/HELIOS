# ==============================================================================
# Module for reading in stuff
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

import h5py
import numpy as np
import planets_and_stars as ps
import phys_const as pc


class Read(object):
    """ class that reads in parameters, which are to be used in the HELIOS code"""

    def __init__(self):
        self.ktable_path = None
        self.restart_path = None
        self.stellar_path = None

    @staticmethod
    def read_yes_no(column, nr):  # TODO: hide this method to global namespace?
        """ transforms yes to 1 and no to zero """
        if column[nr] == "yes":
            value = np.int32(1)
        elif column[nr] == "no":
            value = np.int32(0)
        else:
            print("\nWARNING: Weird value found in input file. "
                  "\nCheck that all (yes/no) parameters do have \"yes\" or \"no\" as value. "
                  "\nAborting...")
            raise SystemExit()
        return value

    def read_input(self, quant):
        """ reads the input file """
        try:
            with open("./input_param.dat", "r") as input_file:
                for line in input_file:
                    column = line.split()
                    if column:
                        if column[0] == "isothermal":
                            quant.iso = self.read_yes_no(column, 3)
                        elif column[0] == "number" and column[2] == "layers":
                            quant.nlayer = np.int32(column[4])
                            quant.ninterface = np.int32(quant.nlayer + 1)
                        elif column[0] == "TOA" and column[1] == "pressure":
                            quant.p_toa = np.float64(column[5])
                        elif column[0] == "BOA" and column[1] == "pressure":
                            quant.p_boa = np.float64(column[5])
                        elif column[0] == "pre-tabulate":
                            quant.tabu = self.read_yes_no(column, 2)
                        elif column[0] == "post-processing":
                            quant.singlewalk = self.read_yes_no(column, 3)
                        elif column[0] == "restart":
                            quant.restart = self.read_yes_no(column, 3)
                        elif column[0] == "path" and column[2] == "restart":
                            self.restart_path = column[5]
                        elif column[0] == "varying":
                            quant.varying_tstep = self.read_yes_no(column, 3)
                        elif column[0] == "timestep":
                            quant.tstep = np.float64(column[3])
                        elif column[0] == "scattering":
                            quant.scat = self.read_yes_no(column, 2)
                        elif column[0] == "exact":
                            quant.direct = self.read_yes_no(column, 3)
                            if quant.direct == 1:
                                quant.tabu = np.int32(2)
                                quant.scat = np.int32(0)
                        elif column[0] == "path" and column[2] == "opacity":
                            self.ktable_path = column[5]
                        elif column[0] == "diffusivity":
                            quant.diffusivity = np.float64(column[3])
                            quant.epsilon = np.float64(1.0/quant.diffusivity)
                        elif column[0] == "f" and column[1] == "factor":
                            quant.f_factor = np.float64(column[3])
                        elif column[0] == "internal" and column[1] == "temperature":
                            quant.T_intern = np.float64(column[4])
                        elif column[0] == "asymmetry":
                            quant.g_0 = np.float64(column[4])
                        elif column[0] == "mean" and column[1] == "molecular":
                            quant.mu = np.float64(column[5])
                            quant.meanmolmass = np.float64(quant.mu * pc.M_P)
                        elif column[0] == "planet":
                            quant.planet = column[2]
                        elif column[0] == "surface" and column[1] == "gravity":
                            quant.g = np.float64(column[5])
                        elif column[0] == "orbital" and column[1] == "distance":
                            quant.a = np.float64(column[4])
                        elif column[0] == "radius" and column[1] == "star":
                            quant.R_star = np.float64(column[4])
                        elif column[0] == "temperature" and column[1] == "star":
                            quant.T_star = np.float64(column[4])
                        elif column[0] == "model":
                            quant.model = column[2]
                        elif column[0] == "path" and column[2] == "stellar":
                            self.stellar_path = column[6]
                        elif column[0] == "name":
                            quant.name = column[2]
                        elif column[0] == "number" and column[2] == "run-in":
                            quant.foreplay = np.int32(column[5])
                        elif column[0] == "artificial":
                            quant.fake_opac = np.float64(column[4])
                        elif column[0] == "realtime":
                            quant.realtime_plot = self.read_yes_no(column, 3)

        except FileNotFoundError:
            print("ABORT - Input file not found!")
            raise SystemExit()

    def read_opac(self, quant):
        """ reads the file with the opacity table """

        try:
            with h5py.File(self.ktable_path, "r") as opac_file:

                for k in opac_file["kpoints"][:]:
                    quant.opac_k.append(k)
                for y in opac_file["ypoints"][:]:
                    quant.opac_y.append(y)
                for x in opac_file["centre wavelengths"][:]:
                    quant.opac_wave.append(x)
                for w in opac_file["wavelength width of bins"][:]:
                    quant.opac_deltawave.append(w)
                for i in opac_file["interface wavelengths"][:]:
                    quant.opac_interwave.append(i)
                for t in opac_file["temperatures"][:]:
                    quant.ktemp.append(t)
                quant.ntemp = len(quant.ktemp)
                quant.ntemp = np.int32(quant.ntemp)
                for p in opac_file["pressures"][:]:
                    quant.kpress.append(p)
                quant.npress = len(quant.kpress)
                quant.npress = np.int32(quant.npress)
                for cross in opac_file["cross_rayleigh"][:]:
                    quant.cross_scat.append(cross)
        except OSError:
            print("\nABORT - \"", self.ktable_path, "\" not found!")
            raise SystemExit()

    @staticmethod
    def read_correct_dataset(path, dataset, flux_list):  # TODO: hide this method to global namespace?

        try:
            with h5py.File(path, "r") as file:
                for f in file[dataset][:]:
                    flux_list.append(f)
        except OSError:
            print("\nABORT - \"", path, "\" not found!")
            raise SystemExit()

    def read_star(self, quant):
        """ reads the correct stellar spectrum from the corresponding file """

        if quant.model == "kurucz" or quant.model == "phoenix":

            if quant.planet in ps.planets_with_stellar_spectra:

                star_name = ""
                for c in quant.planet[:-1]:
                    if c is not "-":
                        star_name += c
                star_name = star_name.lower()
                star_name.replace(" ", "")

                self.read_correct_dataset(self.stellar_path,
                                          "/"+str(quant.nbin)+"/"+quant.model+"/"+star_name,
                                          quant.starflux)

                quant.real_star = np.int32(1)
                print("\nUsing the", quant.model, "stellar model for the host star of "+quant.planet+".")

            else:
                print("\nThere is no stellar spectrum for this star found.")
                inp = None
                while inp != "yes" and inp != "no":
                    inp = input("\n\tProceed with blackbody flux? (yes/no) \n\n\t")
                    if inp == "no":
                        print("\nAborting...")
                        raise SystemExit()
                    elif inp == "yes":
                        print("\nProceeding with blackbody flux...")
                    else:
                        print("\nInvalid input. Try again with \"yes\" or \"no\".")

        elif quant.model == 'blackbody':
            quant.starflux = np.zeros(quant.nbin, np.float64)
            print("\nUsing blackbody flux for the stellar irradiation.")
        else:
            print("\nInvalid choice for stellar spectrum. Aborting...")
            raise SystemExit()

    def read_restart_file(self, quant):
        """ reads the restart temperatures from file """

        try:
            with open(self.restart_path, "r") as restart_file:
                next(restart_file)
                next(restart_file)
                for line in restart_file:
                    column = line.split()
                    quant.T_restart.append(column[1])
        except FileNotFoundError:
            print("ABORT - restart file not found!")
            raise SystemExit()

        # check if correct length
        if len(quant.T_restart) != quant.nlayer:
            print("ABORT - restart file corrupted!")
            raise SystemExit()

if __name__ == "__main__":
    print("This module is for reading in stuff. Stuff like the input file or the opacities or ...")
