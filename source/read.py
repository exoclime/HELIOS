# ==============================================================================
# Module for reading in data
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

import os
import datetime
import h5py
import numpy as npy
from scipy import interpolate
import argparse
from source import phys_const as pc


class Read(object):
    """ class that reads in parameters, which are to be used in the HELIOS code"""

    def __init__(self):
        self.ktable_path = None
        self.ind_mol_opac_path = None
        self.temp_path = None
        self.stellar_path = None
        self.entr_kappa_path = None
        self.planet_file = None
        self.temp_format = None
        self.temp_pressure_unit = None
        self.fastchem_path = None

    @staticmethod
    def delete_duplicates(long_list):
        """ delete all duplicates in a list and return new list """
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
                if float(var) > 1:
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

        with open("./source/kernels.cu", "r") as cudafile:

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

            os.rename("./source/kernels.cu", "./backup/kernels_backup/kernels.cu.backup.{:.0f}".format(datetime.datetime.now().timestamp()))

            with open("./source/kernels.cu", "w") as cudafile:
                contents = "".join(contents)
                cudafile.write(contents)

            os.system("python3 helios.py")  # recompile and restart program
            raise SystemExit()  # prevent old program from resuming at the end


    def read_param_file(self, quant, Vmod):
        """ reads the input file """
        try:
            with open("param.dat", "r", encoding='utf-8') as param_file:

                for line in param_file:
                    column = line.split()
                    if column:

                        # GENERAL
                        if column[0] == "name":
                            quant.name = column[2]
                        elif column[0] == "precision":
                            quant.prec = column[2]
                            self.set_precision(quant)
                            self.set_prec_in_cudafile(quant)
                        elif column[0] == "realtime":
                            quant.realtime_plot, quant.n_plot = self.set_realtime_plotting(column[3])

                        # GRID
                        elif column[0] == "isothermal":
                            quant.iso = self.__read_yes_no__(column[3])
                        elif column[0] == "number" and column[2] == "layers":
                            quant.nlayer = npy.int32(column[4])
                            quant.ninterface = npy.int32(quant.nlayer + 1)
                        elif column[0] == "TOA" and column[1] == "pressure":
                            quant.p_toa = quant.fl_prec(column[5])
                        elif column[0] == "BOA" and column[1] == "pressure":
                            quant.p_boa = quant.fl_prec(column[5])

                        # ITERATION
                        elif column[0] == "post-processing":
                            quant.singlewalk = self.__read_yes_no__(column[3])
                        elif column[0] == "path" and column[2] == "temperature":
                            self.temp_path = column[5]
                        elif column[0] == "temperature" and column[2] == "format":
                            self.temp_format = column[7]
                            self.temp_pressure_unit = column[8]
                        elif column[0] == "varying":
                            quant.varying_tstep = self.__read_yes_no__(column[3])
                        elif column[0] == "timestep":
                            quant.tstep = quant.fl_prec(column[3])
                        elif column[0] == "adaptive" and column[1] == "interval":
                            quant.adapt_interval = npy.int32(column[3])
                        elif column[0] == "TP-profile" and column[1] == "smoothing":
                            quant.smooth = self.__read_yes_no__(column[3])

                        # RADIATION
                        elif column[0] == "direct" and column[2] == "beam":
                            quant.dir_beam = self.__read_yes_no__(column[4])
                        elif column[0] == "scattering":
                            quant.scat = self.__read_yes_no__(column[2])
                        elif column[0] == "imp." and column[1] == "scattering":
                            quant.scat_corr = self.__read_yes_no__(column[4])
                        elif column[0] == "path" and column[2] == "opacity" and column[3] == "file":
                            self.ktable_path = column[5]
                        elif column[0] == "path" and column[2] == "molecular":
                            self.ind_mol_opac_path = column[5]
                        elif column[0] == "path" and column[2] == "FASTCHEM":
                            self.fastchem_path = column[5]
                        elif column[0] == "diffusivity":
                            quant.diffusivity = quant.fl_prec(column[3])
                            quant.epsi = quant.fl_prec(1.0/quant.diffusivity)
                        elif column[0] == "f" and column[1] == "factor":
                            quant.f_factor = quant.fl_prec(column[3])
                        elif column[0] == "stellar" and column[1] == "zenith":
                            quant.dir_angle = quant.fl_prec((180 - float(column[5])) * npy.pi / 180.0)
                            quant.mu_star = quant.fl_prec(npy.cos(quant.dir_angle))
                        elif column[0] == "geom." and column[1] == "zenith":
                            quant.geom_zenith_corr = self.__read_yes_no__(column[5])
                        elif column[0] == "internal" and column[1] == "temperature":
                            quant.T_intern = quant.fl_prec(column[4])
                        elif column[0] == "surface" and column[1] == "temperature":
                            quant.T_surf = quant.fl_prec(column[4])
                        elif column[0] == "asymmetry":
                            quant.g_0 = quant.fl_prec(column[4])
                        elif column[0] == "energy" and column[2] == "correction":
                            quant.energy_correction = self.__read_yes_no__(column[4])
                        elif column[0] == "surface" and column[1] == "albedo":
                            quant.surf_albedo = quant.fl_prec(column[3])
                            quant.surf_albedo = min(0.999, quant.surf_albedo) # everything above 0.999 albedo is not physical. fullstop.
                            quant.surf_albedo = quant.fl_prec(quant.surf_albedo)

                        # CONVECTIVE ADJUSTMENT
                        elif column[0] == "convective":
                            quant.convection = self.__read_yes_no__(column[3])
                        elif column[0] == "kappa" and column[1] == "value":
                            quant.kappa_manual_value = column[3]
                        elif column[0] == "entropy/kappa" and column[2] == "path":
                            self.entr_kappa_path = column[4]

                        # ASTRONOMICAL PARAMETERS
                        elif column[0] == "stellar" and column[2] == "model":
                            quant.stellar_model = column[4]
                        elif column[0] == "path" and column[2] == "stellar":
                            self.stellar_path = column[6]
                        elif column[0] == "planet" and column[1] == "=":
                            quant.planet = column[2]
                        elif column[0] == "path" and column[2] == "planet" and column[3] == "data":
                            self.planet_file = column[6]
                        elif column[0] == "surface" and column[1] == "gravity":
                            quant.g = quant.fl_prec(column[5])
                        elif column[0] == "orbital" and column[1] == "distance":
                            quant.a = quant.fl_prec(column[4])
                        elif column[0] == "radius" and column[1] == "planet":
                            quant.R_planet = quant.fl_prec(column[4])
                        elif column[0] == "radius" and column[1] == "star":
                            quant.R_star = quant.fl_prec(column[4])
                        elif column[0] == "temperature" and column[1] == "star":
                            quant.T_star = quant.fl_prec(column[4])

                        # EXPERIMENTAL
                        elif column[0] == "clouds":
                            quant.clouds = self.__read_yes_no__(column[2])
                        elif column[0] == "path" and column[2] == "cloud":
                            quant.cloud_path = column[5]
                        elif column[0] == "total" and column[1] == "cloud":
                            quant.cloud_opac_tot = quant.fl_prec(column[6])
                        elif column[0] == "cloud" and column[2] == "pressure":
                            quant.cloud_press = quant.fl_prec(column[6])
                        elif column[0] == "cloud" and column[1] == "width":
                            quant.cloud_width = quant.fl_prec(column[5])
                        elif column[0] == "number" and column[2] == "run-in":
                            quant.foreplay = npy.int32(column[5])
                        elif column[0] == "artificial" and column[2] == "opacity":
                            quant.fake_opac = quant.fl_prec(column[4])
                        elif column[0] == "use" and column[1] == "f" and column[2] == "approximation":
                            quant.approx_f = self.__read_yes_no__(column[5])

                        # VULCAN COUPLING
                        elif column[0] == "VULCAN" and column[1] == "coupling":
                            Vmod.V_coupling = self.__read_yes_no__(column[3])
                        elif column[0] == "path" and column[2] == "individual":
                            Vmod.mol_opac_path = column[5]
                        elif column[0] == "species" and column[1] == "file":
                            Vmod.species_file = column[3]
                        elif column[0] == "mixing" and column[2] == "file":
                            Vmod.mix_file = column[4]


        except IOError:
            print("ABORT - Input file not found!")
            raise SystemExit()

    def read_command_line(self, quant, Vmod):
        """ reads any eventual command-line arguments"""

        parser = argparse.ArgumentParser(description=
                                         "The following are the possible command-line parameters for HELIOS")

        parser.add_argument('-g', help='surface gravity [cm s^-2]', required=False)
        parser.add_argument('-a', help='orbital distance [AU]', required=False)
        parser.add_argument('-rstar', help='stellar radius [R_sun]', required=False)
        parser.add_argument('-tstar', help='stellar temperature [K]', required=False)
        parser.add_argument('-tintern', help='internal flux temperature [K]', required=False)
        parser.add_argument('-name', help='name of output', required=False)
        parser.add_argument('-Viter', help='VULCAN coupling iteration step nr.', required=False)
        parser.add_argument('-angle', help='zenith angle measured from the vertical', required=False)
        parser.add_argument('-isothermal', help='isothermal layers?', required=False)
        parser.add_argument('-postprocess', help='pure post-processing?', required=False)
        parser.add_argument('-temperaturepath', help='path to the temperature file', required=False)
        parser.add_argument('-plot', help='realtime plotting?', required=False)
        parser.add_argument('-opacitypath', help='path to the opacity table file', required=False)
        parser.add_argument('-energycorrection', help='include correction for global incoming energy?', required=False)
        parser.add_argument('-planet', help='name of the planet to be modeled', required=False)
        parser.add_argument('-nlayers', help='number of layers in the grid', required=False)
        parser.add_argument('-ptoa', help='pressure at the TOA', required=False)
        parser.add_argument('-pboa', help='pressure at the BOA', required=False)
        parser.add_argument('-f', help='f heat redistribution factor', required=False)
        parser.add_argument('-tau_lw', help='tau_lw', required=False)
        parser.add_argument('-star', help='spectral model of the star', required=False)
        parser.add_argument('-Vfile', help='path to the file with VULCAN mixing ratios', required=False)
        parser.add_argument('-kappa', help='adiabatic coefficient, kappa = (ln T / ln P)_S', required=False)


        args = parser.parse_args()

        if args.g:
            quant.g = quant.fl_prec(args.g)
            if quant.g < 10:
                quant.g = quant.fl_prec(10 ** quant.g)

        if args.a:
            quant.a = quant.fl_prec(args.a)

        if args.rstar:
            quant.R_star = quant.fl_prec(args.rstar)

        if args.tstar:
            quant.T_star = quant.fl_prec(args.tstar)

        if args.tintern:
            quant.T_intern = quant.fl_prec(args.tintern)

        if args.name:
            quant.name = args.name

        if args.Viter:
            Vmod.V_iter_nr = npy.int32(args.Viter)

        if args.angle:
            quant.dir_angle = quant.fl_prec((180 - float(args.angle)) * npy.pi / 180.0)
            quant.mu_star = quant.fl_prec(npy.cos(quant.dir_angle))

        if args.isothermal:
            quant.iso = self.__read_yes_no__(args.isothermal)

        if args.postprocess:
            quant.singlewalk = self.__read_yes_no__(args.postprocess)

        if args.temperaturepath:
            self.temp_path = args.temperaturepath

        if args.plot:
            quant.realtime_plot, quant.n_plot = self.set_realtime_plotting(args.plot)

        if args.opacitypath:
            self.ktable_path = args.opacitypath

        if args.energycorrection:
            quant.energy_correction = self.__read_yes_no__(args.energycorrection)

        if args.planet:
            quant.planet = args.planet

        if args.nlayers:
            quant.nlayer = npy.int32(args.nlayers)
            quant.ninterface = npy.int32(quant.nlayer + 1)

        if args.ptoa:
            quant.p_toa = quant.fl_prec(args.ptoa)

        if args.pboa:
            quant.p_boa = quant.fl_prec(args.pboa)

        if args.f:
            quant.f_factor = quant.fl_prec(args.f)

        if args.tau_lw:
            quant.tau_lw = quant.fl_prec(args.tau_lw)

        if args.star:
            quant.stellar_model = args.star

        if args.Vfile:
            Vmod.mix_file = args.Vfile

        if args.kappa:
            quant.kappa_manual_value = args.kappa

        # now that we know the name for sure, let's do some pleasantries
        print("\n### Welcome to HELIOS! This run has the name: " + quant.name + ". Enjoy the ride! ###")

    def read_opac_file(self, quant, Vmod):
        """ reads the opacity table """

        if Vmod.V_iter_nr == 0:

            try:
                with h5py.File(self.ktable_path, "r") as opac_file:
                    print("\nReading opacity table...")

                    # pre-tabulated opacity values
                    quant.opac_k = [k for k in opac_file["kpoints"][:]]

                    # temperature grid
                    quant.ktemp = [t for t in opac_file["temperatures"][:]]
                    quant.ntemp = npy.int32(len(quant.ktemp))

                    # pressure grid
                    quant.kpress = [p for p in opac_file["pressures"][:]]
                    quant.npress = npy.int32(len(quant.kpress))

                    # Rayleigh scattering cross-sections
                    quant.opac_scat_cross = [c for c in opac_file["weighted Rayleigh cross-sections"][:]]

                    # pre-tabulated mean molecular mass values (& convert from mu to mean mass)
                    quant.opac_meanmass = [m * pc.AMU for m in opac_file["meanmolmass"][:]]
                    # quant.opac_meanmass = [2.5 * pc.AMU for m in opac_file["meanmolmass"][:]]  # for testing

                    # wavelength grid
                    try:
                        quant.opac_wave = [x for x in opac_file["center wavelengths"][:]]
                    except KeyError:
                        quant.opac_wave = [x for x in opac_file["wavelengths"][:]]
                    quant.nbin = npy.int32(len(quant.opac_wave))

                    # Gaussian y-points
                    try:
                        quant.opac_y = [y for y in opac_file["ypoints"][:]]
                    except KeyError:
                        quant.opac_y = [0]
                    quant.ny = npy.int32(len(quant.opac_y))

                    # interface positions of the wavelength bins
                    try:
                        quant.opac_interwave = [i for i in opac_file["interface wavelengths"][:]]
                    except KeyError:
                        # quick and dirty way to get the lamda interface values
                        quant.opac_interwave = []
                        quant.opac_interwave.append(quant.opac_wave[0] - (quant.opac_wave[1] - quant.opac_wave[0])/2)
                        for x in range(len(quant.opac_wave) - 1):
                            quant.opac_interwave.append((quant.opac_wave[x+1] + quant.opac_wave[x])/2)
                        quant.opac_interwave.append(quant.opac_wave[-1] + (quant.opac_wave[-1] - quant.opac_wave[-2])/2)

                    # widths of the wavelength bins
                    try:
                        quant.opac_deltawave = [w for w in opac_file["wavelength width of bins"][:]]
                    except KeyError:
                        quant.opac_deltawave = []
                        for x in range(len(quant.opac_interwave) - 1):
                            quant.opac_deltawave.append(quant.opac_interwave[x + 1] - quant.opac_interwave[x])


            except OSError:
                print("\nABORT - \"", self.ktable_path, "\" not found!")
                raise SystemExit()

            # uncomment the following lines and play with the opacities. for debugging purposes only -- obviously :)
            # nump = len(quant.kpress)
            # nt = len(quant.ktemp)
            # nx = len(quant.opac_wave)
            # ny = len(quant.opac_y)
            # for t in range(nt):
            #     for p in range(nump):
            #         for x in range(nx):
            #             for y in range(ny):
            #                 #if temp 14 fixed at 800K, 22 1200K
            #                 # quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * t] = quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * 14]
            #
            #                 # if press fixed at 1bar
            #                 # quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * t] = quant.opac_k[y + ny * x + ny * nx * 18 + ny * nx * nump * t]
            #
            #                 # max value at 1e0
            #                 # if x < 275:
            #                 #     quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * t] = min(1e-1, quant.opac_k[y + ny * x + ny * nx * p + ny * nx * nump * t])


    @staticmethod
    def __read_correct_dataset__(path, dataset, flux_list):

        try:
            with h5py.File(path, "r") as file:
                for f in file[dataset][:]:
                    flux_list.append(f)
        except OSError:
            print("\nABORT - something is wrong when reading the stellar spectrum!")
            raise SystemExit()

    def read_planet_file(self, quant):

        planet_data = npy.genfromtxt(self.planet_file, names=True, dtype=None, skip_header=3, delimiter=None)

        first_column = npy.array(planet_data['Planet'],dtype='U')

        try:
            row = npy.where(first_column == quant.planet)[0][0]

            quant.R_planet = quant.fl_prec(planet_data['R_pl'][row])
            quant.g = quant.fl_prec(planet_data['g_surf'][row])
            quant.a = quant.fl_prec(planet_data['a'][row])
            quant.R_star = quant.fl_prec(planet_data['R_star'][row])
            quant.T_star = quant.fl_prec(planet_data['T_star'][row])

        except IndexError:
            print("No planet with name", quant.planet, "found. Please make sure the name is written correctly.")
            quant.planet = input("\nType in new planetary name or write \"abort\" to exit: \n\n\t")
            if quant.planet == "abort":
                print("Aborting...")
                raise SystemExit

            self.read_planet_file(quant)

    def read_entropy_table(self, quant):
        """ reads in entropy and kappa (for the stellar community: delad) values from ASCII table """

        if quant.kappa_manual_value == "file":

            print("\nReading entropy/kappa values from file.")

            entropy = []
            kappa = []

            try:
                with open(self.entr_kappa_path, "r") as entr_file:
                    next(entr_file)
                    next(entr_file)
                    for line in entr_file:
                        column = line.split()
                        if column:
                            quant.entr_press.append(10**quant.fl_prec(column[0]))
                            quant.entr_temp.append(10**quant.fl_prec(column[1]))
                            entropy.append(quant.fl_prec(column[4]))
                            kappa.append(quant.fl_prec(column[5]))
            except IndexError:

                with open(self.entr_kappa_path, "r") as entr_file:
                    next(entr_file)
                    next(entr_file)
                    for line in entr_file:
                        column = line.split()
                        if column:
                            quant.entr_press.append(10 ** quant.fl_prec(column[0]))
                            quant.entr_temp.append(10 ** quant.fl_prec(column[1]))
                            entropy.append(0)
                            kappa.append(quant.fl_prec(column[2]))

            quant.entr_press = self.delete_duplicates(quant.entr_press)
            quant.entr_temp = self.delete_duplicates(quant.entr_temp)
            quant.entr_press.sort()
            quant.entr_temp.sort()
            quant.entr_npress = npy.int32(len(quant.entr_press))
            quant.entr_ntemp = npy.int32(len(quant.entr_temp))

            # change into the correct order in terms of pressure and temperature
            for t in range(quant.entr_ntemp):

                for p in range(quant.entr_npress):

                    quant.opac_entropy.append(entropy[t + quant.entr_ntemp * p])
                    quant.opac_kappa.append(kappa[t + quant.entr_ntemp * p])

        else:
            # some value needed by the kernel "kappa_interpol"
            quant.entr_npress = npy.int32(1)
            quant.entr_ntemp = npy.int32(1)
            quant.entr_press = [0]
            quant.entr_temp = [0]
            quant.opac_kappa = [0]
            quant.opac_entropy = [0]

    def read_star(self, quant):
        """ reads the correct stellar spectrum from the corresponding file """

        if quant.stellar_model != "blackbody":

            try:
                self.__read_correct_dataset__(self.stellar_path,
                                              quant.stellar_model,
                                              quant.starflux)
                quant.real_star = npy.int32(1)
                print("\nReading", self.stellar_path + quant.stellar_model, "as spectral model of the host star.")

            except KeyError:

                print("\nThere is no such stellar spectrum found. Please check file path and data set.")
                inp = None
                while inp != "yes" and inp != "no":
                    inp = input("\n\tProceed with blackbody flux? (yes/no) \n\n\t")
                    if inp == "no":
                        print("\nAborting...")
                        raise SystemExit()
                    elif inp == "yes":
                        quant.stellar_model = "blackbody"
                        self.read_star(quant)
                    else:
                        print("\nInvalid input. Try again with \"yes\" or \"no\".")

        elif quant.stellar_model == "blackbody":
            quant.starflux = npy.zeros(quant.nbin, quant.fl_prec)
            print("\nUsing blackbody flux for the stellar irradiation.")

    @staticmethod
    def interpolate_to_own_press(old_press, old_array, new_press):

        new_array = interpolate.interp1d(old_press, old_array, bounds_error=False,
                                         fill_value=(old_array[-1], old_array[0]))(new_press)

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
                        try:
                            quant.T_surf = quant.fl_prec(column[8])  # will override the value in the param.dat file
                        except IndexError:
                            pass

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

        own_press = [quant.p_boa * npy.exp(npy.log(quant.p_toa / quant.p_boa) * p / (quant.nlayer - 1.0)) for p in range(quant.nlayer)]

        quant.T_restart = self.interpolate_to_own_press(file_press, file_temp, own_press)

if __name__ == "__main__":
    print("This module is for reading stuff. "
          "...stuff like the input file, or the opacity container, or the 'Lord of the Rings' by J. R. R. Tolkien.")
