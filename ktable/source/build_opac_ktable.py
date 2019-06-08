# =============================================================================
# Module for building k-distribution functions from the HELIOS-K output
# Copyright (C) 2018 Matej Malik
#
# =============================================================================
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
# =============================================================================

import numpy as np
from numpy.polynomial import Chebyshev as T
from numpy.polynomial.legendre import leggauss as G
import os
import h5py
from source import tools as tls


class Production(object):
    """ class to produce the molecular k-distribution functions """

    def __init__(self):
        self.rootlist = []
        self.filelist = []
        self.press_exp_list = []
        self.temp_list = []
        self.temp_mol_list = []
        self.mol_list = []
        self.numin_list_tot = []
        self.numax_list_tot = []
        self.mol_info = []
        self.ypoints = []

        self.ky = []
        self.coeffs = []
        self.ystart = []
        self.numin_mol_list = []
        self.numax_mol_list = []
        self.press_cgs = []
        self.lamda_delim = []
        self.lamda_mid = []
        self.d_lamda = []
        self.numin_tot = None
        self.numax_tot = None
        self.nbins_this_mol = None
        self.nbins_tot = None
        self.nk_should_be = None
        self.kmin = None

    # Here you add the new molecular names if needed. Position = HELIOS-K molecule number

    mol_name_list = ["xxx"] * 100
    mol_name_list[1] = "H2O"
    mol_name_list[2] = "CO2"
    mol_name_list[5] = "CO"
    mol_name_list[6] = "CH4"
    mol_name_list[7] = "O2"
    mol_name_list[8] = "NO"
    mol_name_list[9] = "SO2"
    mol_name_list[11] = "NH3"
    mol_name_list[13] = "OH"
    mol_name_list[23] = "HCN"
    mol_name_list[26] = "C2H2"
    mol_name_list[28] = "PH3"
    mol_name_list[31] = "H2S"
    mol_name_list[47] = "SO3"
    mol_name_list[80] = "VO"
    mol_name_list[81] = "TiO"
    mol_name_list[83] = "AlO"
    mol_name_list[84] = "SiO"
    mol_name_list[85] = "CaO"
    mol_name_list[86] = "SiH"
    mol_name_list[87] = "CaH"
    mol_name_list[89] = "PO"
    mol_name_list[90] = "MgH"
    mol_name_list[91] = "NaH"
    mol_name_list[92] = "AlH"
    mol_name_list[93] = "CrH"
    mol_name_list[98] = "CIA_H2H2"
    mol_name_list[99] = "CIA_H2He"

    g20 = G(20)
    yg20 = g20[0]
    wg20 = g20[1]
    ny = len(yg20)

    def gen_ypoints(self):
        """ generates ypoints """

        for j in range(self.ny):
            self.ypoints.append(0.5*self.yg20[j]+0.5)

    @staticmethod
    def delete_duplicates(long_list):
        """ delete all duplicates in a list and return new list """

        short_list = []
        for item in long_list:
            if item not in short_list:
                short_list.append(item)
        return short_list

    # changing filenames in hindsight -- good to keep in case this becomes necessary in future
    # @staticmethod
    # def fix_exomol_name(path):
    #     """ change the exomol file name to a manageable one """
    #
    #     root_list = []
    #     file_list = []
    #
    #     for roots, dirs, files in os.walk(path):
    #
    #         if not roots.endswith("/"):
    #             roots += "/"
    #         root_list.append(roots)
    #
    #         for file in files:
    #             file_list.append(file)
    #
    #     for root in root_list:
    #         for file in file_list:
    #             try:
    #                 if file.startswith("Out_h2he"):
    #                     os.rename(root+file, root + "Out_99" + file[8:])
    #                 elif file.startswith("Info_h2he"):
    #                     os.rename(root+file, root + "Info_99" + file[9:])
    #             except FileNotFoundError:
    #                 continue

    def search_dir(self,param):
        """ search directories for output files (without cia)"""

        for roots, dirs, files in os.walk(param.heliosk_path):
            if not roots.endswith("/"):
                roots += "/"
            self.rootlist.append(roots)
            for f in files:
                if f.endswith("_cbin.dat"):
                    self.filelist.append(f)

    def get_parameters(self):
        """ gets the parameters from the filenames """

        for f in self.filelist:
            self.mol_list.append(int(f[4:6]))
            self.temp_list.append(int(f[19:24]))
            if f[25] == 'n':
                self.press_exp_list.append(-0.01*int(f[26:29]))
            elif f[25] == 'p':
                self.press_exp_list.append(0.01*int(f[26:29]))
            self.numin_list_tot.append(int(f[7:12]))
            self.numax_list_tot.append(int(f[13:18]))

    def resort(self):
        """ resort parameters """

        self.mol_list = self.delete_duplicates(self.mol_list)
        self.temp_list = self.delete_duplicates(self.temp_list)
        self.press_exp_list = self.delete_duplicates(self.press_exp_list)
        self.numin_list_tot = self.delete_duplicates(self.numin_list_tot)
        self.numax_list_tot = self.delete_duplicates(self.numax_list_tot)

        self.mol_list = np.sort(self.mol_list)
        self.press_exp_list = np.sort(self.press_exp_list)
        self.temp_list = np.sort(self.temp_list)
        self.numin_list_tot = np.sort(self.numin_list_tot)
        self.numax_list_tot = np.sort(self.numax_list_tot)

        self.numin_tot = min(self.numin_list_tot)
        self.numax_tot = max(self.numax_list_tot)

    def init_mol(self,m):
        """ initializes the molecule treatment """

        for f in self.filelist:
            if f.startswith("Out_"+"{:02d}".format(self.mol_list[m])):
                self.numin_mol_list.append(int(f[7:12]))
                self.numax_mol_list.append(int(f[13:18]))
                self.temp_mol_list.append(int(f[19:24]))

        self.numin_mol_list = self.delete_duplicates(self.numin_mol_list)
        self.numax_mol_list = self.delete_duplicates(self.numax_mol_list)
        self.temp_mol_list = self.delete_duplicates(self.temp_mol_list)

        self.numin_mol_list = np.sort(self.numin_mol_list)
        self.numax_mol_list = np.sort(self.numax_mol_list)
        self.temp_mol_list = np.sort(self.temp_mol_list)

        ## take the real spectral boundaries
        numin_mol = min(self.numin_mol_list)
        numax_mol = max(self.numax_mol_list)

        print("\nmolecule: ",self.mol_name_list[self.mol_list[m]], self.mol_list[m])
        self.mol_info.append(self.mol_name_list[self.mol_list[m]])
        print("spectral range goes from",numin_mol,"until",numax_mol,".")
        print("number of temperature points:", len(self.temp_mol_list))
        if max(self.temp_mol_list) < max(self.temp_list):
            print("WARNING: The temperature is extrapolated from " + str(max(self.temp_mol_list)) + " to " + str(max(self.temp_list)))
        print("number of pressure points:", len(self.press_exp_list))
        print("\nGenerating k-table...\n")

    def read_chebyshev(self,m,p,t,char,n):
        """ reads in chebyshev coefficients from a file """

        try:
            temp = self.temp_mol_list[t]
        except IndexError:
            temp = max(self.temp_mol_list)

        file = "Out_"+"{:02d}".format(self.mol_list[m])+"_"+"{:05d}".format(self.numin_mol_list[n])+"_"+"{:05d}".format(self.numax_mol_list[n])+"_"+"{:05d}".format(temp)+"_"+char+"{:03.0f}".format(abs(100*self.press_exp_list[p]))+"_cbin.dat"

        for root in self.rootlist:

            try:
                with open(root+file, "r") as cbin_file:
                    counting_lines = 0
                    for line in cbin_file:
                            wholeline = line.split()
                            if len(wholeline) != 0:
                                counting_lines += 1
                            for i in range(0,len(wholeline)):
                                #first nr. each line kmin
                                if i == 0:
                                    self.kmin = float(wholeline[i])
                                #2nd nr. each line is ystart
                                elif i == 1:
                                    self.ystart.append(float(wholeline[i]))
                                #rest are the chebyshev coefficients
                                else:
                                    self.coeffs.append(float(wholeline[i]))
                break

            except IOError:
                continue

        if n == 0 and p == 0 and t == 0:
            ## number of bins and blocks consistency check
            self.nbins_perfile = counting_lines
            print('bins counted per file:',self.nbins_perfile)

            #checks how long the chebyshev series is
            self.ncoeffs = int(len(self.coeffs)/self.nbins_perfile)
            print("number of chebyshev coefficients:",self.ncoeffs)

            print("\nLooks good! Generating k-table...")

    def calc_opac(self):
        """ calculates the opacities from the Chebyshev coefficients """

        for b in range(self.nbins_perfile):

            # chebyshev series with the correct coefficients
            series = T(self.coeffs[self.ncoeffs*b:self.ncoeffs*(b+1)])

            for y in range(self.ny):

                yp = self.ypoints[y]

                if yp >= self.ystart[b]:
                    arg = (2.0*yp-1.0-self.ystart[b])/(1.0-self.ystart[b])
                    self.ky.append(np.exp(series(arg)))
                else:
                    self.ky.append(self.kmin)

    def cons_check(self):
        """ checks for self-consistency of reading """

        self.nbins_this_mol = self.nbins_perfile * len(self.numin_mol_list)
        print("\nWe have",self.nbins_this_mol," bins for this molecule.")

        self.nbins_tot = self.nbins_perfile * len(self.numin_list_tot)
        print("But",self.nbins_tot,"bins in total.")
        self.nk_should_be = self.ny * self.nbins_tot

    def fill_rearr(self):
        """ fills up all the k-values and rearranges to increasing wavelength order """

        ## fill until all ky same length
        if self.ny * self.nbins_this_mol < self.nk_should_be:

            To_refill = self.nk_should_be - self.ny * self.nbins_this_mol
            for i in range(0,To_refill):
                self.ky.append(self.kmin)

        ### rearrange ky according to wavelength
        nk = self.nk_should_be
        last_nk = self.ky[len(self.ky)-nk:len(self.ky)]
        self.ky = self.ky[0:len(self.ky)-nk]
        last_nk_new=[]
        for n in np.arange(self.nbins_tot-1,0-1,-1):
            last_nk_new.extend(last_nk[n*self.ny:(n+1)*self.ny])
        self.ky.extend(last_nk_new)

    def conversion(self):
        """ rearranges and converts parameters for write out """

        ########wavenumber to wavelength conversion plus generating grid ######
        d_nu=(self.numax_tot-self.numin_tot)/self.nbins_tot

        for l in range(0,self.nbins_tot+1):
            nu_delim = self.numin_tot+l*d_nu
            if l == 0:
                self.lamda_delim.append(10.0)
            else:
                self.lamda_delim.append(1.0/nu_delim)
            if l < self.nbins_tot:
                nu_mid = self.numin_tot+(l+0.5)*d_nu
                self.lamda_mid.append(1.0/nu_mid)

        for l in range(0,self.nbins_tot):
                self.d_lamda.append(abs(self.lamda_delim[l+1]-self.lamda_delim[l]))

        ### sort the wavelength arrays to be from small to large ###
        self.lamda_delim=np.sort(self.lamda_delim)
        self.lamda_mid=np.sort(self.lamda_mid)
        self.d_lamda=np.sort(self.d_lamda)

        ### convert pressure exponents to cgs pressure
        for p in self.press_exp_list:

            # increase accuracy of pressure values
            if p < 0:
                char = "n"
            else:
                char = "p"

            p_string = char + "{:.2f}".format(abs(p))

            if p_string.endswith("3"):
                p_string += "333333"
            elif p_string.endswith("6"):
                p_string += "666666"

            if p_string[0] == 'n':
                p_fixed = - float(p_string[1:])
            else:
                p_fixed = float(p_string[1:])

            # convert to cgs units
            self.press_cgs.append(10**p_fixed*1e6)

    def write_tell(self,param, m):
        """ writes the output for each molecule and tells the user """

        h5_name = self.mol_name_list[self.mol_list[m]]

        # create directory if necessary
        try:
            os.makedirs(param.resampling_path)
        except OSError:
            if not os.path.isdir(param.resampling_path):
                raise

        with h5py.File(param.resampling_path+h5_name+"_opacities.h5", "w") as f:

            f.create_dataset("pressures", data=self.press_cgs)
            f.create_dataset("temperatures", data=self.temp_list)
            f.create_dataset("interface wavelengths",data=self.lamda_delim)
            f.create_dataset("center wavelengths",data=self.lamda_mid)
            f.create_dataset("wavelength width of bins",data=self.d_lamda)
            f.create_dataset("ypoints",data=self.ypoints)
            f.create_dataset("kpoints",data=self.ky)

        print("\nSuccesfully completed molecule", self.mol_name_list[self.mol_list[m]],"!")
        print("--------------------------------------------")
        print("\nLet's move on -- Bam Bam Bam Bam")

    def check_if_exist(self, param, m):

        h5_name = self.mol_name_list[self.mol_list[m]]

        try:

            h5file = h5py.File(param.resampling_path + h5_name + "_opacities.h5", "r")
            print("File " + param.resampling_path + h5_name + "_opacities.h5 already exists. Skipping this molecule...")
            return True

        except OSError:
            return False


    def big_loop(self, param):
        """ loops through the molecules and connects the other methods - aka the big guy """

        for m in range(len(self.mol_list)):

            self.press_cgs = []
            self.lamda_delim = []
            self.lamda_mid = []
            self.d_lamda = []
            self.numin_mol_list = []
            self.numax_mol_list = []
            self.temp_mol_list = []
            # list with all the opacities for each y value
            self.ky = []

            self.init_mol(m)

            percentage = 0

            if self.check_if_exist(param, m):

                # skip to next molecule
                continue

            for t in range(len(self.temp_list)):

                for p in range(len(self.press_exp_list)):

                    print('t:', t, 'p:', p)

                    tls.percent_counter(t, len(self.temp_list), p, len(self.press_exp_list))

                    if self.press_exp_list[p] >= 0:
                        char = 'p'
                    else:
                        char = 'n'

                    for n in range(len(self.numin_mol_list)):

                        self.coeffs = []
                        self.ystart = []

                        self.read_chebyshev(m,p,t,char,n)

                        self.calc_opac()

                    if p == 0 and t == 0:
                        self.cons_check()

                    self.fill_rearr()

            self.conversion()

            self.write_tell(param, m)

    def write_names(self):
        """ stores the names of the included molecules for use in the info file """

        self.mol_info = " "
        for m in self.mol_list:
            if m != 0:
                self.mol_info += self.mol_name_list[m] + " "

    @staticmethod
    def success():
        """ prints out a success message """

        print("\nSuccessfully produced the k-distribution tables of the individual molecules :)")

if __name__ == "__main__":
    print("This module produces the molecular k-distribution functions from the HELIOS-K output.")