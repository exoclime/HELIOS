# =============================================================================
# Module for building k-distribution functions from the HELIOS-K output
# Copyright (C) 2016 Matej Malik
# 
# ATTENTION: THIS REQUIRES A SPECIAL OUTPUT FORMAT. 
# YOU USUALLY WANT TO USE THE GENERAL SCRIPT "build_opac_gen.py"
#
# Input:
# - the Helios-k output "Out_*_cbin.dat" with the Chebyshev coefficients
#   and the corresponding Info_*_.dat is needed for each opacity source
#   They need to be located in the directory specified in the parameter file.
#   (subdirectories allowed)
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

class Production(object):
    """ class to produce the molecular k-distribution functions """
    def __init__(self):
        self.rootlist=[]
        self.filelist=[]
        self.press_exp_list=[]
        self.temp_list=[]
        self.mol_nr_list=[]
        self.numin_list_tot=[]
        self.numax_list_tot=[]
        self.mol_info=[]
        self.ypoints=[]
        
        self.ky=[] 
        self.coeffs=[]
        self.ystart=[]
        self.numin_mol_list=[]
        self.numax_mol_list=[]
        self.press_cgs = [] 
        self.lamda_delim=[]
        self.lamda_mid=[]
        self.d_lamda=[]
        
    mol_name_list=["cia","h2o","co2","xxx","xxx","co","ch4"]
    g20=G(20)
    yg20=g20[0]
    wg20=g20[1]
    ny = len(yg20)
    
    def gen_ypoints(self):
        """ generates ypoints """
        for j in range(self.ny):
            self.ypoints.append(0.5*self.yg20[j]+0.5)
        
    def delete_duplicates(self,long_list):
        """ delete all duplicates in a list and return new list """
        short_list=[]
        for item in long_list:
           if item not in short_list:
              short_list.append(item)
        return short_list
     
    def search_dir(self,param):
        """ search directories for output files (without cia)"""
        for roots, dirs, files in os.walk(param.dir):
            if not roots.endswith("/"):
                roots=roots+"/"
            self.rootlist.append(roots)
            for f in files:
                   if not "cia" in f:
                       if f.endswith("_cbin.dat"):
                           self.filelist.append(f)
                   
    def get_parameters(self):
        """ gets the parameters from the filenames """
        for f in self.filelist:
            self.mol_nr_list.append(int(f[4:6]))
            self.temp_list.append(int(f[19:23]))
            if f[24]=='n':
                self.press_exp_list.append(-0.01*int(f[25:28]))
            else:
                self.press_exp_list.append(0.01*int(f[25:28]))
            self.numin_list_tot.append(int(f[7:12]))
            self.numax_list_tot.append(int(f[13:18]))
            
    def resort(self):
        """ resort parameters """
        self.mol_nr_list=self.delete_duplicates(self.mol_nr_list)
        self.temp_list=self.delete_duplicates(self.temp_list)
        self.press_exp_list=self.delete_duplicates(self.press_exp_list)
        self.numin_list_tot=self.delete_duplicates(self.numin_list_tot)
        self.numax_list_tot=self.delete_duplicates(self.numax_list_tot)
        
        self.mol_nr_list=np.sort(self.mol_nr_list)
        self.press_exp_list=np.sort(self.press_exp_list)
        self.temp_list=np.sort(self.temp_list)
        self.numin_list_tot=np.sort(self.numin_list_tot)
        self.numax_list_tot=np.sort(self.numax_list_tot)
        
        self.numin_tot=min(self.numin_list_tot)
        self.numax_tot=max(self.numax_list_tot)

    def init_mol(self,m):
        """ initializes the molecule treatment """
        for f in self.filelist:
            if f.startswith("Out_"+"{:02d}".format(self.mol_nr_list[m])):
                self.numin_mol_list.append(int(f[7:12]))
                self.numax_mol_list.append(int(f[13:18]))
        
        self.numin_mol_list=self.delete_duplicates(self.numin_mol_list)
        self.numax_mol_list=self.delete_duplicates(self.numax_mol_list)

        self.numin_mol_list=np.sort(self.numin_mol_list)
        self.numax_mol_list=np.sort(self.numax_mol_list)

        ## take the real spectral boundaries
        numin_mol=min(self.numin_mol_list)
        numax_mol=max(self.numax_mol_list)
        print(self.mol_nr_list)
        print("Starting molecule: ",self.mol_name_list[self.mol_nr_list[m]])
        self.mol_info.append(self.mol_name_list[self.mol_nr_list[m]])
        print("spectral range goes from",numin_mol,"until",numax_mol,".")
        print("\nnumber of temperature points:", len(self.temp_list))
        print("\nnumber of pressure points:", len(self.press_exp_list))
        print("\nResampling opacities...")

    def read_chebyshev(self,m,p,t,char,n):
        """ reads in chebyshev coefficients from a file """

        file="Out_"+"{:02d}".format(self.mol_nr_list[m])+"_"+"{:05d}".format(self.numin_mol_list[n])+"_"+"{:05d}".format(self.numax_mol_list[n])+"_"+"{:04d}".format(self.temp_list[t])+"_"+char+"{:03.0f}".format(abs(100*self.press_exp_list[p]))+"_cbin.dat"
                
        for root in self.rootlist:
            try:
                with open(root+file, "r") as cbin_file:
                    counting_lines=0
                    for line in cbin_file:
                            wholeline = line.split()
                            if len(wholeline)!=0:
                                counting_lines+=1
                            for i in range(0,len(wholeline)):
                                #first nr. each line kmin
                                if i==0:                       
                                    self.kmin=float(wholeline[i])
                                #2nd nr. each line is ystart
                                elif i==1:                       
                                    self.ystart.append(float(wholeline[i]))
                                #rest are the chebyshev coefficients
                                else:
                                    self.coeffs.append(float(wholeline[i]))
                break
            
            except(FileNotFoundError):
                continue
            
        if n==0:
            ## number of bins and blocks consistency check
            self.nbins_perfile=counting_lines
            print('bins counted per file:',self.nbins_perfile)

            #how long is the chebyshev series?
            self.ncoeffs=int(len(self.coeffs)/self.nbins_perfile)
            print("number of chebyshev coefficients:",self.ncoeffs)

            print("\nLooks good! Generating k-table...") 
        
    def calc_opac(self):
        """ calculates the opacities from the Chebyshev coefficients """
        for b in range(self.nbins_perfile):

            #chebyshev series with the correct coefficients
            series=T(self.coeffs[self.ncoeffs*b:self.ncoeffs*(b+1)])
                    
            for y in range(self.ny):

                yp=self.ypoints[y]
                
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
        print("\nBut",self.nbins_tot,"bins in total.")
        self.nk_should_be = self.ny * self.nbins_tot
        
    def fill_rearr(self):
        """ fills up all the k-values and rearranges to increasingwavelength order """
       
        ## fill until all ky same length
        if self.ny * self.nbins_this_mol < self.nk_should_be:
            
            print("\nFilling until 30k wavenumber...\n")
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
            nu_delim=self.numin_tot+l*d_nu
            if l==0:
                self.lamda_delim.append(10.0)
            else:
                self.lamda_delim.append(1.0/nu_delim)
            if l < self.nbins_tot:
                nu_mid=self.numin_tot+(l+0.5)*d_nu
                self.lamda_mid.append(1.0/nu_mid)
            
        for l in range(0,self.nbins_tot):
                self.d_lamda.append(np.abs(self.lamda_delim[l+1]-self.lamda_delim[l]))

        ### sort the wavelength arrays to be from small to large ###
        self.lamda_delim=np.sort(self.lamda_delim)
        self.lamda_mid=np.sort(self.lamda_mid)
        self.d_lamda=np.sort(self.d_lamda)
        
        ### convert pressure exponents to cgs pressure
        for p in self.press_exp_list:
            self.press_cgs.append(10**p*1e6)

    def write_tell(self,m):
        """ writes the output for each molecule """
        
        h5_name = self.mol_name_list[self.mol_nr_list[m]]
        
        with h5py.File("output/"+h5_name+"_opacities.h5", "w") as f:
    
            f.create_dataset("pressures", data=self.press_cgs)
            f.create_dataset("temperatures", data=self.temp_list)
            f.create_dataset("interface wavelengths",data=self.lamda_delim)
            f.create_dataset("centre wavelengths",data=self.lamda_mid)
            f.create_dataset("wavelength width of bins",data=self.d_lamda)
            f.create_dataset("ypoints",data=self.ypoints)
            f.create_dataset("kpoints",data=self.ky)

        print("\nSuccesfully completed molecule", self.mol_name_list[self.mol_nr_list[m]],"!")
        print("--------------------------------------------")
        print("\nLet's move on -- Bam Bam Bam Bam")        
        
    def big_loop(self):
        """ loops through the molecules and connects the other methods - aka the big guy """
    
        for m in range(len(self.mol_nr_list)):
                        
            self.press_cgs = []    
            self.lamda_delim=[]
            self.lamda_mid=[]
            self.d_lamda=[]
            self.numin_mol_list=[]
            self.numax_mol_list=[]
            #list with all the opacities for each y value
            self.ky=[]
            
            self.init_mol(m)

            for t in range(len(self.temp_list)):
        
                for p in range(len(self.press_exp_list)):
        
                    if self.press_exp_list[p]>=0:
                        char='p'
                    else:
                        char='n'

                    for n in range(len(self.numin_mol_list)):
                             
                        self.coeffs = []
                        self.ystart = []
                        
                        self.read_chebyshev(m,p,t,char,n)                      
                        
                        self.calc_opac()

                    self.cons_check()
                    
                    self.fill_rearr()
              
            self.conversion()
            
            self.write_tell(m)
         
    def write_names(self):
        """ stores the names of the included molecules for use in the info file """
        
        self.mol_info = " "
        for m in self.mol_nr_list:
            if m != 0:
                self.mol_info += self.mol_name_list[m] + " "
            
    def success(self):
        print("\nSuccessfully produced the k-distribution tables of the individual molecules :)")
        
if __name__ == "__main__":
    print("This module produces the molecular k-distribution functions from the HELIOS-K output.")