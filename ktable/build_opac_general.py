# =============================================================================
# Module for building k-distribution functions from the HELIOS-K output
# Copyright (C) 2016 Matej Malik
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
        self.red_filelist=[]
        self.name_list=[]
        self.press_list=[]
        self.press_info=[]
        self.temp_list=[]
        self.mol_nr_list=[]
        self.numin_list=[]
        self.numax_list=[]
        self.mol_info=[]
        self.ypoints=[]
        self.file_name=[]
        self.cia=[]
        self.press_info=[]
        self.pfile=None
        
        self.ky=[] 
        self.coeffs=[]
        self.ystart=[]
        self.press_cgs = [] 
        self.lamda_delim=[]
        self.lamda_mid=[]
        self.d_lamda=[]
        
    mol_name_list=["cia","h2o","co2","xxx","xxx","co","ch4"]
    cia_name_list=["H2-H2","H2-He"]
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
        """ search directories for output files and check that each file has a different name. otherwise code is confused."""
        for roots, dirs, files in os.walk(param.dir):
            if not roots.endswith("/"):
                roots=roots+"/"
            self.rootlist.append(roots)
            for f in files:
                self.filelist.append(f)
                    
        abort_f=[]
        for f in self.filelist:
            if f not in abort_f or f == ".DS_Store":
                abort_f.append(f)
            else:
                print("\n--> Duplicate filenames found. Please make sure, that each file is named differently.")
                print("\naborting...")
                print("\nPS Do not forget to change name in Info-file.")
                raise SystemExit()

    def kick_files(self):
        """removes files not associated cia from filelist """
        for root in self.rootlist:
            for f in self.filelist:
                if "Info" in f:
                    try:
                        with open(root+f, "r") as infofile:
                            for line in infofile:
                                column = line.split()
                                if column:
                                    if column[0] =="name":
                                         name = column[2]
                                    elif column[0] =="cia":
                                        cia_value = column[3]
                                        if cia_value == "H2-H2" or cia_value == "H2-He":
                                            self.red_filelist.append("Info_"+name+".dat")
                                            self.red_filelist.append("Out_"+name+"_cbin.dat")
                    except(FileNotFoundError):
                        continue

        self.red_filelist = self.delete_duplicates(self.red_filelist)
        self.filelist = self.red_filelist
        
    def read_info(self):
        """ reads in the parameters from the info files and gets the correct pressures """
        
        for root in self.rootlist:
            for f in self.filelist:
                if "Info" in f:
                    try:
                        with open(root+f, "r") as infofile:                                           
                            for line in infofile:
                                column = line.split()
                                if column:
                                    if column[0] =="name":
                                        self.name_list.append(column[2])
                                    elif column[0] =="T":
                                        self.temp_list.append(float(column[2]))
                                    elif column[0] =="P" and column[1] != "in":
                                        self.press_info.append(float(column[2]))
                                    elif column[0] =="numin":
                                        self.numin_list.append(float(column[2]))
                                    elif column[0] =="numax":
                                        self.numax_list.append(float(column[2]))
                                    elif column[0] =="dnu":
                                        self.dnu = float(column[2])
                                    elif column[0] =="cutMode":
                                        self.cutMode=int(column[2])
                                    elif column[0] =="cut":
                                        self.cut=int(column[2])
                                    elif column[0] =="nC":
                                        self.ncoeffs=int(column[2])
                                    elif column[0] =="nbins":
                                        self.nbins=int(column[2])
                                    elif column[0] =="Molecule":
                                        self.mol_nr_list.append(int(column[2]))
                                    elif column[0] =="cia":
                                        cia_value = column[3]
                                        self.cia.append(cia_value)
                                    elif column[0] =="P" and column[1] == "in" and column[2] == "file:":
                                        self.pfile=column[3] 
                    except(FileNotFoundError):
                        continue
                        
        if self.pfile != None:
            for root in self.rootlist:
                try:
                    with open(root+self.pfile, "r") as pressfile:
                        for line in pressfile:
                            self.press_list.append(float(line))
                    break
                except(FileNotFoundError):
                    continue
        
        if len(self.press_list) == 0:
            for p in self.press_info:
                self.press_list.append(p)
                
        if len(self.press_list) == 1:
            print("\nWARNING. Only 1 pressure value found, namely:"+str(self.press[0]))
            input("\nPlease acknowledge that this is correct. Press ENTER or SPACE.")
        elif len(self.press_list) == 0:
            print("\nPressure array is zero.")
            print("\nPlease check if pressure file is in the main directory.")
            print("\nPlease also check whether the path to the main directory is set correctly.")
            print("\nAborting...")
            raise SystemExit()
    
    def resort(self):   
        self.temp_list=self.delete_duplicates(self.temp_list)
        self.press_list=self.delete_duplicates(self.press_list)
        self.numin_list=self.delete_duplicates(self.numin_list)
        self.numax_list=self.delete_duplicates(self.numax_list)
        
        self.numin=min(self.numin_list)
        self.numax=max(self.numax_list)

    def read_chebyshev(self,root,index):
        """ reads in chebyshev coefficients from a file """
        
        with open(root+"Out_" + self.name_list[index] + "_cbin.dat", "r") as cbin_file:
            print("\nsearching...and found the file "+"Out_" + self.name_list[index] + "_cbin.dat")
            #counting_lines to obtain the number of bins
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
                        elif i > 1 and i < self.ncoeffs + 2:
                            self.coeffs.append(float(wholeline[i]))

        
        ## number of bins and blocks consistency check
        nblocks=counting_lines
        self.nbins_counted = nblocks//(len(self.press_list)*len(self.temp_list))
        print('nbins_counted:',self.nbins_counted)
        if self.nbins_counted//self.nbins == 1:
            print("No spectral division found. We have "+str(self.nbins)+" bins in total.")
        else:
            n_sub = self.nbins_counted//self.nbins
            print("\nSpectral division found. We have "+str(n_sub)+" times "+str(self.nbins)+" = "+str(self.nbins_counted)+" bins in total.")

        if (self.nbins_counted/self.nbins)%1!=0:
            print("\nDANGER: Helios-K file is broken. Its length is too short.")
            print("\naborting....")
            raise SystemExit()

        ## number of chebyshev coefficients consistency check
        if self.ncoeffs == len(self.coeffs)//nblocks:
            print("\nPassed file dimension test!")
        else:
            print("\nfailed file dimension test!")
            print("aborting...")
            raise SystemExit()

        print("\nLooks good! Generating k-table...") 
        
    def calc_opac(self):
        """ calculates the opacities from the Chebyshev coefficients for the whole T, P range"""         

        for t in range(0,len(self.temp_list)):
        
            for p in range(0,len(self.press_list)):
                
                for n in range(0,self.nbins_counted):

                    #pick block in descending bin order, but ascending P and T order
                    pick_block=t*len(self.press_list) * self.nbins_counted + (p+1) * self.nbins_counted - 1 - n
                    #chebyshev series with the correct coefficients
                    series=T(self.coeffs[self.ncoeffs*pick_block:self.ncoeffs*(pick_block+1)])
                    
                    for y in range(0,self.ny):

                        yp=self.ypoints[y]
                        if yp >= self.ystart[pick_block]:
                            self.ky.append(np.exp(series
                                ((2.0*yp-1.0-self.ystart[pick_block])/(1.0-self.ystart[pick_block]))
                                         ))
                        else:
                            self.ky.append(self.kmin)
        
    def screen_info(self,index):
        ## additional screen output at 1st iteration to check whether parameters fine                  
        if self.mol_nr_list[index]!=0:
            print("\nMolecule: ",self.mol_name_list[self.mol_nr_list[index]])
            if self.cia[index] =="H2-He" or self.cia[index] =="H2-H2":
                print("CIA is included in form of "+self.cia[index])
                print("!!! WARNING: INCLUDING CIA INTO MOLECULAR OPACITIES IS STRONGLY DISCOURAGED AS THE CORRECT MIXING RATIOS CANNOT BE APPLIED !!!")
                inpt=input("\n\n\n\tDo you want to proceed anyway? (y/n):")
                while inpt != "y" and inpt != "n":
                    inpt = input("\n Input not valid. Please type 'y' or 'n'.")
                if inpt == "y":
                    print("\ncontinuing...\n")
                elif inpt == "n":
                    print("\nk-table production aborted!")
                    raise SystemExit()                        
            else:
                print("CIA is not included")
        elif self.mol_nr_list[index]==0:
            print("\nCIA opacities for the molecules "+self.cia[index])
        
        print("minimum wavenumber: ",self.numin," , maximum wavenumber: ",self.numax)

        #how long is the chebyshev series?
        print("number of chebyshev coefficients:",self.ncoeffs)
        
    def conversion(self):
        """ rearranges and converts parameters for write out """
        
        ########wavenumber to wavelength conversion plus generating grid #######   
        d_nu=(self.numax-self.numin)/self.nbins_counted

        for l in range(0,self.nbins_counted+1):
            nu_delim=self.numin+l*d_nu
            if l==0:
                self.lamda_delim.append(10.0)
            else:
                self.lamda_delim.append(1.0/nu_delim)
            if l < self.nbins_counted:
                nu_mid=self.numin+(l+0.5)*d_nu
                self.lamda_mid.append(1.0/nu_mid)

        for l in range(0,self.nbins_counted):
                self.d_lamda.append(np.abs(self.lamda_delim[l+1]-self.lamda_delim[l]))

                    
        ### sort the wavelength arrays to be from small to large ###
        self.lamda_delim=np.sort(self.lamda_delim)
        self.lamda_mid=np.sort(self.lamda_mid)
        self.d_lamda=np.sort(self.d_lamda)
            
        for p in self.press_list:
            self.press_cgs.append(p*1e6*1.10325) # convert from atmo units to cgs units
        
    def write_tell(self,index):
        """ writes the output for each molecule with correct name """
        
        # naming the files correctly depending on CIA choice
        if self.mol_nr_list[index] != 0:
            h5_name = self.mol_name_list[self.mol_nr_list[index]]
            if self.cia[index] == "H2-H2" or self.cia[index] == "H2-He":
                h5_name = self.mol_name_list[self.mol_nr_list[index]]+"_cia_" + self.cia[index]
        elif self.mol_nr_list[index] == 0:
            h5_name = "cia_" + self.cia[index]
        
        with h5py.File("output/"+h5_name+"_opacities.h5", "w") as f:
    
            f.create_dataset("pressures", data=self.press_cgs)
            f.create_dataset("temperatures", data=self.temp_list)
            f.create_dataset("interface wavelengths",data=self.lamda_delim)
            f.create_dataset("centre wavelengths",data=self.lamda_mid)
            f.create_dataset("wavelength width of bins",data=self.d_lamda)
            f.create_dataset("ypoints",data=self.ypoints)
            f.create_dataset("kpoints",data=self.ky)

        print("\nSuccesfully completed molecule", self.mol_name_list[self.mol_nr_list[index]],"!")
        print("--------------------------------------------")
        print("\nLet's move on -- Bam Bam Bam Bam") 
        
    def big_loop(self):
        """ loops through the molecules and connects the other methods - aka the big guy """
        
        for root in self.rootlist:
        
            print("\nSearching directory: "+root)
            
            for index in range(0,len(self.name_list)):
                
                ## loop over only the 1st occurence of a new name (FLASH try to remove this)
                if index==0 or self.name_list[index] != self.name_list[index-1]:
        
                    self.press_cgs = []    
                    self.lamda_delim=[]
                    self.lamda_mid=[]
                    self.d_lamda=[]
                    self.coeffs=[]
                    self.ystart = []
                    
                    #list with all the opacities for each y value
                    self.ky=[]
                    
                    try:
                        self.read_chebyshev(root,index)
                    except(FileNotFoundError):
                        print("\nsearching...and did not find "+"Out_" + self.name_list[index] + "_cbin.dat")
                        continue
                    
                    self.calc_opac()
                    
                    self.screen_info(index)
        
                    self.conversion()
                
                    self.write_tell(index)
    
    def write_names(self):
        """ stores the names of the included molecules for use in the info file """
       
        self.mol_info = " "
        self.mol_nr_list=self.delete_duplicates(self.mol_nr_list)
        for m in self.mol_nr_list:
            if m != 0:
                self.mol_info += self.mol_name_list[m] + " "
        
        self.cia_info = " "
        self.cia=self.delete_duplicates(self.cia)
        for c in self.cia:
            if c != 0 and c != "-":
                self.cia_info += c + " "
            
    def success(self):
        print("\nSuccessfully produced the k-distribution tables of the individual molecules :)")

if __name__ == "__main__":
    print("This module produces the molecular k-distribution functions from the HELIOS-K output.")
