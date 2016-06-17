# =============================================================================
# Module for combining the individual opacity sources
# Copyright (C) 2016 Matej Malik
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

import h5py


class Read_Write(object):
    """ class responsible for reading and writing data """
    
    def __init__(self):
        self.temp = []
        self.press = []
        self.h2o_k = []
        self.h2o_y = []
        self.h2o_w = []
        self.h2o_i = []
        self.h2o_x = []
        self.co2_k = []
        self.co_k = []
        self.ch4_k = []
        self.cia_h2h2_k = []
        self.cia_h2he_k = []
        self.h2o_conti = []
        self.n_h2o=[]
        self.n_co2=[]
        self.n_co= []
        self.n_ch4=[]
        
    def read_quant(self):
        """ read in physical quantities """
        try:
            with h5py.File("output/h2o_opacities.h5", "r") as h2o_file:
            
                for k in h2o_file["kpoints"][:]:
                    self.h2o_k.append(k)
                for y in h2o_file["ypoints"][:]:
                    self.h2o_y.append(y)
                for x in h2o_file["centre wavelengths"][:]:
                    self.h2o_x.append(x)
                for w in h2o_file["wavelength width of bins"][:]:
                    self.h2o_w.append(w)
                for i in h2o_file["interface wavelengths"][:]:
                    self.h2o_i.append(i)
                for t in h2o_file["temperatures"][:]:
                    self.temp.append(t)
                for p in h2o_file["pressures"][:]:
                    self.press.append(p)
        except(OSError):
            print("\nABORT - \"h2o_opacities.h5\" not found!")
            raise SystemExit()
            
        try:
            with h5py.File("output/co2_opacities.h5", "r") as co2_file:
                for c in co2_file["kpoints"][:]:
                    self.co2_k.append(c)
        except(OSError):
            print("\nABORT - \"co2_opacities.h5\" not found!")
            raise SystemExit()
            
        try:
            with h5py.File("output/co_opacities.h5", "r") as co_file:
                for c in co_file["kpoints"][:]:
                    self.co_k.append(c)
        except(OSError):
            print("\nABORT - \"co2_opacities.h5\" not found!")
            raise SystemExit()
            
        try:
            with h5py.File("output/ch4_opacities.h5", "r") as ch4_file:
                for c in ch4_file["kpoints"][:]:
                    self.ch4_k.append(c)
        except(OSError):
            print("\nABORT - \"co2_opacities.h5\" not found!")
            raise SystemExit()

    def read_cia_continuum(self):
        """ read in cia data and water continuum, if they exist """
        
        nx = len(self.h2o_x)
        ny = len(self.h2o_y)
        np = len(self.press)
        nt = len(self.temp)

        try:
            with h5py.File("output/cia_H2-H2_opacities.h5", "r") as cia_h2h2_file:
                for cia in cia_h2h2_file["kpoints"][:]:
                    self.cia_h2h2_k.append(cia)
            print("\nIncluding H2-H2 CIA opacities!")
        except(OSError):
            print("\nNo H2-H2 CIA found. Proceeding without!")
            for i in range(0,ny*nx*np*nt):
                self.cia_h2h2_k.append(0)
        
        try:
            with h5py.File("output/cia_H2-He_opacities.h5", "r") as cia_h2he_file:
                for cia in cia_h2he_file["kpoints"][:]:
                    self.cia_h2he_k.append(cia)
            print("\nIncluding H2-He CIA opacities!")
        except(OSError):
            print("\nNo H2-He CIA found. Proceeding without!")
            for i in range(0,ny*nx*np*nt):
                self.cia_h2he_k.append(0)
    
        ## the use of water continuum is still experimental and should be neglected for now.
        try:
            with h5py.File("continuum/watercontinuum.h5", "r") as conti_file:
                if nx == 300:
                    for c in conti_file["300"][:]:
                        self.h2o_conti.append(c)
                if nx == 3000:
                    for c in conti_file["3000"][:]:
                        self.h2o_conti.append(c)
            print("\nIncluding water continuum!")
            self.continuum_incl = 1
        except(OSError):
            # print("\nNo water continuum file found. Proceeding without the continuum!")
            self.continuum_incl = 0
            for i in range(nx*np*nt):
                self.h2o_conti.append(0)


    def read_mix(self):
        """ read in the mixing ratios """
        
        try:
            with h5py.File("output/chemistry_data.h5", "r") as chemfile:
                for n in chemfile["mixratio_h2o"][:]:
                    self.n_h2o.append(n)
                for n in chemfile["mixratio_co2"][:]:
                    self.n_co2.append(n)
                for n in chemfile["mixratio_co"][:]:
                    self.n_co.append(n)
                for n in chemfile["mixratio_ch4"][:]:
                    self.n_ch4.append(n)
        except(OSError):
            print("\nABORT - chemistry_data.h5 not found!")
            raise SystemExit()
            
    def write_h5(self,comb):
        """ write to hdf5 file """
    
        try:
            with h5py.File("output/mixed_opacities.h5", "w") as mixed_file:
                mixed_file.create_dataset("pressures", data=self.press)
                mixed_file.create_dataset("temperatures", data=self.temp)
                mixed_file.create_dataset("interface wavelengths",data=self.h2o_i)
                mixed_file.create_dataset("centre wavelengths",data=self.h2o_x)
                mixed_file.create_dataset("wavelength width of bins",data=self.h2o_w)
                mixed_file.create_dataset("ypoints",data=self.h2o_y)
                mixed_file.create_dataset("kpoints",data=comb.mixed_k)
        except():
                print("ABORT - something wrong with writing to the mixed_opacities.h5 file!")
                raise SystemExit()

    def success(self):
        """ prints success message """
        print("\nCombination of opacities --- Successful!")
            
class Combine(object):
    """ class to combine the individual molecular opacities """
    
    def __init__(self):
        self.mix_h2o=[]
        self.mix_co2=[]
        self.mix_co=[]
        self.mix_ch4=[]
        self.mix_h2h2=[]
        self.mix_h2he=[]
        self.mixed_k = []
    
    def conv_comb(self,readparam,rw):
        """ convert volume to mass mixing ratios and combine opacities """
        
        mu = readparam.mu
        nt = len(rw.temp)
        np = len(rw.press)
        nx = len(rw.h2o_x)
        ny = len(rw.h2o_y)
                
        for n in range(0,len(rw.n_h2o)):
            self.mix_h2o.append(rw.n_h2o[n]*18.0/mu)
            self.mix_co2.append(rw.n_co2[n]*44.0/mu)
            self.mix_co.append(rw.n_co[n]*28.0/mu)
            self.mix_ch4.append(rw.n_ch4[n]*16.0/mu)
            self.mix_h2h2.append(1.0*2.0/mu)
            self.mix_h2he.append(0.1*4.0/mu)
            
        for t in range(0,nt):
            for p in range(0,np):
                for x in range(0,nx):
                    for y in range(0,ny):
                        mix = self.mix_h2o[p+np*t] * rw.h2o_k[y+ny*x+ny*nx*p+ny*nx*np*t] \
                        + self.mix_h2o[p+np*t] * rw.h2o_conti[x+nx*p+nx*np*t] \
                        + self.mix_co2[p+np*t] * rw.co2_k[y+ny*x+ny*nx*p+ny*nx*np*t] \
                        + self.mix_co[p+np*t] * rw.co_k[y+ny*x+ny*nx*p+ny*nx*np*t] \
                        + self.mix_ch4[p+np*t] * rw.ch4_k[y+ny*x+ny*nx*p+ny*nx*np*t] \
                        + self.mix_h2h2[p+np*t] * rw.cia_h2h2_k[y+ny*x+ny*nx*p+ny*nx*np*t] \
                        + self.mix_h2he[p+np*t] * rw.cia_h2he_k[y+ny*x+ny*nx*p+ny*nx*np*t]
                        self.mixed_k.append(mix)

if __name__ == "__main__":
    print("This module is for the combination of the individual molecular opacities.")