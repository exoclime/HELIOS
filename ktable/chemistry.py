# =============================================================================
# Python script to compute atmospheric chemistry (C-H-O system including CO2)
# Copyright (C) 2016 Kevin Heng & Matej Malik
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

from numpy import arange,polynomial,interp,exp
import h5py
import warnings

class Analyt_core(object):
    def __init__(self): 
        self.runiv = 8.3144621 # J/K/mol
    
    def kprime(self,my_temperature,pbar):
        """ function to compute first equilibrium constant (K') """
        temperatures = arange(500.0, 3100.0, 100.0)
        dg = [96378.0, 72408.0, 47937.0, 23114.0, -1949.0, -27177.0, -52514.0, -77918.0, -103361.0, -128821.0, -154282.0, -179733.0, -205166.0, -230576.0, -255957.0, -281308.0, -306626.0, -331911.0, -357162.0, -382380.0, -407564.0, -432713.0, -457830.0, -482916.0, -507970.0, -532995.0]
        my_dg = interp(my_temperature,temperatures,dg)
        result = exp(-my_dg/self.runiv/my_temperature)/pbar/pbar
        return result
    
    def kprime2(self,my_temperature):
        """ function to compute second equilibrium constant (K2') """
        temperatures = arange(500.0, 3100.0, 100.0)
        dg2 = [20474.0, 16689.0, 13068.0, 9593.0, 6249.0, 3021.0, -107.0, -3146.0, -6106.0, -8998.0, -11828.0, -14600.0, -17323.0, -20000.0, -22634.0, -25229.0, -27789.0, -30315.0, -32809.0, -35275.0, -37712.0, -40123.0, -42509.0, -44872.0, -47211.0, -49528.0]
        my_dg = interp(my_temperature,temperatures,dg2)
        result = exp(-my_dg/self.runiv/my_temperature)
        return result
    

    def kprime3(self,my_temperature,pbar):
        """ function to compute second equilibrium constant (K3') """
        temperatures = arange(500.0, 3100.0, 100.0)
        dg3 = [262934.0, 237509.0, 211383.0, 184764.0, 157809.0, 130623.0, 103282.0, 75840.0, 48336.0, 20797.0, -6758.0, -34315.0, -61865.0, -89403.0, -116921.0, -144422.0, -171898.0, -199353.0, -226786.0, -254196.0, -281586.0, -308953.0, -336302.0, -363633.0, -390945.0, -418243.0]
        my_dg = interp(my_temperature,temperatures,dg3)
        result = exp(-my_dg/self.runiv/my_temperature)/pbar/pbar
        return result
        
    def n_methane(self,n_o,n_c,temp,pbar):
        """ function to compute mixing ratio for methane, (note: n_o is oxygen abundance, n_c is carbon abundance, kk is K') """
        k1 = self.kprime(temp,pbar)
        k2 = self.kprime2(temp)
        k3 = self.kprime3(temp,pbar)
        a0 = 8.0*k1*k3*k3/k2
        a1 = 8.0*k1*k3/k2
        a2 = 2.0*k1/k2*( 1.0 + 8.0*k3*(n_o-n_c) ) + 2.0*k1*k3
        a3 = 8.0*k1/k2*(n_o-n_c) + 2.0*k3 + k1
        a4 = 8.0*k1/k2*(n_o-n_c)*(n_o-n_c) + 1.0 + 2.0*k1*(n_o-n_c)
        a5 = -2.0*n_c
        result = polynomial.polynomial.polyroots([a5,a4,a3,a2,a1,a0])
        return result[4]   # picks out the correct root of the cubic equation
    
    def n_water(self,n_o,n_c,temp,pbar):
        """ function to compute mixing ratio for water"""
        k3 = self.kprime3(temp,pbar)
        n_ch4 = self.n_methane(n_o,n_c,temp,pbar)
        result = 2.0*k3*n_ch4*n_ch4 + n_ch4 + 2.0*(n_o-n_c)
        return result
    
    def n_cmono(self,n_o,n_c,temp,pbar):
        """ function to compute mixing ratio for carbon monoxide """
        kk = self.kprime(temp,pbar)
        n_ch4 = self.n_methane(n_o,n_c,temp,pbar)
        n_h2o = self.n_water(n_o,n_c,temp,pbar)
        result = kk*n_ch4*n_h2o
        return result
    
    def n_cdio(self,n_o,n_c,temp,pbar):
        """ function to compute mixing ratio for carbon dioxide """
        kk2 = self.kprime2(temp)
        n_h2o = self.n_water(n_o,n_c,temp,pbar)
        n_co = self.n_cmono(n_o,n_c,temp,pbar)
        result = n_co*n_h2o/kk2
        return result
    
    def n_acet(self,n_o,n_c,temp,pbar):
        """ function to compute mixing ratio for acetylene """
        kk3 = self.kprime3(temp,pbar)
        n_ch4 = self.n_methane(n_o,n_c,temp,pbar)
        result = kk3*n_ch4*n_ch4
        return result
    
class Read_Write(object):
    """ class for reading and writing of chemical data """
    def __init__(self):
        self.temp = []
        self.press = []

    def read_tp(self):
        """reads temperature and pressure from water file"""
        
        try:
            with h5py.File("output/h2o_opacities.h5", "r") as file:
                for t in file["temperatures"][:]:
                    self.temp.append(t)
                for p in file["pressures"][:]:
                    self.press.append(p/1e6)
        except(OSError):
            print("ABORT - file \"h2o_opacities.h5\" missing!")
            raise SystemExit()
                
    def write_h5(self,mixed):
        """writes data to hdf5 file"""
        try:
            with h5py.File("output/chemistry_data.h5", "w") as chemfile:
                chemfile.create_dataset("mixratio_h2o", data=mixed.n_h2o)
                chemfile.create_dataset("mixratio_co2", data=mixed.n_co2)
                chemfile.create_dataset("mixratio_co",data=mixed.n_co)
                chemfile.create_dataset("mixratio_ch4",data=mixed.n_ch4)
        except():
            print("ABORT - something wrong with writing to the chemistry_data.h5 file!")
            raise SystemExit()
            
    def success(self):
        """ prints success message """
        print("\nProduction of chemistry mixing ratios --- Successful!")
    
class Mix_calc(object):
    """ class including the main calculation of mixing ratios """
    def __init__(self):
        """

        :rtype: object
        """
        self.n_ch4 = []
        self.n_h2o = []
        self.n_co = []
        self.n_co2 = []
#        self.n_c2h2 = []  ### acetylene not yet implemented
    
    def abundances(self,analyt,readparam,readtp):
        """ calculated the molecular mixing ratios """
        np = len(readtp.press)
        nt = len(readtp.temp)
        n_o = readparam.n_o
        n_c = readparam.n_c
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("complex", Warning)
            
            for t in range(0,nt):
                temperature = readtp.temp[t]
                if readtp.temp[t] < 500:
                    temperature = 500
                for p in range(0,np):
                    pressure = readtp.press[p]
                    water = analyt.n_water(n_o,n_c,temperature,pressure)
                    self.n_h2o.append(water.real)
                    methane = analyt.n_methane(n_o,n_c,temperature,pressure)
                    self.n_ch4.append(methane.real)
                    cmonoxide = analyt.n_cmono(n_o,n_c,temperature,pressure)
                    self.n_co.append(cmonoxide.real)
                    cdioxide = analyt.n_cdio(n_o,n_c,temperature,pressure)
                    self.n_co2.append(cdioxide.real)
#                    self.n_c2h2.append(analyt.n_acet(n_o,n_c,temperature,pressure))
    
if __name__ == "__main__":
    print("This is a module for the analytical calculation of chemical abundances.")