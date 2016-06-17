# =============================================================================
# Module for computing Rayleigh scattering opacities
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

import numpy as np
import h5py

class Rayleigh_scat(object):
    """ class providing the Rayleigh scattering calculation"""
    
    def __init__(self):
        self.lamda = []
        self.cross_ray_h2 = []
        self.King_h2 = 1
        self.n_ref_h2 = 2.68678e19
        self.cross_ray_he = []
        self.King_he = 1
        self.n_ref_he = 2.546899e19
    
    def read(self):
        """ reads in the wavelength data"""
        try:
            with h5py.File("output/mixed_opacities.h5", "r") as fmix:
                for x in fmix["centre wavelengths"][:]:
                    self.lamda.append(x)
        except(OSError):
            print("ABORT - file \"mixed_opacities.h5\" missing!")
            raise SystemExit()
                
    def write(self): 
        """ writes the scattering cross sections to the hdf5 file"""
        with h5py.File("output/mixed_opacities.h5", "a") as fmix:
            try:
                fmix.create_dataset("cross_rayleigh", data=self.cross_ray_h2)
            except(RuntimeError):
                del fmix["cross_rayleigh"]
                fmix.create_dataset("cross_rayleigh", data=self.cross_ray_h2)
    
    def index_h2(self,lam):
        """ calculates the refractive index for hydrogen """
        result = 13.58e-5 * (1 + 7.52e-11 * lam**-2) + 1
        return result
    
    def index_he(self,lam):
        """ calculates the refractive index for helium """
        result = 1e-8 * (2283 + 1.8102e13/(1.5342e10 - lam**-2)) + 1
        return result
        
    def cross_sect(self, lamda, index, n_ref, King):
        """ calculates the scattering cross sections """
        result = 24.0*np.pi**3/(n_ref**2*lamda**4)*((index**2-1.0)/(index**2+2.0))**2*King
        return result

    def calc(self):
        """ applies the calculation to a wavelength range and saves as array """
        for lam in self.lamda:
            cross_sect = self.cross_sect(lam, self.index_h2(lam),self.n_ref_h2,self.King_h2)
            self.cross_ray_h2.append(cross_sect)
    
    def success(self):
        """ prints success message """
        print("\nInclusion of Rayleigh scattering cross sections --- Successful!")

if __name__ == "__main__":
    print("This module is for the computation of Rayleigh scattering cross sections.")
