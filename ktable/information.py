# =============================================================================
# Module for creating the information file of the ktables
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

class Info(object):
    """ class to produce an information file about the k-table production """
    def __init__(self):
        pass
    
    def write(self,param,build_gen,build_spec,comb):
        """ writes the information file """
        
        try:
            with open("output/ktable_info.dat", "w") as file:
                file.write("""
K - T A B L E   I N F O R M A T I O N
====================================

This is a small program to produce k-tables from HELIOS-K standard output.

/// D A T A   S T R U C T U R E ///

Each H5 file stores the following datasets.

"pressures":                        pressure values used for calculation of the opacities

"temperatures":                     temperature values used for calculation of the opacities

"interface wavelengths":            wavelength at bin interfaces

"centre wavelengths":               wavelength at bin centers

"wavelength width of bins":         width of the bins

"ypoints":                          abszissa points for the 20th order Gauss-Legendre quadrature rule 
                                    (= roots of the 20th order Legendre Polynomial), 
                                    applied to the interval [0,1]. At these points the k-distribution 
                                    function is evaluated.
                                    
"kpoints":                          opacity values in the format: 
                                    opacity\[Y-point, Lambda, Press, Temp\] = kpoints\[y + n_y*l + n_y*n_l*p + n_y*n_l*n_p*t\], 
                                    where n_* the length of the according list and y, l, p, t are the indices 
                                    in the according lists, e.g. Temp = temperatures\[t\], 
                                    Lambda = centre wavelengths\[l\], etc.
                                    
"cross rayleigh":                   Rayleigh scattering cross sections in the format: 
                                    Rayleigh cross section\[Lambda\] = cross_rayleigh\[l\], 
                                    where Lambda = centre wavelengths\[l\].

>>>> All quantities are in cgs-units. The opacity is given in [cm^2 g^-1] and the cross sections in [cm^2]. <<<<


/// P A R A M E T E R   I N F O R M A T I O N ///

""")

                file.writelines("MOLECULAR OPACITY RESAMPLING")
                file.writelines("\nFile path to HELIOS-K output used: "+param.dir)
                if param.form == 2:
                    file.writelines("\nopacity sources: "+build_spec.mol_info+" and CIA: "+build_gen.cia_info)
                else:  
                    file.writelines("\nopacity sources: "+build_gen.mol_info+" and CIA: "+build_gen.cia_info)

                if comb.continuum_incl == 1:
                    file.writelines("\nWater continuum opacities included.")
                else:
                    file.writelines("\nWater continuum opacities NOT included.")
                    file.writelines("\nnumber of temperature points: {:g}".format(len(comb.temp)))
                file.writelines("\nminimum/maximum temperature: {:g}".format(comb.temp[0])+
                                " K / {:g}".format(comb.temp[len(comb.temp)-1])+" K")
                file.writelines("\nnumber of pressure points: {:g}".format(len(comb.press)))
                file.writelines("\nminimum/maximum pressure: {:g}".format(comb.press[0]/1e6)+
                                " bar / {:g}".format(comb.press[len(comb.press)-1]/1e6)+" bar")
                if param.form == 2:
                    file.writelines("\nminimum/maximum wavenumber: {:g}".format(build_spec.numin_tot)+
                                    " cm^-1 / {:g}".format(build_spec.numax_tot)+" cm^-1")
                    file.writelines("\nNote: maximum wavelength readjusted to: 10 cm")
                else:
                    file.writelines("\nminimum/maximum wavenumber: {:g}".format(build_gen.numin)+
                                    " cm^-1 / {:g}".format(build_gen.numax)+" cm^-1")
                file.writelines("\nminimum/maximum wavelength: {:g}".format(comb.h2o_i[0])+
                                " micron / {:g}".format(comb.h2o_i[len(comb.h2o_i)-1])+" micron")
                if param.form != 2:
                    file.writelines("\nresampling resolution (molecules): {:g}".format(build_gen.dnu)+" cm^-1")
                else:
                    file.writelines("\nresampling resolution (molecules): 10^-5 cm^-1")
                if param.form != 2:
                    if build_gen.cutMode == 1:
                        file.writelines("\nlinewing cut off after {:g}".format(build_gen.cut)+" Lorentzian widths ")
                    if build_gen.cutMode == 2:
                        file.writelines("\nlinewing cut off after {:g}".format(build_gen.cut)+" Doppler widths ")
                    if build_gen.cutMode == 0:
                        file.writelines("\nlinewing cut off after {:g}".format(build_gen.cut)+" cm^-1")
                file.writelines("\nnumber of bins: {:g}".format(len(comb.h2o_x)))
                file.writelines("\nnumber of k-points in each bin: {:g}".format(len(comb.h2o_y)))
                file.writelines("\n")
                file.writelines("\n\nCHEMISTRY")
                file.writelines("\nelemental oxygen abundance: {:g}".format(param.n_o))
                file.writelines("\nelemental carbon abundance: {:g}".format(param.n_c))
                file.writelines("\n")
                file.writelines("\n\nMIXING")
                file.writelines("\nmean molecular weight of the atmosphere: {:g}".format(param.mu))
                file.writelines("\n")
                file.writelines("\n\nRAYLEIGH SCATTERING")
                file.writelines("\n Rayleigh scattering is included for hydrogen only.")
                file.write("\n\n\nFor questions or bug reports write to Matej Malik (matej.malik@csh.unibe.ch) :)")
            print("\nInformation file generation --- Successful!")
        except():
            print("Information file generation corrupted. You might want to look into it!")
            

if __name__ == "__main__":
    print("This module is for the information file of the ktables.")
