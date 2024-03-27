# =============================================================================
# Module for creating the information file of the ktables
# Copyright (C) 2018 - 2022 Matej Malik
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

    @staticmethod
    def convert_1_0_to_yes_no(variable):
        """ converts a variable with value 1 or 0 to a "yes" or "no" message """
        if variable == 1:
            return "yes"
        else:
            return "no"

    def write(self, param):
        """ writes the information file """

        try:

            if param.format == 'k-distribution':

                with open(param.final_path + "opac_table_info.dat", "w") as file:
                    file.write("""
                                K - T A B L E   I N F O R M A T I O N
                                ====================================
                                
                                This is a small program to produce k-tables from HELIOS-K standard output.
                                
                                /// D A T A   S T R U C T U R E ///
                                
                                Each H5 file stores the following datasets.
                                
                                "pressures":                        pressure values used for calculation of the opacities
                                
                                "temperatures":                     temperature values used for calculation of the opacities
                                
                                "interface wavelengths":            wavelength at bin interfaces
                                
                                "center wavelengths":               wavelength at bin centers
                                
                                "wavelength width of bins":         width of the bins
                                
                                "ypoints":                          abszissa points for the 20th order Gauss-Legendre quadrature rule 
                                                                    (= roots of the 20th order Legendre Polynomial), 
                                                                    applied to the interval [0,1]. At these points the k-distribution 
                                                                    function is evaluated.
                                                                    
                                "meanmolmass":                      the mean molecular mass (also called weight) per temperature 
                                                                    and pressure in the format:
                                                                    meanmolmass[Press, Temp] = mu[p + n_p * t], where where n_p is the length of 
                                                                    the pressure list and Press = pressures[p] and Temp = temperatures[t].
                                                                    
                                "kpoints":                          opacity values in the format: 
                                                                    opacity[Y-point, Lambda, Press, Temp] = kpoints[y + n_y*l + n_y*n_l*p + n_y*n_l*n_p*t],
                                                                    where n_* the length of the according list and * = y, l, p, t are the indices
                                                                    in the according lists, e.g. Temp = temperatures[t],
                                                                    Lambda = center wavelengths[l], etc.
                                                                    
                                "weighted Rayleigh cross-sections": Rayleigh scattering cross sections for H2, H, He, H2O, CO2 weighted by their volume mixing ratio in the format:
                                                                    Rayleigh cross section[Lambda, Press, Temp] = cross_rayleigh[l + n_l*p + n_l*n_p*t],
                                                                    where Lambda = wavelengths[l], etc.
                                                
                                "included species":                 List of included opacity sources
                                
                                "FastChem path":                    Path to FastChem output used for the chemical abundances
                                
                                "units":                            'CGS' or 'SI'. 
                                                                    For 'CGS', all units are cgs, namely the opacity unit is cm^2 g^-1, cross sections are in cm^2, 
                                                                    wavelength in cm and pressure in dyne cm^-2 =  1e-6 bar. 
                                                                    For 'SI', all units are in SI, namely opacity is in m^2 kg^-1, cross sections are in m^2, 
                                                                    wavelength in m and pressure in Pa.
                                
                                """)

            elif param.format == 'sampling':

                with open(param.final_path + "opac_table_info.dat", "w") as file:
                    file.write("""
                                O P A C I T Y   I N F O R M A T I O N
                                ====================================

                                This is an opacity table produced from HELIOS-K standard output.

                                /// D A T A   S T R U C T U R E ///

                                Each H5 file stores the following datasets.

                                "pressures":                        pressure values used for calculation of the opacities

                                "temperatures":                     temperature values used for calculation of the opacities

                                "wavelengths":                      wavelengths
                                
                                "meanmolmass":                      the mean molecular mass (also called weight) per temperature 
                                                                    and pressure in the format:
                                                                    meanmolmass[Press, Temp] = mu[p + n_p * t], where where n_p is the length of 
                                                                    the pressure list and Press = pressures[p] and Temp = temperatures[t].

                                "kpoints":                          opacity values in the format: 
                                                                    opacity[Lambda, Press, Temp] = kpoints[l + n_l*p + n_l*n_p*t],
                                                                    where n_* the length of the according list and * = l, p, t are the indices
                                                                    in the according lists, e.g. Temp = temperatures[t],
                                                                    Lambda = wavelengths[l], etc.

                                "weighted Rayleigh cross-sections": Rayleigh scattering cross sections for H2, H, He, H2O, CO2 weighted by their volume mixing ratio in the format:
                                                                    Rayleigh cross section[Lambda, Press, Temp] = cross_rayleigh[l + n_l*p + n_l*n_p*t],
                                                                    where Lambda = wavelengths[l], etc.
                                                
                                "included species":                 List of included opacity sources
                                
                                "FastChem path":                    Path to FastChem output used for the chemical abundances
                                
                                "units":                            'CGS' or 'SI'. 
                                                                    For 'CGS', all units are cgs, namely the opacity unit is cm^2 g^-1, cross sections are in cm^2, 
                                                                    wavelength in cm and pressure in dyne cm^-2 =  1e-6 bar. 
                                                                    For 'SI', all units are in SI, namely opacity is in m^2 kg^-1, cross sections are in m^2, 
                                                                    wavelength in m and pressure in Pa.

                                """)

            print("\nInformation file generation --- Successful!")
        except():
            print("Information file generation corrupted. You might want to look into it!")
            

if __name__ == "__main__":
    print("This module is for the information file of the ktables.")
