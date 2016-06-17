# ==============================================================================
# This is the main file in the HELIOS radiative transfer code.
# Copyright (C) 2016 Matej Malik
#
# To run HELIOS simply run this file with python.
# 
# Requirements:
# - Following files in the same directory:
#   read.py
#   input.dat
#   write.py
#   quantities.py
#   planets_and_stars.py
#   phys_const.py
#   computation.py
#   host_functions.py
#   kernels.cu
# - Subdirectories:
#   /opacities/ - From here the opacity tables are read in.
#   /output/ - This is where the output will be produced.
#   /star/ - With the stellar spectra file (optional)
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

import read
import quantities as quant
import host_functions as hsfunc
import write
import computation as comp
import realtime_plotting as rt_plot


def main():
    """ runs the HELIOS RT computation """

    # instantiate the classes
    reader = read.Read()
    keeper = quant.Store()
    computer = comp.Compute()
    writer = write.Write()
    plotter = rt_plot.Plot()

    # read input files and do preliminary calculations
    reader.read_input(keeper)
    reader.read_opac(keeper)
    keeper.dimensions()
    reader.read_star(keeper)
    hsfunc.gaussian_weights(keeper)
    hsfunc.spec_heat_cap(keeper)
    hsfunc.planet_param(keeper)
    hsfunc.initial_temp(keeper, reader)

    # get ready for GPU computations
    keeper.convert_input_list_to_array()
    keeper.create_zero_arrays()
    keeper.copy_host_to_device()
    keeper.allocate_on_device()

    # conduct the GPU core computations
    computer.construct_planck_table(keeper)
    computer.construct_grid(keeper)
    computer.construct_capital_table(keeper)
    computer.init_spectral_flux(keeper)
    computer.iteration_loop(keeper, writer, plotter)
    computer.calculate_mean_opacities(keeper)
    computer.calculate_transmission(keeper)

    # copy everything back to host and write to files
    keeper.copy_device_to_host()
    writer.write_info(keeper, reader)
    writer.write_tp(keeper)
    writer.write_column_mass(keeper)
    writer.write_integrated_flux(keeper)
    writer.write_downward_spectral_flux(keeper)
    writer.write_upward_spectral_flux(keeper)
    writer.write_planck_interface(keeper)
    writer.write_planck_center(keeper)
    writer.write_opacities(keeper)
    writer.write_transmission(keeper)
    writer.write_mean_extinction(keeper)

    # prints the success message - yay!
    hsfunc.success_message()

main()
