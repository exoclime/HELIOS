# ==============================================================================
# This program generates the opacity table used in HELIOS.
# Copyright (C) 2018 - 2022 Matej Malik
#
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
import sys

# including the Helios main directory into Python paths in order to read files from 'helios/source'
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir)

from source_ktable import param as para
from source_ktable import build_individual_opacities as bio
from source_ktable import combination as comb
from source_ktable import rayleigh as ray
from source_ktable import continuous as cont
from source_ktable import information as inf


def main():
    """ main function to run k-table generation """

    # create objects of classes
    param = para.Param()
    opacity_builder = bio.Production()
    comber = comb.Comb()
    scatter = ray.Rayleigh_scat()
    conti = cont.ContiClass()
    info = inf.Info()

    # read in the parameter file
    param.read_param_file_and_command_line()

    # 1st stage -- generate opacity files for individual species
    if param.building == "yes":

        opacity_builder.read_individual_species_file(param)
        opacity_builder.initialize_wavelength_grid(param)
        opacity_builder.set_up_press_dict()
        opacity_builder.big_loop(param)
        opacity_builder.success()

    # 2nd stage -- combine individual opacities weighted by their mixing ratio to mixed opacity table
    if param.mixing == "yes":

        comber.combine_all_species(param, scatter, conti)
        comber.success()
        # write information file
        info.write(param)

    print("\nDone! The 'ktable' program finished successfully!")

# run the whole thing
main()


