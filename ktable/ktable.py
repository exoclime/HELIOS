# ==============================================================================
# This program generates the opacity table used in HELIOS.
# Copyright (C) 2018 Matej Malik
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

from source import param as para
from source import build_opac_ktable as bok
from source import build_opac_sampling as bos
from source import combination as comb
from source import rayleigh as ray
from source import continuous as cont
from source import information as inf
from source import condensation as condens


def main():
    """ main function to run k-table generation """

    # create objects of classes
    param = para.Param()
    ktable_builder = bok.Production()
    sampling_builder = bos.Production()
    comber = comb.Comb()
    scatter = ray.Rayleigh_scat()
    conti = cont.ContiClass()
    cond = condens.Condense()
    info = inf.Info()

    # read in the parameter file
    param.read_param_file()

    # generate ktables / downsample HELIOS-K output
    if param.building == "yes":

        if param.format == "ktable":
            # ktable_builder.fix_exomol_name(param)  # in case the files have the wrong name one can use this script to fix that
            ktable_builder.gen_ypoints()
            ktable_builder.search_dir(param)
            ktable_builder.get_parameters()
            ktable_builder.resort()
            ktable_builder.big_loop(param)
            ktable_builder.write_names()
            ktable_builder.success()

        elif param.format == "sampling":

            sampling_builder.read_param_sampling(param)
            sampling_builder.initialize_wavelength_grid(param)
            sampling_builder.set_up_press_dict()
            sampling_builder.big_loop(param)
            sampling_builder.success()

    # combine individual opacities and weight with their equilibrium abundances obtained from FastChem
    # it starts with water (water is always necessary) and then adds more species
    comber.combine_all_species(param, scatter, cond, conti)

    comber.success()

    # write information file
    info.write(param)

    print("\nDone! Production of k-tables went fine :)")

# run the whole thing
main()


