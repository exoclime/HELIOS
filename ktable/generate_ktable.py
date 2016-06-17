# ==============================================================================
# This program generates the k-table used in HELIOS.
# Copyright (C) 2016 Matej Malik
# 
# Requirements:
# - Following files in the same directory:
#     param_ktable.dat
#     input_param.py
#     resample.py
#     mixing.py
#     chemistry.py
# - Subdirectory ./ktable/
#   For inclusion of water continuum (optional)
#   - Subdirectory ./continuum/ with the file watercontinuum.hdf5 for the water
#     continuum.
# - HELIOS-K output must be accessible and path defined in param_ktable.dat
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

import input_param as ip
import build_opac_general as gen
import build_opac_special as spec
import chemistry as chem
import combination as comb
import rayleigh as ray
import information as inf


def main():
    """ main function to run k-table generation """
    
    # read in the parameter file
    param = ip.Read_param()
    param.read()

    # produce the molecular opacities (general HELIOS output format)
    build_gen = gen.Production()
    build_gen.gen_ypoints()
    build_gen.search_dir(param)
    if param.form == 2:
        build_gen.kick_files()
    build_gen.read_info()
    build_gen.resort()
    build_gen.big_loop()
    build_gen.write_names()
    build_gen.success()

    # produce the molecular opacities (special HELIOS-K output format)
    build_spec = spec.Production()
    if param.form == 2:
        build_spec.gen_ypoints()
        build_spec.search_dir(param)
        build_spec.get_parameters()
        build_spec.resort()
        build_spec.big_loop()
        build_spec.write_names()
        build_spec.success()

    # calculation of chemical abundances
    chem_al = chem.Analyt_core()
    chem_mx = chem.Mix_calc()
    chem_rw = chem.Read_Write()
    chem_rw.read_tp()
    chem_mx.abundances(chem_al, param, chem_rw)
    chem_rw.write_h5(chem_mx)
    chem_rw.success()

    # combination of individual molecular opacities to "mixed" opacity table
    comb_rw = comb.Read_Write()
    comb_comb = comb.Combine()
    comb_rw.read_quant()
    comb_rw.read_cia_continuum()
    comb_rw.read_mix()
    comb_comb.conv_comb(param, comb_rw)
    comb_rw.write_h5(comb_comb)
    comb_rw.success()

    # Rayleigh scattering inclusion
    ray_scat = ray.Rayleigh_scat()
    ray_scat.read()
    ray_scat.calc()
    ray_scat.write()
    ray_scat.success()

    # write information file
    info = inf.Info()
    info.write(param,build_gen,build_spec,comb_rw)

    print("\nDone! Production of k-tables went fine :)")

# run the whole thing
main()


