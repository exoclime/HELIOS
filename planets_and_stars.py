# ==============================================================================
# Module with dictionaries of planets
# Copyright (C) 2016 Matej Malik
#
# For each planet there is a dictionary with the parameters as entries.
# WARNING: Only works if the planets have the same index in planet_list and dict_list.
#
# All values are in cgs units.
#
# Also added a list of all planets, for whom we provide the Kurucz and Phoenix
# stellar spectra.
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

# dictionaries of the planetary parameters
gj1214b = {'g': 768, 'a': 0.01411, 'R_star': 0.211, 'T_star': 3252}

gj436b = {'g': 1318, 'a': 0.03, 'R_star': 0.455, 'T_star': 3416}

hd189733b = {'g': 1950, 'a': 0.03142, 'R_star': 0.805, 'T_star': 5050}

wasp8b = {'g': 5510, 'a': 0.0801, 'R_star': 0.945, 'T_star': 5600}

wasp12b = {'g': 1164, 'a': 0.02293, 'R_star': 1.599, 'T_star': 6300}

wasp14b = {'g': 10233, 'a': 0.036, 'R_star': 1.306, 'T_star': 6475}

wasp33b = {'g': 2884, 'a': 0.0259, 'R_star': 1.509, 'T_star': 7430}

wasp43b = {'g': 4699, 'a': 0.0152, 'R_star': 0.667, 'T_star': 4520}

# lists the name in the input file
planet_list = ["GJ1214b", "GJ436b", "HD189733b",
               "WASP-8b", "WASP-12b", "WASP-14b", "WASP-33b", "WASP-43b"]

# lists the according dictionary
dict_list = [gj1214b, gj436b, hd189733b, wasp8b, wasp12b, wasp14b, wasp33b, wasp43b]

planets_with_stellar_spectra = ["HD189733b", "WASP-8b", "WASP-12b", "WASP-14b", "WASP-33b", "WASP-43b"]

if __name__ == "__main__":
    print("This module is for planetary data. There is not much else to do.")
