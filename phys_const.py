# ==============================================================================
# Module with physical constants
# Copyright (C) 2016 Matej Malik
#
# All values are in cgs units.
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

# TODO: how to combine cuda physical constants with those?
PI = 3.14159265359
C_SPEED = 29979245800
K_BOLTZMANN = 1.3806488e-16
H_CONST = 6.62606957e-27
R_UNIV = 8.31446e7
STEFAN_BOLTZMANN = 5.6704e-5
R_SUN = 6.955e10
AU = 1.496e13
M_P = 1.67262e-24

if __name__ == "__main__":
    print("This module provides physical constants. Nothing more, nothing less.")
