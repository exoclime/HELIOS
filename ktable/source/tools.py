# ==============================================================================
# Module with some useful tools i.e. helper functions
# Copyright (C) 2018 Matej Malik
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

import sys


def percent_counter(z, nz, y=0, ny=1, x=0, nx=1):
    """ displays percent done of long operation for two for entwined loops """

    percentage = float((x + nx * y + nx * ny * z) / (nx * ny * nz) * 100.0)
    sys.stdout.write("calculating: {:.1f}%\r".format(percentage))
    sys.stdout.flush()

if __name__ == "__main__":
    print("Unlike a classic tool box which always hides just the tool you need, this one actually tries to be helpful...for now.")