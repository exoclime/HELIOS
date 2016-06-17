# ==============================================================================
# Module with plotting script to show quantities in realtime during the iteration
# Copyright (C) 2016 Matej Malik
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class Plot(object):
    """ class including all the plotting utensils """

    def __init__(self):
        pass

    def plot_tp(self, quant):
        """ plots the TP-profile in realtime """

        nr_layer = np.arange(0, quant.nlayer)

        plt.ion()
        plt.plot(quant.T_lay, nr_layer, color='cornflowerblue', linewidth=2.0)
        plt.scatter(quant.T_lay, nr_layer, color='orangered', s=20)
        plt.ylim(0, quant.nlayer-1)
        plt.xlim(200, 3800)
        plt.xticks(np.arange(200, 4200, 400))
        minorloc = MultipleLocator(50)
        plt.axes().xaxis.set_minor_locator(minorloc)
        plt.xlabel('temperature [K]')
        plt.ylabel('number of layer')
        plt.axes().xaxis.grid(True, 'minor', color='grey')
        plt.axes().xaxis.grid(True, 'major', color='grey')
        plt.axes().yaxis.grid(True, 'minor', color='grey')
        plt.axes().yaxis.grid(True, 'major', color='grey')
        plt.show()
        plt.pause(0.001)
        plt.clf()


if __name__ == "__main__":
    print("This module is for plooting realtime stuff during the iteration. ")
