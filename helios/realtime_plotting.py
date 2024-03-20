# ==============================================================================
# Module to plot quantities in realtime during the iteration
# Copyright (C) 2018 - 2022 Matej Malik
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
import matplotlib.animation as anim
import matplotlib.ticker as tkr


class Plot(object):
    """ class including all the plotting utensils """

    def __init__(self):
        self.fig = None
        self.ax_left = None
        self.ax_right = None
        self.ax_right2 = None

    def create_canvas_for_realtime_plotting(self):

        # set up canvas
        self.fig, (self.ax_left, self.ax_right) = plt.subplots(nrows=1, ncols=2, gridspec_kw = {'width_ratios':[5, 4]}, figsize=(10, 5))

        # create 2nd vertical axis
        self.ax_right2 = self.ax_right.twinx()

        # set tight layout (i.e., remove too much padding)
        self.fig.set_tight_layout(tight=True)

        # show plot
        self.fig.canvas.manager.show()

    def plot_tp_and_flux(self, quant):
        """ plots the tp profile and the net flux in realtime """

        # set to 1 for video output
        video = 0

        # left panel
        nr_layer = np.arange(-1, quant.nlayer)

        red_layer = []
        red_temp = []
        conv_layer = []
        conv_temp = []

        for i in range(quant.nlayer + 1):
            if quant.marked_red[i] == 1:
                if i < quant.nlayer:
                    red_layer.append(i)
                elif i == quant.nlayer:
                    red_layer.append(-1)
                red_temp.append(quant.T_lay[i])
            if quant.conv_layer[i] == 1:
                if i < quant.nlayer:
                    conv_layer.append(i)
                elif i == quant.nlayer:
                    conv_layer.append(-1)
                conv_temp.append(quant.T_lay[i])

        temp_plot = np.insert(quant.T_lay[:-1], 0, quant.T_lay[-1])

        if video == 1:
            fig = plt.gcf()
            fig.set_size_inches(10, 6)

        self.ax_left.plot(temp_plot, nr_layer, color='cornflowerblue', linewidth=2)
        self.ax_left.scatter(temp_plot, nr_layer, color='forestgreen', s=30)
        self.ax_left.scatter(red_temp, red_layer, color='red', s=30)
        self.ax_left.scatter(conv_temp, conv_layer, color='orange', s=50)

        self.ax_left.set(ylim=[-1, quant.nlayer-1], ylabel='layer index', xlabel='temperature (K)')

        majorloc_y = tkr.MultipleLocator(10)
        self.ax_left.yaxis.set_major_locator(majorloc_y)

        self.ax_left.xaxis.grid(True, 'minor', color='grey')
        self.ax_left.xaxis.grid(True, 'major', color='grey')
        self.ax_left.yaxis.grid(True, 'minor', color='grey')
        self.ax_left.yaxis.grid(True, 'major', color='grey')

        # self.ax2.set(ylabel=r'pressure (bar)', ylim=[quant.p_boa * 1e-6, quant.p_toa * 1e-6], yscale='log')
        #
        # log10_pboa_bar = int(np.log10(quant.p_boa) - 6)
        # log10_ptoa_bar = int(np.log10(quant.p_toa) - 6)
        #
        # self.ax2.yaxis.set_major_locator(tkr.FixedLocator(locs=np.logspace(log10_pboa_bar, log10_ptoa_bar, log10_pboa_bar - log10_ptoa_bar + 1)))

        # right panel
        nr_interface = np.arange(-1, quant.ninterface)

        fnet_plot = np.insert(quant.F_net, 0, quant.F_intern)

        self.ax_right.plot(fnet_plot, nr_interface, color='cornflowerblue', linewidth=2)
        self.ax_right.scatter(fnet_plot, nr_interface, color='forestgreen', s=30)

        for i in conv_layer:
            self.ax_right.axhspan(i, i + 1, color='orange', alpha=0.5)
        for i in red_layer:
            self.ax_right.axhspan(i, i + 1, color='red', alpha=0.4)

        self.ax_right.set(ylim=[-1, quant.ninterface - 1], ylabel='interface index', xlabel='rad. net flux (erg s$^{-1}$ cm$^{-2}$)')

        self.ax_right.vlines(quant.F_intern, -1, quant.ninterface, colors='blue', linestyles='--', linewidth=2, alpha=0.5)

        if quant.F_intern > 0:
            self.ax_right.set(xlim=[-quant.F_intern/2, quant.F_intern*2])

        majorloc_y = tkr.MultipleLocator(10)
        self.ax_right.yaxis.set_major_locator(majorloc_y)

        self.ax_right.xaxis.grid(True, 'minor', color='grey')
        self.ax_right.xaxis.grid(True, 'major', color='grey')
        self.ax_right.yaxis.grid(True, 'minor', color='grey')
        self.ax_right.yaxis.grid(True, 'major', color='grey')

        self.ax_right2.set(ylabel=r'pressure (bar)', ylim=[quant.p_boa * 1e-6, quant.p_toa * 1e-6], yscale='log')

        log10_pboa_bar = int(np.log10(quant.p_boa) - 6)
        log10_ptoa_bar = int(np.log10(quant.p_toa) - 6)

        self.ax_right2.yaxis.set_major_locator(tkr.FixedLocator(locs=np.logspace(log10_pboa_bar, log10_ptoa_bar, log10_pboa_bar - log10_ptoa_bar + 1)))

        self.fig.canvas.draw()

        if video == 1:
            plt.savefig("./video/radconv_{:0>4d}.png".format(quant.iter_value))

        self.ax_left.clear()
        self.ax_right.clear()
        self.ax_right2.clear()

        self.fig.canvas.flush_events()


if __name__ == "__main__":
    print("This module is for providing realtime graphical output during the iteration. "
          "It has a kind of 'in-your-face' attitude. ")
