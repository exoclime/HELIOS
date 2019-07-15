# ==============================================================================
# Module to plot quantities in realtime during the iteration
# Copyright (C) 2018 Matej Malik
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
import matplotlib.ticker as tkr


class Plot(object):
    """ class including all the plotting utensils """

    def __init__(self):
        pass

    @staticmethod
    def plot_tp(quant):
        """ plots the TP-profile in realtime """

        # set to 1 for video output
        video = 0

        nr_layer = np.arange(0, quant.nlayer)

        red_layer = []
        red_temp = []

        for i in range(quant.nlayer):
            if quant.marked_red[i] == 1:
                red_layer.append(i)
                red_temp.append(quant.T_lay[i])

        plt.ion()

        if video == 1:
            fig=plt.gcf()
            fig.set_size_inches(10,6)

        plt.plot(quant.T_lay, nr_layer, color='cornflowerblue', linewidth=2.0)
        plt.scatter(quant.T_lay, nr_layer, color='forestgreen', s=40)
        plt.scatter(red_temp, red_layer, color='red', s=40)

        plt.ylim(0, quant.nlayer-1)
        if video == 1:
            plt.xlim(800, 3600)

        majorloc_y = tkr.MultipleLocator(10)
        plt.axes().yaxis.set_major_locator(majorloc_y)

        plt.xlabel('temperature (K)')
        plt.ylabel('number of layer')

        plt.axes().xaxis.grid(True, 'minor', color='grey')
        plt.axes().xaxis.grid(True, 'major', color='grey')
        plt.axes().yaxis.grid(True, 'minor', color='grey')
        plt.axes().yaxis.grid(True, 'major', color='grey')
        plt.show()

        if video == 1:
            plt.savefig("./video/rad_{:0>4d}.png".format(quant.iter_value))

        plt.pause(0.001)
        plt.clf()

    @staticmethod
    def plot_convective_feedback(quant):
        """ plots the tp profile and the net flux in realtime """

        # set to 1 for video output
        video = 0

        nr_layer = np.arange(0, quant.nlayer)
        nr_interface = np.arange(0, quant.ninterface)

        conv_list = []
        red_list = []
        for i in range(quant.nlayer):
            if quant.conv_layer[i] == 1:
                conv_list.append(i)
            if quant.marked_red[i] == 1:
                red_list.append(i)

        # prepare figure
        plt.ion()
        fig = plt.gcf()

        if video == 1:
            fig.set_size_inches(10, 6)

        # left subplot
        subtp = fig.add_subplot(121)

        subtp.plot(quant.T_lay, nr_layer, color='cornflowerblue', linewidth=2.0)
        subtp.scatter(quant.T_lay, nr_layer, color='orangered', s=20)

        plt.ylim(0, quant.nlayer-1)
        # plt.xlim(800, 3600)

        #x_low_lim = max(0, 100*(min(quant.T_lay) // 100 - 1))
        #x_up_lim = 100*(max(quant.T_lay) // 100 + 2)
        #plt.xlim(x_low_lim, x_up_lim)
        plt.xlabel('temperature (K)')
        plt.ylabel('number of layer')
        #plt.xticks(np.arange(x_low_lim, x_up_lim+400, 400))

        #minorloc_x = tkr.MultipleLocator(50)
        #subtp.xaxis.set_minor_locator(minorloc_x)
        majorloc_y = tkr.MultipleLocator(10)
        subtp.yaxis.set_major_locator(majorloc_y)

        subtp.xaxis.grid(True, 'minor', color='grey')
        subtp.xaxis.grid(True, 'major', color='grey')
        subtp.yaxis.grid(True, 'minor', color='grey')
        subtp.yaxis.grid(True, 'major', color='grey')

        p_plot = [p * 1e-6 for p in quant.p_int]

        # right subplot
        subfnet = fig.add_subplot(122)
        subfnet.plot(quant.F_net, nr_interface, color='forestgreen', linewidth=2.0)
        subfnet.scatter(quant.F_net, nr_interface, color='orangered', s=20)

        for i in conv_list:
            subfnet.axhspan(i, i + 1, color='orange', alpha=0.5)
        for i in red_list:
            subfnet.axhspan(i, i + 1, color='magenta', alpha=0.5)

        plt.ylim(0, quant.ninterface - 1)

        plt.vlines(quant.F_intern, 0, quant.ninterface, colors='blue', linestyles='--', linewidth=2.0, alpha=0.5)

        if quant.F_intern > 0:
            plt.xlim(-quant.F_intern/2, quant.F_intern*2)

        if video == 1:
            plt.xlim(0, 40000)

        plt.xlabel('rad. net flux (erg s$^{-1}$ cm$^{-2}$)')
        plt.ylabel('number of interface')

        majorloc_y = tkr.MultipleLocator(10)
        subfnet.yaxis.set_major_locator(majorloc_y)

        subfnet.xaxis.grid(True, 'minor', color='grey')
        subfnet.xaxis.grid(True, 'major', color='grey')
        subfnet.yaxis.grid(True, 'minor', color='grey')
        subfnet.yaxis.grid(True, 'major', color='grey')


        # show and close figure
        plt.show()

        # for debugging uncomment the next line
        # input("\n\nplease hit enter\n\n")

        if video == 1:
            plt.savefig("./video/radconv_{:0>4d}.png".format(quant.iter_value))

        plt.pause(0.001)
        plt.clf()


if __name__ == "__main__":
    print("This module is for providing realtime graphical output during the iteration. "
          "It has a kind of 'in-your-face' attitude. ")
