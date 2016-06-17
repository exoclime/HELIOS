# ==============================================================================
# Module for the core computational part of HELIOS.
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
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


class Compute(object):
    """ class incorporating the computational core of HELIOS """

    def __init__(self):
        self.kernel_file = open("kernels.cu")
        self.kernels = self.kernel_file.read()
        self.mod = SourceModule(self.kernels)

    def construct_planck_table(self, quant):
        """ constructs the Planck table """

        plancktable = self.mod.get_function("plancktable")

        plancktable(quant.dev_planckband_grid, quant.dev_opac_interwave, quant.dev_opac_deltawave,
                    quant.nbin, quant.T_star, quant.T_intern,
                    block=(16, 16, 1), grid=((int(quant.nbin)+15)//16, (402+15)//16, 1))

        cuda.Context.synchronize()

    def construct_grid(self, quant):
        """ constructs the atmospheric grid """

        gridkernel = self.mod.get_function("gridkernel")

        gridkernel(quant.dev_p_lay, quant.dev_p_int, quant.dev_T_lay, quant.dev_delta_colmass,
                   quant.dev_delta_colupper, quant.dev_delta_collower,
                   quant.p_boa, quant. p_toa, quant.nlayer, quant.g,
                   block=(16, 1, 1), grid=((int(quant.nlayer)+15)//16, 1, 1))

        cuda.Context.synchronize()

    def construct_capital_table(self, quant):
        """ constructs the table of the capital letters terms """

        if quant.tabu == 1:
            captable = self.mod.get_function("captable")

            captable(quant.dev_Mterm_grid, quant.dev_Nterm_grid, quant.dev_Pterm_grid, quant.dev_Qterm_grid,
                     quant.dev_delta_colmass, quant.dev_cross_scat, quant.epsilon, quant.nlayer, quant.nbin,
                     quant.scat, quant.meanmolmass, quant.g_0,
                     block=(16, 16, 1), grid=((190+15)//16, (int(quant.nbin)+15)//16, int(quant.nlayer)))

            if quant.iso == 0:
                captable(quant.dev_Mterm_uppergrid, quant.dev_Nterm_uppergrid, quant.dev_Pterm_uppergrid,
                         quant.dev_Qterm_uppergrid, quant.dev_delta_colupper, quant.dev_cross_scat, quant.epsilon,
                         quant.nlayer, quant.nbin, quant.scat, quant.meanmolmass, quant.g_0,
                         block=(16, 16, 1), grid=((190+15)//16, (int(quant.nbin)+15)//16, int(quant.nlayer)))

                captable(quant.dev_Mterm_lowergrid, quant.dev_Nterm_lowergrid, quant.dev_Pterm_lowergrid,
                         quant.dev_Qterm_lowergrid, quant.dev_delta_collower, quant.dev_cross_scat, quant.epsilon,
                         quant.nlayer, quant.nbin, quant.scat, quant.meanmolmass, quant.g_0,
                         block=(16, 16, 1), grid=((190+15)//16, (int(quant.nbin)+15)//16, int(quant.nlayer)))

            cuda.Context.synchronize()

    def init_spectral_flux(self, quant):
        """ initializes the spectral flux arrays """

        flux_init = self.mod.get_function("flux_init")

        flux_init(quant.dev_Fdown_wg_band, quant.dev_Fup_wg_band, quant.dev_Fc_down_wg_band, quant. dev_Fc_up_wg_band,
                  quant.ny, quant.nbin, quant.ninterface,
                  block=(16, int(quant.ny), 1), grid=((int(quant.nbin)+15)//16, 1, int(quant.ninterface)))

        cuda.Context.synchronize()

    def interpolate_temperatures(self, quant):
        """ interpolates the layer temperatures to the interfaces """

        temp_inter = self.mod.get_function("temp_inter")

        temp_inter(quant.dev_T_lay, quant.dev_T_int, quant.ninterface,
                   block=(16, 1, 1), grid=((int(quant.ninterface)+15)//16, 1, 1))

        cuda.Context.synchronize()

    def interpolate_opacities(self, quant):
        """ builds the layer and interface opacities by interpolating the values from the opacity table """

        opac_interpol = self.mod.get_function("opac_interpol")

        opac_interpol(quant.dev_T_lay, quant.dev_ktemp, quant.dev_p_lay, quant.dev_kpress, quant.dev_opac_k,
                      quant.dev_opac_wg_lay, quant.dev_opac_y, quant.npress, quant.ntemp, quant.ny, quant.nbin,
                      quant.fake_opac, quant.nlayer,
                      block=(16, 16, 1), grid=((int(quant.nbin)+15)//16, (int(quant.nlayer)+15)//16, 1))

        cuda.Context.synchronize()

        if quant.iso == 0:
            opac_interpol(quant.dev_T_int, quant.dev_ktemp, quant.dev_p_int, quant.dev_kpress, quant.dev_opac_k,
                          quant.dev_opac_wg_int, quant.dev_opac_y, quant.npress, quant.ntemp, quant.ny, quant.nbin,
                          quant.fake_opac, quant.ninterface,
                          block=(16, 16, 1), grid=((int(quant.nbin) + 15) // 16, (int(quant.ninterface) + 15) // 16, 1))

        cuda.Context.synchronize()

    def interpolate_cap_letter_terms(self, quant):
        """ interpolates the layer and interface capital letter terms from the pre-tabulated values """

        if quant.tabu == 1:
            if quant.iso == 1:
                cap_interpol_iso = self.mod.get_function("cap_interpol_iso")

                cap_interpol_iso(quant.dev_Mterm_grid, quant.dev_Nterm_grid, quant.dev_Pterm_grid, quant.dev_Qterm_grid,
                                 quant.dev_Mterm, quant.dev_Nterm, quant.dev_Pterm, quant.dev_Qterm,
                                 quant.dev_opac_wg_lay, quant.nlayer, quant.nbin, quant.ny,
                                 block=(16, 16, 1), grid=((int(quant.nbin)+15)//16, (int(quant.nlayer)+15)//16, 1))

            elif quant.iso == 0:
                cap_interpol_noniso = self.mod.get_function("cap_interpol_noniso")

                cap_interpol_noniso(quant.dev_Mterm_uppergrid, quant.dev_Nterm_uppergrid, quant.dev_Pterm_uppergrid,
                                    quant.dev_Qterm_uppergrid, quant.dev_Mterm_lowergrid, quant.dev_Nterm_lowergrid,
                                    quant.dev_Pterm_lowergrid, quant.dev_Qterm_lowergrid, quant.dev_Mterm_upper,
                                    quant.dev_Nterm_upper, quant.dev_Pterm_upper, quant.dev_Qterm_upper,
                                    quant.dev_Mterm_lower, quant.dev_Nterm_lower, quant.dev_Pterm_lower,
                                    quant.dev_Qterm_lower, quant.dev_opac_wg_lay, quant.dev_opac_wg_int,
                                    quant.dev_cross_scat, quant.nlayer, quant.nbin, quant.ny,
                                    block=(16, 16, 1), grid=((int(quant.nbin)+15)//16, (int(quant.nlayer)+15)//16, 2))

            cuda.Context.synchronize()

    def interpolate_planck(self, quant):
        """ interpolates the pre-tabulated Planck function to the layer/interface values
        plus stellar and internal blackbody """

        planck_interpol_layer = self.mod.get_function("planck_interpol_layer")

        planck_interpol_layer(quant.dev_opac_wave, quant.dev_opac_interwave, quant.dev_opac_deltawave, quant.dev_T_lay,
                              quant.dev_planckband_lay, quant.dev_planckband_grid, quant.dev_starflux, quant.real_star,
                              quant.nlayer, quant.nbin,
                              block=(16, 16, 1), grid=((int(quant.nbin)+15)//16, (int(quant.ninterface+1)+15)//16, 1))

        cuda.Context.synchronize()

        if quant.iso == 0:
            planck_interpol_interface = self.mod.get_function("planck_interpol_interface")

            planck_interpol_interface(quant.dev_opac_wave, quant.dev_opac_interwave, quant.dev_opac_deltawave,
                                      quant.dev_T_int, quant.dev_planckband_int, quant.dev_planckband_grid,
                                      quant.ninterface, quant.nbin,
                                      block=(16, 16, 1),
                                      grid=((int(quant.nbin)+15)//16, (int(quant.ninterface+1)+15)//16, 1))

        cuda.Context.synchronize()

    def populate_spectral_flux(self, quant):
        """ populates the down- and upstream spectral fluxes """

        nscat_step = None
        if quant.singlewalk == 0:
            nscat_step = 4
        if quant.singlewalk == 1:
            nscat_step = 80

        for scat_iter in range(nscat_step * quant.scat + 1):

            if quant.iso == 1:
                if quant.tabu == 1:

                    fband_iso_tabu = self.mod.get_function("fband_iso_tabu")

                    fband_iso_tabu(quant.dev_Fdown_wg_band, quant.dev_Fup_wg_band, quant.dev_planckband_lay,
                                   quant.dev_Mterm, quant.dev_Nterm, quant.dev_Pterm, quant.dev_Qterm,
                                   quant.dev_delta_colmass, quant.dev_opac_wg_lay, quant.dev_cross_scat, quant.scat,
                                   quant.singlewalk, quant.meanmolmass, quant.R_star, quant.a, quant.g_0,
                                   quant.ninterface, quant.nbin, quant.f_factor, quant.ny, quant.epsilon,
                                   block=(16, int(quant.ny), 1), grid=((int(quant.nbin)+15)//16, 1, 1))

                if quant.tabu == 0:

                    fband_iso_notabu = self.mod.get_function("fband_iso_notabu")

                    fband_iso_notabu(quant.dev_Fdown_wg_band, quant.dev_Fup_wg_band, quant.dev_planckband_lay,
                                     quant.dev_delta_colmass, quant.dev_opac_wg_lay, quant.dev_cross_scat, quant.scat,
                                     quant.singlewalk, quant.meanmolmass, quant.R_star, quant.a, quant.g_0,
                                     quant.ninterface, quant.nbin, quant.f_factor, quant.ny, quant.epsilon,
                                     block=(16, int(quant.ny), 1), grid=((int(quant.nbin)+15)//16, 1, 1))

                if quant.tabu == 2:

                    fband_iso_direct = self.mod.get_function("fband_iso_direct")

                    fband_iso_direct(quant.dev_Fdown_wg_band, quant.dev_Fup_wg_band, quant.dev_planckband_lay,
                                     quant.dev_delta_colmass, quant.dev_opac_wg_lay, quant.R_star, quant.a,
                                     quant.singlewalk, quant.ninterface, quant.nbin, quant.f_factor, quant.ny,
                                     block=(16, int(quant.ny), 1), grid=((int(quant.nbin)+15)//16, 1, 1))

            if quant.iso == 0:
                if quant.tabu == 1:

                    fband_noniso_tabu = self.mod.get_function("fband_noniso_tabu")

                    fband_noniso_tabu(quant.dev_Fdown_wg_band, quant.dev_Fup_wg_band, quant.dev_Fc_down_wg_band,
                                      quant.dev_Fc_up_wg_band, quant.dev_planckband_lay, quant.dev_planckband_int,
                                      quant.dev_Mterm_upper, quant.dev_Nterm_upper, quant.dev_Pterm_upper,
                                      quant.dev_Qterm_upper, quant.dev_Mterm_lower, quant.dev_Nterm_lower,
                                      quant.dev_Pterm_lower, quant.dev_Qterm_lower, quant.dev_delta_colupper,
                                      quant.dev_delta_collower, quant.dev_opac_wg_lay, quant.dev_opac_wg_int,
                                      quant.dev_cross_scat, quant.scat, quant.singlewalk, quant.meanmolmass,
                                      quant.R_star, quant.a, quant.g_0, quant.epsilon, quant.ninterface, quant.nbin,
                                      quant.f_factor, quant.ny,
                                      block=(16, int(quant.ny), 1), grid=((int(quant.nbin)+15)//16, 1, 1)
                                      )

                if quant.tabu == 0:

                    fband_noniso_notabu = self.mod.get_function("fband_noniso_notabu")

                    fband_noniso_notabu(quant.dev_Fdown_wg_band, quant.dev_Fup_wg_band, quant.dev_Fc_down_wg_band,
                                        quant.dev_Fc_up_wg_band, quant.dev_planckband_lay, quant.dev_planckband_int,
                                        quant.dev_delta_colupper, quant.dev_delta_collower, quant.dev_opac_wg_lay,
                                        quant.dev_opac_wg_int, quant.dev_cross_scat, quant.scat, quant.singlewalk,
                                        quant.meanmolmass, quant.R_star, quant.a, quant.g_0, quant.epsilon,
                                        quant.ninterface, quant.nbin, quant.f_factor, quant.ny,
                                        block=(16, int(quant.ny), 1), grid=((int(quant.nbin)+15)//16, 1, 1))

                if quant.tabu == 2:

                    fband_noniso_direct = self.mod.get_function("fband_noniso_direct")

                    fband_noniso_direct(quant.dev_Fdown_wg_band, quant.dev_Fup_wg_band, quant.dev_Fc_down_wg_band,
                                        quant.dev_Fc_up_wg_band, quant.dev_planckband_lay, quant.dev_planckband_int,
                                        quant.dev_delta_colupper, quant.dev_delta_collower, quant.dev_opac_wg_lay,
                                        quant.dev_opac_wg_int, quant.R_star, quant.a, quant.singlewalk,
                                        quant.ninterface, quant.nbin, quant.f_factor, quant.ny,
                                        block=(16, int(quant.ny), 1), grid=((int(quant.nbin)+15)//16, 1, 1))

            cuda.Context.synchronize()

    def integrate_flux(self, quant):
        """ integrates the spectral fluxes first over each bin and then the whole spectral range """

        flux_integrate = self.mod.get_function("flux_integrate")

        flux_integrate(quant.dev_opac_deltawave, quant.dev_Fdown_tot, quant.dev_Fup_tot, quant.dev_Fdown_wg_band,
                       quant.dev_Fup_wg_band, quant.dev_Fdown_band, quant.dev_Fup_band, quant.dev_opac_weight,
                       quant.nbin, quant.ninterface, quant.ny,
                       block=(16, 1, 1), grid=((int(quant.ninterface)+15)//16, 1, 1))

        cuda.Context.synchronize()

    def net_flux_and_temp_iteration(self, quant):
        """ calculates the net flux and advances the layer temperature """

        netfluxandtempiter = self.mod.get_function("netfluxandtempiter")

        netfluxandtempiter(quant.dev_Fdown_tot, quant.dev_Fup_tot, quant.dev_F_net, quant.dev_T_lay, quant.dev_p_lay,
                           quant.dev_T_int, quant.dev_p_int, quant.dev_abort, quant.dev_delta_t_lay,
                           quant.dev_delta_T_store, quant.dev_delta_t_prefactor, quant.iter_value, quant.f_factor,
                           quant.foreplay, quant.meanmolmass, quant.R_star, quant.T_star, quant.g, quant.a, quant.tstep,
                           quant.nlayer, quant.varying_tstep, quant.c_p,
                           block=(16, 1, 1), grid=((int(quant.nlayer)+15)//16, 1, 1))

        cuda.Context.synchronize()

    def iteration_loop(self, quant, write, plot):
        """ loops over the relevant kernels iteratively until equilibrium TP - profile reached """

        condition = True
        quant.iter_value = np.int32(0)

        # measuring time
        start_loop = cuda.Event()
        end_loop = cuda.Event()
        start_total = cuda.Event()
        end_total = cuda.Event()

        start_total.record()

        while condition:

            if quant.iter_value % 10 == 0:
                start_loop.record()

            self.interpolate_temperatures(quant)

            self.interpolate_opacities(quant)

            self.interpolate_cap_letter_terms(quant)

            self.interpolate_planck(quant)

            self.populate_spectral_flux(quant)

            self.integrate_flux(quant)

            if quant.singlewalk == 0:

                abortsum = 0

                if quant.iter_value % 100 == 0:
                    print("\nWe are at iteration step nr. : "+str(quant.iter_value))

                if quant.iter_value >= quant.foreplay:

                    self.net_flux_and_temp_iteration(quant)

                    quant.abort = quant.dev_abort.get()

                    for i in range(quant.nlayer):
                        abortsum += quant.abort[i]

                    if quant.iter_value % 10 == 0:
                        print("Layers converged: "+str(abortsum)+" out of "+str(quant.nlayer)+".")

                if quant.iter_value % 10 == 0:
                    write.write_restart_file(quant)
                    if quant.realtime_plot == 1:
                        plot.plot_tp(quant)

                # checks whether to continue the loop
                condition = abortsum < quant.nlayer
                quant.iter_value += 1
                quant.iter_value = np.int32(quant.iter_value)

                # records the time needed for 10 loops
                if (quant.iter_value-1) % 10 == 9:
                    end_loop.record()
                    end_loop.synchronize()
                    time_loop = start_loop.time_till(end_loop)
                    print("\nTime for the last 10 loops [s]: {:g}".format(time_loop * 1e-3))

            elif quant.singlewalk == 1:
                condition = False

        end_total.record()
        end_total.synchronize()
        time_total = start_total.time_till(end_total)
        print("\nTime for total run [s]: {:g}".format(time_total * 1e-3))

    def calculate_mean_opacities(self, quant):
        """ calculates the atmospheric Planck and Rosseland mean opacities """

        mean_opacities = self.mod.get_function("mean_opacities")

        mean_opacities(quant.dev_planck_opac_T_pl, quant.dev_ross_opac_T_pl, quant.dev_planck_opac_T_star,
                       quant.dev_ross_opac_T_star, quant.dev_opac_wg_lay, quant.dev_planckband_lay, quant.dev_opac_wave,
                       quant.dev_opac_interwave, quant.dev_opac_deltawave, quant.dev_T_lay, quant.dev_opac_weight,
                       quant.dev_opac_y, quant.dev_opac_lay, quant.nlayer, quant.nbin, quant.ny, quant.T_star,
                       block=(16, 1, 1), grid=((int(quant.nlayer)+15)//16, 1, 1))

        cuda.Context.synchronize()

    def calculate_transmission(self, quant):
        """ calculates the transmission function in each layer for separate analysis """

        transmission = self.mod.get_function("transmission")

        transmission(quant.dev_transmission, quant.dev_opac_wg_lay, quant.dev_opac_weight, quant.dev_delta_colmass,
                     quant.dev_cross_scat, quant.scat, quant.epsilon, quant.nbin, quant.nlayer, quant.ny,
                     quant.meanmolmass, quant.g_0,
                     block=(16, 16, 1), grid=((int(quant.nbin)+15)//16, (int(quant.nlayer)+15)//16, 1))

        cuda.Context.synchronize()


if __name__ == "__main__":
    print("This module is for computational purposes.")
