# ==============================================================================
# Module for the core computational part of HELIOS.
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
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from source import host_functions as hsfunc


class Compute(object):
    """ class incorporating the computational core of HELIOS """

    def __init__(self):
        self.kernel_file = open("./source/kernels.cu")
        self.kernels = self.kernel_file.read()
        self.mod = SourceModule(self.kernels)

    def construct_planck_table(self, quant):
        """ constructs the Planck table """

        # constructs the planck table
        plancktable = self.mod.get_function("plancktable")

        # iteration over smaller kernels instead of one big to prevent kernel timeouts.
        for p_iter in range(0,10):

            plancktable(quant.dev_planckband_grid,
                        quant.dev_opac_interwave,
                        quant.dev_opac_deltawave,
                        quant.nbin,
                        quant.T_star,
                        np.int32(p_iter),
                        quant.plancktable_dim,
                        quant.plancktable_step,
                        block=(16, 16, 1),
                        grid=((int(quant.nbin) + 15) // 16, (int(quant.plancktable_dim/10 + 1) + 15) // 16, 1)
                        )

        cuda.Context.synchronize()

    def correct_incident_energy(self, quant):
        """ adjusts the incoming energy flux to obtain the correct brightness temperature of the planet"""

        if quant.energy_correction == 1 and quant.T_star > 10:

            corr_inc_energy = self.mod.get_function("corr_inc_energy")

            corr_inc_energy(quant.dev_planckband_grid,
                            quant.dev_starflux,
                            quant.dev_opac_deltawave,
                            quant.real_star,
                            quant.nbin,
                            quant.T_star,
                            quant.plancktable_dim,
                            block=(16, 1, 1),
                            grid=((int(quant.nbin) + 15) // 16, 1, 1)
                            )

            cuda.Context.synchronize()

        cuda.Context.synchronize()

    def construct_grid(self, quant):
        """ constructs the atmospheric grid """

        gridkernel = self.mod.get_function("gridkernel")

        gridkernel(quant.dev_p_lay,
                   quant.dev_p_int,
                   quant.dev_delta_colmass,
                   quant.dev_delta_col_upper,
                   quant.dev_delta_col_lower,
                   quant.p_boa,
                   quant.p_toa,
                   quant.nlayer,
                   quant.g,
                   block=(16, 1, 1),
                   grid=((int(quant.nlayer)+15) // 16, 1, 1)
                   )

        cuda.Context.synchronize()

    def interpolate_temperatures(self, quant):
        """ interpolates the layer temperatures to the interfaces """

        temp_inter = self.mod.get_function("temp_inter")

        temp_inter(quant.dev_T_lay,
                   quant.dev_T_int,
                   quant.ninterface,
                   block=(16, 1, 1),
                   grid=((int(quant.ninterface)+15) // 16, 1, 1)
                   )

        cuda.Context.synchronize()

    def interpolate_opacities_and_scattering_cross_sections(self, quant):
        """ builds the layer and interface opacities by interpolating the values from the opacity table """

        opac_interpol = self.mod.get_function("opac_interpol")

        opac_interpol(quant.dev_T_lay,
                      quant.dev_ktemp,
                      quant.dev_p_lay,
                      quant.dev_kpress,
                      quant.dev_opac_k,
                      quant.dev_opac_wg_lay,
                      quant.dev_opac_scat_cross,
                      quant.dev_scat_cross_lay,
                      quant.npress,
                      quant.ntemp,
                      quant.ny,
                      quant.nbin,
                      quant.fake_opac,
                      quant.nlayer,
                      block=(16, 16, 1),
                      grid=((int(quant.nbin)+15) // 16, (int(quant.nlayer)+15) // 16, 1)
                      )

        cuda.Context.synchronize()

        if quant.iso == 0:
            opac_interpol(quant.dev_T_int,
                          quant.dev_ktemp,
                          quant.dev_p_int,
                          quant.dev_kpress,
                          quant.dev_opac_k,
                          quant.dev_opac_wg_int,
                          quant.dev_opac_scat_cross,
                          quant.dev_scat_cross_int,
                          quant.npress,
                          quant.ntemp,
                          quant.ny,
                          quant.nbin,
                          quant.fake_opac,
                          quant.ninterface,
                          block=(16, 16, 1),
                          grid=((int(quant.nbin) + 15) // 16, (int(quant.ninterface) + 15) // 16, 1)
                          )

        cuda.Context.synchronize()

    def interpolate_meanmolmass(self, quant):
        """ interpolates the mean molecular mass values from the k-table to the atmospheric layer values """

        mmm_interpol = self.mod.get_function("meanmolmass_interpol")
        mmm_interpol(quant.dev_T_lay,
                        quant.dev_ktemp,
                        quant.dev_meanmolmass_lay,
                        quant.dev_opac_meanmass,
                        quant.dev_p_lay,
                        quant.dev_kpress,
                        quant.npress,
                        quant.ntemp,
                        quant.nlayer,
                        block=(16, 1, 1),
                        grid=((int(quant.nlayer) + 15) // 16, 1, 1)
                        )

        cuda.Context.synchronize()

        if quant.iso == 0:

            mmm_interpol(quant.dev_T_int,
                         quant.dev_ktemp,
                         quant.dev_meanmolmass_int,
                         quant.dev_opac_meanmass,
                         quant.dev_p_int,
                         quant.dev_kpress,
                         quant.npress,
                         quant.ntemp,
                         quant.ninterface,
                         block=(16, 1, 1),
                         grid=((int(quant.ninterface) + 15) // 16, 1, 1)
                         )

        cuda.Context.synchronize()

    def interpolate_kappa(self, quant):

        # changes kappa to correct format
        if quant.kappa_manual_value == "file":
            quant.kappa_kernel_value = quant.fl_prec(0)
        else:
            quant.kappa_kernel_value = quant.fl_prec(quant.kappa_manual_value)

        kappa_interpol = self.mod.get_function("kappa_interpol")
        kappa_interpol(quant.dev_T_lay,
                       quant.dev_entr_temp,
                       quant.dev_p_lay,
                       quant.dev_entr_press,
                       quant.dev_kappa_lay,
                       quant.dev_opac_kappa,
                       quant.entr_npress,
                       quant.entr_ntemp,
                       quant.nlayer,
                       quant.kappa_kernel_value,
                       block=(16, 1, 1),
                       grid=((int(quant.nlayer) + 15) // 16, 1, 1)
                       )

        cuda.Context.synchronize()

        if quant.iso == 0:

            kappa_interpol = self.mod.get_function("kappa_interpol")
            kappa_interpol(quant.dev_T_int,
                           quant.dev_entr_temp,
                           quant.dev_p_int,
                           quant.dev_entr_press,
                           quant.dev_kappa_int,
                           quant.dev_opac_kappa,
                           quant.entr_npress,
                           quant.entr_ntemp,
                           quant.ninterface,
                           quant.kappa_kernel_value,
                           block=(16, 1, 1),
                           grid=((int(quant.ninterface) + 15) // 16, 1, 1)
                           )

            cuda.Context.synchronize()

    def interpolate_entropy(self, quant):

        if quant.convection == 1:

            entr_interpol = self.mod.get_function("entropy_interpol")
            entr_interpol(quant.dev_T_lay,
                          quant.dev_entr_temp,
                          quant.dev_p_lay,
                          quant.dev_entr_press,
                          quant.dev_entropy_lay,
                          quant.dev_opac_entropy,
                          quant.entr_npress,
                          quant.entr_ntemp,
                          quant.nlayer,
                          block=(16, 1, 1),
                          grid=((int(quant.nlayer) + 15) // 16, 1, 1)
                          )

            cuda.Context.synchronize()

    def calculate_c_p(self, quant):
        """ calculates the layer heat capacity from kappa and the mean molecular mass """

        if quant.convection == 1:

            calc_c_p = self.mod.get_function("calculate_cp")
            calc_c_p(quant.dev_kappa_lay,
                     quant.dev_meanmolmass_lay,
                     quant.dev_c_p_lay,
                     quant.nlayer,
                     block=(16, 1, 1),
                     grid=((int(quant.nlayer) + 15) // 16, 1, 1)
                     )

            cuda.Context.synchronize()

    def interpolate_planck(self, quant):
        """ interpolates the pre-tabulated Planck function to the layer/interface values
        plus stellar and internal blackbody """

        planck_interpol_layer = self.mod.get_function("planck_interpol_layer")

        planck_interpol_layer(quant.dev_T_lay,
                              quant.dev_planckband_lay,
                              quant.dev_planckband_grid,
                              quant.dev_starflux,
                              quant.real_star,
                              quant.nlayer,
                              quant.nbin,
                              quant.plancktable_dim,
                              quant.plancktable_step,
                              block=(16, 16, 1),
                              grid=((int(quant.nbin)+15)//16, (int(quant.nlayer+2)+15)//16, 1)
                              )

        cuda.Context.synchronize()

        if quant.iso == 0:
            planck_interpol_interface = self.mod.get_function("planck_interpol_interface")

            planck_interpol_interface(quant.dev_T_int,
                                      quant.dev_planckband_int,
                                      quant.dev_planckband_grid,
                                      quant.ninterface,
                                      quant.nbin,
                                      quant.plancktable_dim,
                                      quant.plancktable_step,
                                      block=(16, 16, 1),
                                      grid=((int(quant.nbin)+15)//16, (int(quant.ninterface)+15)//16, 1)
                                      )

        cuda.Context.synchronize()

    def normalize_cloud_scattering(self, quant):
        """ normalizes the cloud scattering to own lognormal grey absorption """

        if quant.clouds == 1:

            cloud_norm = self.mod.get_function("cloud_normalization")

            cloud_norm(
                       quant.dev_p_lay,
                       quant.dev_abs_cross_cloud,
                       quant.dev_cloud_opac_lay,
                       quant.dev_scat_cross_cloud,
                       quant.dev_cloud_scat_cross_lay,
                       quant.dev_scat_cross_lay,
                       quant.dev_meanmolmass_lay,
                       quant.dev_g_0_tot_lay,
                       quant.dev_g_0_cloud,
                       quant.g_0,
                       quant.cloud_opac_tot,
                       quant.cloud_press,
                       quant.cloud_width,
                       quant.nbin,
                       quant.nlayer,
                       block=(16, 16, 1),
                       grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
            )

            cuda.Context.synchronize()

            if quant.iso == 0:

                cloud_norm(
                           quant.dev_p_int,
                           quant.dev_abs_cross_cloud,
                           quant.dev_cloud_opac_int,
                           quant.dev_scat_cross_cloud,
                           quant.dev_cloud_scat_cross_int,
                           quant.dev_scat_cross_int,
                           quant.dev_meanmolmass_int,
                           quant.dev_g_0_tot_int,
                           quant.dev_g_0_cloud,
                           quant.g_0,
                           quant.cloud_opac_tot,
                           quant.cloud_press,
                           quant.cloud_width,
                           quant.nbin,
                           quant.ninterface,
                           block=(16, 16, 1),
                           grid=((int(quant.nbin) + 15) // 16, (int(quant.ninterface) + 15) // 16, 1)
                )

                cuda.Context.synchronize()

    def calculate_transmission(self, quant):
        """ calculates the transmission function in each layer for separate analysis """

        if quant.iso == 1:
            trans_iso = self.mod.get_function("calc_trans_iso")

            trans_iso(quant.dev_trans_wg,
                      quant.dev_delta_tau_wg,
                      quant.dev_M_term,
                      quant.dev_N_term,
                      quant.dev_P_term,
                      quant.dev_G_plus,
                      quant.dev_G_minus,
                      quant.dev_delta_colmass,
                      quant.dev_opac_wg_lay,
                      quant.dev_cloud_opac_lay,
                      quant.dev_meanmolmass_lay,
                      quant.dev_scat_cross_lay,
                      quant.dev_cloud_scat_cross_lay,
                      quant.dev_w_0,
                      quant.dev_g_0_tot_lay,
                      quant.g_0,
                      quant.epsi,
                      quant.mu_star,
                      quant.w_0_limit,
                      quant.scat,
                      quant.nbin,
                      quant.ny,
                      quant.nlayer,
                      quant.clouds,
                      quant.scat_corr,
                      quant.debug,
                      quant.i2s_transition,
                      block=(16, 4, 4),
                      grid=((int(quant.nbin)+15)//16, (int(quant.ny)+3)//4, (int(quant.nlayer)+3)//4)
                      )

        elif quant.iso == 0:
            trans_noniso = self.mod.get_function("calc_trans_noniso")

            trans_noniso(quant.dev_trans_wg_upper,
                         quant.dev_trans_wg_lower,
                         quant.dev_delta_tau_wg_upper,
                         quant.dev_delta_tau_wg_lower,
                         quant.dev_M_upper,
                         quant.dev_M_lower,
                         quant.dev_N_upper,
                         quant.dev_N_lower,
                         quant.dev_P_upper,
                         quant.dev_P_lower,
                         quant.dev_G_plus_upper,
                         quant.dev_G_plus_lower,
                         quant.dev_G_minus_upper,
                         quant.dev_G_minus_lower,
                         quant.dev_delta_col_upper,
                         quant.dev_delta_col_lower,
                         quant.dev_opac_wg_lay,
                         quant.dev_opac_wg_int,
                         quant.dev_cloud_opac_lay,
                         quant.dev_cloud_opac_int,
                         quant.dev_meanmolmass_lay,
                         quant.dev_meanmolmass_int,
                         quant.dev_scat_cross_lay,
                         quant.dev_scat_cross_int,
                         quant.dev_cloud_scat_cross_lay,
                         quant.dev_cloud_scat_cross_int,
                         quant.dev_w_0_upper,
                         quant.dev_w_0_lower,
                         quant.dev_g_0_tot_lay,
                         quant.dev_g_0_tot_int,
                         quant.g_0,
                         quant.epsi,
                         quant.mu_star,
                         quant.w_0_limit,
                         quant.scat,
                         quant.nbin,
                         quant.ny,
                         quant.nlayer,
                         quant.clouds,
                         quant.scat_corr,
                         quant.debug,
                         quant.i2s_transition,
                         block=(16, 4, 4),
                         grid=((int(quant.nbin) + 15) // 16, (int(quant.ny) + 3) // 4, (int(quant.nlayer) + 3) // 4)
                         )

        cuda.Context.synchronize()

    def calculate_delta_z(self, quant):
        """ calculates the vertical widths of the layers """

        calc_delta_z = self.mod.get_function("calc_delta_z")
        calc_delta_z(quant.dev_T_lay,
                     quant.dev_p_int,
                     quant.dev_p_lay,
                     quant.dev_meanmolmass_lay,
                     quant.dev_delta_z_lay,
                     quant.g,
                     quant.nlayer,
                     block=(16, 1, 1),
                     grid=((int(quant.nlayer) + 15) // 16, 1, 1)
                     )

        cuda.Context.synchronize()

    def calculate_direct_beamflux(self, quant):
        """ calculates the direct stellar flux at each interface """

        if quant.iso == 1:

            fdir_iso = self.mod.get_function("fdir_iso")
            fdir_iso(quant.dev_F_dir_wg,
                     quant.dev_planckband_lay,
                     quant.dev_delta_tau_wg,
                     quant.dev_z_lay,
                     quant.mu_star,
                     quant.R_planet,
                     quant.R_star,
                     quant.a,
                     quant.dir_beam,
                     quant.geom_zenith_corr,
                     quant.ninterface,
                     quant.nbin,
                     quant.ny,
                     block=(4, 32, 4),
                     grid=((int(quant.ninterface) + 3) // 4, (int(quant.nbin) + 31) // 32, (int(quant.ny) + 3) // 4)
                     )

        elif quant.iso == 0:

            fdir_noniso = self.mod.get_function("fdir_noniso")
            fdir_noniso(quant.dev_F_dir_wg,
                        quant.dev_Fc_dir_wg,
                        quant.dev_planckband_lay,
                        quant.dev_delta_tau_wg_upper,
                        quant.dev_delta_tau_wg_lower,
                        quant.dev_z_lay,
                        quant.mu_star,
                        quant.R_planet,
                        quant.R_star,
                        quant.a,
                        quant.dir_beam,
                        quant.geom_zenith_corr,
                        quant.ninterface,
                        quant.nbin,
                        quant.ny,
                        block=(4, 32, 4),
                        grid=((int(quant.ninterface) + 3) // 4, (int(quant.nbin) + 31) // 32, (int(quant.ny) + 3) // 4)
                        )

        cuda.Context.synchronize()

    def populate_spectral_flux(self, quant):
        """ populates the down- and upstream spectral fluxes """

        nscat_step = None
        if quant.singlewalk == 0:
            nscat_step = 3
        if quant.singlewalk == 1:
            nscat_step = 200

        for scat_iter in range(nscat_step * quant.scat + 1):

            if quant.iso == 1:

                fband_iso = self.mod.get_function("fband_iso")
                fband_iso(quant.dev_F_down_wg,
                                 quant.dev_F_up_wg,
                                 quant.dev_F_dir_wg,
                                 quant.dev_planckband_lay,
                                 quant.dev_w_0,
                                 quant.dev_M_term,
                                 quant.dev_N_term,
                                 quant.dev_P_term,
                                 quant.dev_G_plus,
                                 quant.dev_G_minus,
                                 quant.dev_g_0_tot_lay,
                                 quant.g_0,
                                 quant.singlewalk,
                                 quant.R_star,
                                 quant.a,
                                 quant.ninterface,
                                 quant.nbin,
                                 quant.f_factor,
                                 quant.mu_star,
                                 quant.ny,
                                 quant.epsi,
                                 quant.dir_beam,
                                 quant.clouds,
                                 quant.scat_corr,
                                 quant.surf_albedo,
                                 quant.debug,
                                 quant.i2s_transition,
                                 block=(16, 16, 1),
                                 grid=((int(quant.nbin)+15)//16, (int(quant.ny)+15)//16, 1)
                                 )

            elif quant.iso == 0:

                fband_noniso = self.mod.get_function("fband_noniso")
                fband_noniso(quant.dev_F_down_wg,
                                    quant.dev_F_up_wg,
                                    quant.dev_Fc_down_wg,
                                    quant.dev_Fc_up_wg,
                                    quant.dev_F_dir_wg,
                                    quant.dev_Fc_dir_wg,
                                    quant.dev_planckband_lay,
                                    quant.dev_planckband_int,
                                    quant.dev_w_0_upper,
                                    quant.dev_w_0_lower,
                                    quant.dev_delta_tau_wg_upper,
                                    quant.dev_delta_tau_wg_lower,
                                    quant.dev_M_upper,
                                    quant.dev_M_lower,
                                    quant.dev_N_upper,
                                    quant.dev_N_lower,
                                    quant.dev_P_upper,
                                    quant.dev_P_lower,
                                    quant.dev_G_plus_upper,
                                    quant.dev_G_plus_lower,
                                    quant.dev_G_minus_upper,
                                    quant.dev_G_minus_lower,
                                    quant.dev_g_0_tot_lay,
                                    quant.dev_g_0_tot_int,
                                    quant.g_0,
                                    quant.singlewalk,
                                    quant.R_star,
                                    quant.a,
                                    quant.ninterface,
                                    quant.nbin,
                                    quant.f_factor,
                                    quant.mu_star,
                                    quant.ny,
                                    quant.epsi,
                                    quant.delta_tau_limit,
                                    quant.dir_beam,
                                    quant.clouds,
                                    quant.scat_corr,
                                    quant.surf_albedo,
                                    quant.debug,
                                    quant.i2s_transition,
                                    block=(16, 16, 1),
                                    grid=((int(quant.nbin)+15)//16, (int(quant.ny)+15)//16, 1)
                                    )

            cuda.Context.synchronize()

    def integrate_flux(self, quant):
        """ integrates the spectral fluxes first over each bin and then the whole spectral range """

        if quant.prec == "double":
            integrate_flux = self.mod.get_function("integrate_flux_double")
        elif quant.prec == "single":
            integrate_flux = self.mod.get_function("integrate_flux_single")

        integrate_flux(quant.dev_opac_deltawave,
                       quant.dev_F_down_tot,
                       quant.dev_F_up_tot,
                       quant.dev_F_net,
                       quant.dev_F_down_wg,
                       quant.dev_F_up_wg,
                       quant.dev_F_dir_wg,
                       quant.dev_F_down_band,
                       quant.dev_F_up_band,
                       quant.dev_F_dir_band,
                       quant.dev_gauss_weight,
                       quant.nbin,
                       quant.ninterface,
                       quant.ny,
                       block=(32, 4, 8),
                       grid=(1, 1, 1)
                      )

        # OLD Version of integrating the fluxes -- keep here for debugging purposes
        # integrate_flux = self.mod.get_function("integrate_flux")
        # integrate_flux(quant.dev_opac_deltawave,
        #                quant.dev_F_down_tot,
        #                quant.dev_F_up_tot,
        #                quant.dev_F_net,
        #                quant.dev_F_down_wg,
        #                quant.dev_F_up_wg,
        #                quant.dev_F_dir_wg,
        #                quant.dev_F_down_band,
        #                quant.dev_F_up_band,
        #                quant.dev_F_dir_band,
        #                quant.dev_gauss_weight,
        #                quant.nbin,
        #                quant.ninterface,
        #                quant.ny,
        #                block=(16, 1, 1),
        #                grid=((int(quant.ninterface)+15)//16, 1, 1)
        #                )

        cuda.Context.synchronize()

    def rad_temp_iteration(self, quant):
        """ calculates the net flux and advances the layer temperature """

        radtempiter = self.mod.get_function("rad_temp_iter")
        radtempiter(quant.dev_F_down_tot,
                    quant.dev_F_up_tot,
                    quant.dev_F_net,
                    quant.dev_F_net_diff,
                    quant.dev_T_lay,
                    quant.dev_p_lay,
                    quant.dev_T_int,
                    quant.dev_p_int,
                    quant.dev_abort,
                    quant.dev_T_store,
                    quant.dev_delta_t_prefactor,
                    quant.iter_value,
                    quant.f_factor,
                    quant.foreplay,
                    quant.tstep,
                    quant.nlayer,
                    quant.varying_tstep,
                    quant.local_limit_rad_iter,
                    quant.adapt_interval,
                    quant.smooth,
                    quant.plancktable_dim,
                    quant.plancktable_step,
                    quant.F_intern,
                    block=(16, 1, 1),
                    grid=((int(quant.nlayer+1)+15)//16, 1, 1)
                    )

        cuda.Context.synchronize()

    def conv_temp_iteration(self, quant):
        """ temperature progression for the convection loop """

        convtempiter = self.mod.get_function("conv_temp_iter")
        convtempiter(quant.dev_F_down_tot,
                     quant.dev_F_up_tot,
                     quant.dev_F_net,
                     quant.dev_F_net_diff,
                     quant.dev_T_lay,
                     quant.dev_p_lay,
                     quant.dev_p_int,
                     quant.dev_c_p_lay,
                     quant.dev_T_store,
                     quant.dev_delta_t_prefactor,
                     quant.dev_conv_layer,
                     quant.g,
                     quant.nlayer,
                     quant.iter_value,
                     quant.adapt_interval,
                     quant.smooth,
                     quant.F_intern,
                     block=(16, 1, 1),
                     grid=((int(quant.nlayer+1)+15)//16, 1, 1)
                     )

        cuda.Context.synchronize()

    def radiation_loop(self, quant, write, rt_plot, Vmod):
        """ loops over the relevant kernels iteratively until the equilibrium TP - profile reached """

        condition = True
        quant.iter_value = np.int32(0)
        quant.p_lay = quant.dev_p_lay.get()
        quant.p_int = quant.dev_p_int.get()

        # measures the runtime of a specified number of iterations
        start_loop = cuda.Event()
        end_loop = cuda.Event()
        start_total = cuda.Event()
        end_total = cuda.Event()

        # uncomment for time testing purposes
        # start_test = cuda.Event()
        # end_test = cuda.Event()

        start_total.record()

        while condition:

            if quant.iter_value % 100 == 0:
                start_loop.record()

            self.interpolate_temperatures(quant)
            self.interpolate_planck(quant)

            if quant.iter_value % 10 == 0:

                if Vmod.V_coupling == 0:
                    self.interpolate_opacities_and_scattering_cross_sections(quant)
                    self.interpolate_meanmolmass(quant)
                elif Vmod.V_coupling == 1:
                    if Vmod.V_iter_nr == 0:
                        self.interpolate_opacities_and_scattering_cross_sections(quant)
                        self.interpolate_meanmolmass(quant)
                    Vmod.interpolate_molecular_and_mixed_opac(quant)
                    Vmod.combine_to_mixed_opacities(quant)
                self.interpolate_kappa(quant)
                self.calculate_c_p(quant)
                self.normalize_cloud_scattering(quant)
                self.calculate_transmission(quant)

                self.calculate_delta_z(quant)
                quant.delta_z_lay = quant.dev_delta_z_lay.get()
                hsfunc.calculate_height_z(quant)
                quant.dev_z_lay = gpuarray.to_gpu(quant.z_lay)
                self.calculate_direct_beamflux(quant)
            self.populate_spectral_flux(quant)
            self.integrate_flux(quant)

            # uncomment for time testing purposes
            # start_test.record()
            # end_test.record()
            # end_test.synchronize()
            # time_test = start_test.time_till(end_test)
            # print("\nTime for test [s]: {:g}".format(time_test * 1e-3))

            if quant.singlewalk == 0:

                abortsum = 0
                quant.marked_red = np.zeros(quant.nlayer+1)

                if quant.iter_value % 100 == 0:
                    print("\nWe are running \"" + quant.name + "\" at iteration step nr. : "+str(quant.iter_value))
                    if quant.iter_value > 99:
                        print("Time for the last 100 steps [s]: {:.2f}".format(time_loop * 1e-3))
                if quant.iter_value >= quant.foreplay:

                    # radiative temperature progression
                    self.rad_temp_iteration(quant)
                    quant.abort = quant.dev_abort.get()

                    for i in range(quant.nlayer+1):
                        if quant.abort[i] == 0:
                            quant.marked_red[i] = 1
                    abortsum = sum(quant.abort)

                    if quant.iter_value % 100 == 0:
                        print("Layers converged: "+str(abortsum)+" out of "+str(quant.nlayer+1)+".")

                # checks whether to continue the loop
                condition = abortsum < quant.nlayer + 1  # including "ghost layer" below grid
                quant.iter_value += 1
                quant.iter_value = np.int32(quant.iter_value)

                if quant.iter_value % quant.n_plot == 0 and quant.realtime_plot == 1:
                    rt_plot.plot_tp(quant)

                # records the time needed for 100 loops
                if quant.iter_value % 100 == 99:
                    end_loop.record()
                    end_loop.synchronize()
                    time_loop = start_loop.time_till(end_loop)

                # time restriction for the run. It aborts automatically after the following time steps and prevents a hung job.
                if quant.iter_value > 1e5:
                    write.write_abort_file(quant)

                    print("\nRun exceeds allowed maximum allowed number of iteration steps. Aborting...")

                    raise SystemExit()

            elif quant.singlewalk == 1:
                condition = False

        end_total.record()
        end_total.synchronize()
        time_total = start_total.time_till(end_total)
        print("\nTime for radiative iteration [s]: {:.2f}".format(time_total * 1e-3))
        print("Total number of iterative steps: "+str(quant.iter_value))

    def convection_loop(self, quant, write, rt_plot, Vmod):
        """ loops interchangeably through the radiative and convection schemes """

        # kappa is required for the conv. instability check
        self.interpolate_kappa(quant)
        quant.T_lay = quant.dev_T_lay.get()
        quant.p_lay = quant.dev_p_lay.get()
        quant.p_int = quant.dev_p_int.get()
        quant.kappa_lay = quant.dev_kappa_lay.get()
        if quant.iso == 0:
            quant.kappa_int = quant.dev_kappa_int.get()
            hsfunc.conv_check(quant)
            hsfunc.mark_convective_layers(quant, stitching=0)

        # only starts the loop if convective adjustment is switched on
        if quant.singlewalk == 0 and quant.convection == 1:

            condition = sum(quant.conv_unstable) > 0

            start_total = cuda.Event()
            end_total = cuda.Event()
            start_total.record()

            quant.iter_value = np.int32(0)

            if condition:
                # measures time
                start_loop = cuda.Event()
                end_loop = cuda.Event()
                print("\nConvectively unstable layers found. Starting convective adjustment")
            else:
                print("\nAll layers convectively stable. No convective adjustment necessary.\n")

            # quantities required on the host for the first convective adjustment
            quant.F_net = quant.dev_F_net.get()
            quant.F_up_tot = quant.dev_F_up_tot.get()
            quant.F_down_tot = quant.dev_F_down_tot.get()

            while condition:

                if quant.iter_value % 100 == 0:
                    start_loop.record()

                if quant.iter_value % 100 == 0:
                    print("\nWe are running \"" + quant.name + "\" at iteration step nr. : "+str(quant.iter_value))
                    if quant.iter_value > 99:
                        print("Time for the last 100 steps [s]: {:.2f}".format(time_loop * 1e-3))
                # start with the convective adjustment and then recalculate the rad. fluxes, go back to conv. adjustment, then rad. fluxes, etc.
                self.interpolate_temperatures(quant)

                if Vmod.V_coupling == 0:
                    self.interpolate_meanmolmass(quant)
                elif Vmod.V_coupling == 1:
                    if Vmod.V_iter_nr == 0:
                        self.interpolate_meanmolmass(quant)

                self.interpolate_kappa(quant)
                self.calculate_c_p(quant)
                quant.kappa_lay = quant.dev_kappa_lay.get()
                quant.kappa_int = quant.dev_kappa_int.get()
                quant.c_p_lay = quant.dev_c_p_lay.get()
                quant.T_lay = quant.dev_T_lay.get()
                hsfunc.convective_adjustment(quant)
                quant.dev_T_lay = gpuarray.to_gpu(quant.T_lay)

                self.interpolate_temperatures(quant)

                if Vmod.V_coupling == 0:
                    self.interpolate_opacities_and_scattering_cross_sections(quant)
                    self.interpolate_meanmolmass(quant)
                elif Vmod.V_coupling == 1:
                    if Vmod.V_iter_nr == 0:
                        self.interpolate_opacities_and_scattering_cross_sections(quant)
                        self.interpolate_meanmolmass(quant)
                    Vmod.interpolate_molecular_and_mixed_opac(quant)
                    Vmod.combine_to_mixed_opacities(quant)

                self.normalize_cloud_scattering(quant)
                self.interpolate_planck(quant)
                self.calculate_transmission(quant)
                if quant.iter_value % 10 == 0:
                    self.calculate_delta_z(quant)
                    quant.delta_z_lay = quant.dev_delta_z_lay.get()
                    hsfunc.calculate_height_z(quant)
                    quant.dev_z_lay = gpuarray.to_gpu(quant.z_lay)
                    self.calculate_direct_beamflux(quant)
                self.populate_spectral_flux(quant)
                self.integrate_flux(quant)

                # copy back fluxes to determine convergence
                quant.F_net = quant.dev_F_net.get()
                quant.F_down_tot = quant.dev_F_down_tot.get()
                quant.F_up_tot = quant.dev_F_up_tot.get()
                quant.F_net_diff = quant.dev_F_net_diff.get()

                # required to mark convective zones
                self.interpolate_kappa(quant)
                self.calculate_c_p(quant)
                quant.kappa_lay = quant.dev_kappa_lay.get()
                quant.kappa_int = quant.dev_kappa_int.get()
                quant.T_lay = quant.dev_T_lay.get()

                # mark convection zone. used by realtime plotting
                hsfunc.mark_convective_layers(quant, stitching=1)

                # checks whether to continue the loop
                condition = not(hsfunc.check_for_global_local_equilibrium(quant)) or quant.iter_value < 500

                # relax global convergence limit somewhat if taking too long to converge
                if quant.iter_value == 1e4:  # warning: hardcoded number
                    hsfunc.relax_global_limit(quant)

                # radiative forward stepping if local flux criterion not satisfied
                if condition:

                    # realtime plotting every 10th step
                    if quant.iter_value % quant.n_plot == 0 and quant.realtime_plot == 1:
                        rt_plot.plot_convective_feedback(quant)

                    # kernel that advances the temperature in a radiative way
                    quant.dev_conv_layer = gpuarray.to_gpu(quant.conv_layer)
                    self.conv_temp_iteration(quant)

                    quant.T_lay = quant.dev_T_lay.get()

                    # records the time needed for 100 loops
                    if quant.iter_value % 100 == 99:
                        end_loop.record()
                        end_loop.synchronize()
                        time_loop = start_loop.time_till(end_loop)

                    quant.iter_value += 1
                    quant.iter_value = np.int32(quant.iter_value)

                # length restriction for the run. aborts after a upper limit on the number of steps and thus prevents a hung up job.
                if quant.iter_value > 1e5:  # warning: hardcoded number

                    write.write_abort_file(quant)

                    print("\nRun exceeds allowed maximum allowed number of iteration steps. Aborting...")

                    raise SystemExit()

            end_total.record()
            end_total.synchronize()
            time_total = start_total.time_till(end_total)

            print("\nTime for rad.-conv. iteration [s]: {:.2f}".format(time_total * 1e-3))
            print("Total number of iterative steps: " + str(quant.iter_value))

    def integrate_optdepth_transmission(self, quant):
        """ calculates the transmission function in each layer """

        if quant.iso == 1:
            integrate_optdepth_trans = self.mod.get_function("integrate_optdepth_transmission_iso")

            integrate_optdepth_trans(quant.dev_trans_wg,
                            quant.dev_trans_band,
                            quant.dev_delta_tau_wg,
                            quant.dev_delta_tau_band,
                            quant.dev_gauss_weight,
                            quant.nbin,
                            quant.nlayer,
                            quant.ny,
                            block=(16, 16, 1),
                            grid=((int(quant.nbin)+15)//16, (int(quant.nlayer)+15)//16, 1)
                            )

        elif quant.iso == 0:
            integrate_optdepth_trans = self.mod.get_function("integrate_optdepth_transmission_noniso")

            integrate_optdepth_trans(quant.dev_trans_wg_upper,
                                     quant.dev_trans_wg_lower,
                                     quant.dev_trans_band,
                                     quant.dev_delta_tau_wg_upper,
                                     quant.dev_delta_tau_wg_lower,
                                     quant.dev_delta_tau_band,
                                     quant.dev_gauss_weight,
                                     quant.nbin,
                                     quant.nlayer,
                                     quant.ny,
                                     block=(16, 16, 1),
                                     grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                                     )

        cuda.Context.synchronize()

    def calculate_contribution_function(self, quant):
        """ calculate the transmission weighting function and the contribution function for each layer and waveband """

        if quant.iso == 1:
            calc_contr_func = self.mod.get_function("calc_contr_func_iso")

            calc_contr_func(quant.dev_trans_wg,
                            quant.dev_trans_weight_band,
                            quant.dev_contr_func_band,
                            quant.dev_gauss_weight,
                            quant.dev_planckband_lay,
                            quant.epsi,
                            quant.nbin,
                            quant.nlayer,
                            quant.ny,
                            block=(16, 16, 1),
                            grid=((int(quant.nbin)+15)//16, (int(quant.nlayer)+15)//16, 1)
                            )

        elif quant.iso == 0:
            calc_contr_func = self.mod.get_function("calc_contr_func_noniso")

            calc_contr_func(quant.dev_trans_wg_upper,
                            quant.dev_trans_wg_lower,
                            quant.dev_trans_weight_band,
                            quant.dev_contr_func_band,
                            quant.dev_gauss_weight,
                            quant.dev_planckband_lay,
                            quant.epsi,
                            quant.nbin,
                            quant.nlayer,
                            quant.ny,
                            block=(16, 16, 1),
                            grid=((int(quant.nbin) + 15) // 16, (int(quant.nlayer) + 15) // 16, 1)
                            )

        cuda.Context.synchronize()

    def calculate_mean_opacities(self, quant):
        """ calculates the atmospheric Planck and Rosseland mean opacities """

        mean_opacities = self.mod.get_function("calc_mean_opacities")

        mean_opacities(quant.dev_planck_opac_T_pl,
                       quant.dev_ross_opac_T_pl,
                       quant.dev_planck_opac_T_star,
                       quant.dev_ross_opac_T_star,
                       quant.dev_opac_wg_lay,
                       quant.dev_cloud_opac_lay,
                       quant.dev_planckband_lay,
                       quant.dev_opac_interwave,
                       quant.dev_opac_deltawave,
                       quant.dev_T_lay,
                       quant.dev_gauss_weight,
                       quant.dev_opac_y,
                       quant.dev_opac_band_lay,
                       quant.nlayer,
                       quant.nbin,
                       quant.ny,
                       quant.T_star,
                       block=(16, 1, 1),
                       grid=((int(quant.nlayer)+15)//16, 1, 1)
                       )

        cuda.Context.synchronize()

    def integrate_beamflux(self, quant):
        """ integrates the spectral direct beam flux first over each bin and then the whole spectral range """

        integrate_flux = self.mod.get_function("integrate_beamflux")

        integrate_flux(quant.dev_F_dir_tot,
                       quant.dev_F_dir_band,
                       quant.dev_opac_deltawave,
                       quant.dev_gauss_weight,
                       quant.nbin,
                       quant.ninterface,
                       block=(16, 1, 1),
                       grid=((int(quant.ninterface)+15)//16, 1, 1)
                       )

if __name__ == "__main__":
    print("This module is for computational purposes. It is the working horse of the whole code.")
