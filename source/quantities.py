# ==============================================================================
# Module for storing all the necessary quantities like parameters,
# arrays, input & output data, etc.
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


class Store(object):
    """ class that stores parameters, quantities, arrays, etc., used in the HELIOS code"""

    def __init__(self):

        # Single variables to be filled with input data
        # Device can read those directly from the host.
        # They need to be in correct data format! Check if necessary!
        self.iso = None
        self.nlayer = None
        self.ninterface = None
        self.p_toa = None
        self.p_boa = None
        self.singlewalk = None
        self.varying_tstep = None
        self.tstep = None
        self.scat = None
        self.diffusivity = None
        self.epsi = None
        self.epsi2 = None
        self.f_factor = None
        self.T_intern = None
        self.ntemp = None
        self.npress = None
        self.entr_ntemp = None
        self.entr_npress = None
        self.g_0 = None
        self.planet = None
        self.g = None
        self.a = None
        self.R_planet = None
        self.R_star = None
        self.T_star = None
        self.T_eff_final = None
        self.model = None
        self.real_star = np.int32(0)
        self.name = None
        self.foreplay = None
        self.fake_opac = None
        self.realtime_plot = None
        self.prec = None
        self.fl_prec = None
        self.nr_bytes = None
        self.iter_value = None
        self.ny = None
        self.nbin = None
        self.nlayer_nbin = None
        self.nlayer_plus2_nbin = None
        self.ninterface_nbin = None
        self.nlayer_wg_nbin = None
        self.ninterface_wg_nbin = None
        self.nplanck_grid = None
        self.dir_beam = None
        self.dir_angle = None
        self.mu_star = None
        self.w_0_limit = None
        self.delta_tau_limit = None
        self.local_limit_rad_iter = None
        self.global_limit = None
        self.n_plot = None
        self.energy_correction = None
        self.clouds = None
        self.cloud_path = None
        self.cloud_opac_tot = None
        self.cloud_press = None
        self.cloud_width = None
        self.star_corr_factor = np.int32(1)
        self.input_dampara = None
        self.dampara = None
        self.F_intern = None
        self.adapt_interval = None
        self.smooth = None
        self.geom_zenith_corr = None
        self.scat_corr = None
        self.kappa_manual_value = None
        self.kappa_kernel_value = None
        self.surf_albedo = None
        self.T_below = 0  # T_below is either the surface temperature for rocky planets or the below grid temperature for gas planets
        self.approx_f = None
        self.tau_lw = 1
        self.planet_type = None
        self.F_sens = 0
        self.debug = None
        self.i2s_transition = None
        # number of pre-tabulated temperature values for the planck table
        self.plancktable_dim = np.int32(8000)
        # temperature step for the planck table. e.g. dim = 10000 and step = 2 will give a table from 1 K to 19999 K in 2 K steps
        self.plancktable_step = np.int32(2)

        # arrays/lists exclusively used on the CPU
        self.T_restart = []
        self.conv_unstable = None
        self.F_net_conv = []
        self.F_ratio = []
        self.marked_red = None

        # input arrays to be copied CPU --> GPU
        # these need to be converted from lists to np.arrays of correct data format
        # and then copied to GPU with "gpuarray"
        self.p_lay = None
        self.dev_p_lay = None
        self.p_int = None
        self.dev_p_int = None
        self.delta_colmass = []
        self.dev_delta_colmass = None
        self.delta_col_upper = []
        self.dev_delta_col_upper = None
        self.delta_col_lower = []
        self.dev_delta_col_lower = None
        self.ktemp = None
        self.dev_ktemp = None
        self.kpress = None
        self.dev_kpress = None
        self.entr_temp = []
        self.dev_entr_temp = None
        self.entr_press = []
        self.dev_entr_press = None
        self.entr_kappa = []
        self.dev_entr_kappa = None
        self.entr_c_p = []
        self.dev_entr_c_p = None
        self.entr_phase_number = []
        self.dev_entr_phase_number = None
        self.entr_entropy = []
        self.dev_entr_entropy = None
        self.opac_k = None
        self.dev_opac_k = None
        self.opac_y = None
        self.dev_opac_y = None
        self.gauss_weight = None
        self.dev_gauss_weight = None
        self.opac_wave = None
        self.dev_opac_wave = None
        self.opac_deltawave = None
        self.dev_opac_deltawave = None
        self.opac_interwave = None
        self.dev_opac_interwave = None
        self.opac_scat_cross = None
        self.dev_opac_scat_cross = None
        self.opac_meanmass = None
        self.dev_opac_meanmass = None
        self.opac_kappa = []
        self.dev_opac_kappa = None
        self.opac_entropy = []
        self.dev_opac_entropy = None
        self.starflux = []
        self.dev_starflux = None
        self.scat_cross_cloud = []
        self.dev_scat_cross_cloud = None
        self.abs_cross_cloud = []
        self.dev_abs_cross_cloud = None
        self.g_0_cloud = []
        self.dev_g_0_cloud = None
        self.opac_wg_lay = []  # only used for the chem type calculation
        self.opac_k_h2o = None
        self.dev_opac_k_h2o = None
        self.opac_k_co2 = None
        self.dev_opac_k_co2 = None
        self.opac_k_co = None
        self.dev_opac_k_co = None
        self.opac_k_ch4 = None
        self.dev_opac_k_ch4 = None
        self.opac_k_nh3 = None
        self.dev_opac_k_nh3 = None
        self.opac_k_hcn = None
        self.dev_opac_k_hcn = None
        self.opac_k_c2h2 = None
        self.dev_opac_k_c2h2 = None
        self.opac_k_tio = None
        self.dev_opac_k_tio = None
        self.opac_k_vo = None
        self.dev_opac_k_vo = None
        self.opac_k_h2s = None
        self.dev_opac_k_h2s = None
        self.opac_k_na = None
        self.dev_opac_k_na = None
        self.opac_k_k = None
        self.dev_opac_k_k = None
        self.opac_k_cia_h2h2 = None
        self.dev_opac_k_cia_h2h2 = None
        self.opac_k_cia_h2he = None
        self.dev_opac_k_cia_h2he = None
        self.k_scat_cross = None
        self.dev_k_scat_cross = None
        self.f_h2o_tab = None
        self.dev_f_h2o_tab = None
        self.f_co2_tab = None
        self.dev_f_co2_tab = None
        self.f_co_tab = None
        self.dev_f_co_tab = None
        self.f_ch4_tab = None
        self.dev_f_ch4_tab = None
        self.f_nh3_tab = None
        self.dev_f_nh3_tab = None
        self.f_hcn_tab = None
        self.dev_f_hcn_tab = None
        self.f_h2s_tab = None
        self.dev_f_h2s_tab = None
        self.f_h_tab = None
        self.dev_f_h_tab = None
        self.f_h2_tab = None
        self.dev_f_h2_tab = None
        self.f_he_tab = None
        self.dev_f_he_tab = None
        self.f_c2h2_tab = None
        self.dev_f_c2h2_tab = None
        self.f_k_tab = None
        self.dev_f_k_tab = None
        self.f_na_tab = None
        self.dev_f_na_tab = None
        self.conv_layer = None
        self.dev_conv_layer = None

        # only used by Vmod
        self.f_h2o_lay = None
        self.dev_f_h2o_lay = None
        self.f_co2_lay = None
        self.dev_f_co2_lay = None
        self.f_co_lay = None
        self.dev_f_co_lay = None
        self.f_ch4_lay = None
        self.dev_f_ch4_lay = None
        self.f_nh3_lay = None
        self.dev_f_nh3_lay = None
        self.f_hcn_lay = None
        self.dev_f_hcn_lay = None
        self.f_c2h2_lay = None
        self.dev_f_c2h2_lay = None
        self.f_h2s_lay = None
        self.dev_f_h2s_lay = None
        self.f_h2_lay = None
        self.dev_f_h2_lay = None
        self.f_he_lay = None
        self.dev_f_he_lay = None
        self.f_na_lay = None
        self.dev_f_na_lay = None
        self.f_k_lay = None
        self.dev_f_k_lay = None
        self.f_tio_lay = None
        self.dev_f_tio_lay = None
        self.f_vo_lay = None
        self.dev_f_vo_lay = None


        # arrays to be copied CPU --> GPU --> CPU
        # these are copied to GPU by "gpuarray" and copied back
        self.T_lay = []
        self.dev_T_lay = None
        self.abort = None
        self.dev_abort = None

        # arrays to be copied GPU --> CPU
        # for these, zero arrays of correct size are created and then copied to GPU with "gpuarray" and copied back
        self.F_up_band = None
        self.dev_F_up_band = None
        self.F_down_band = None
        self.dev_F_down_band = None
        self.F_dir_band = None
        self.dev_F_dir_band = None
        self.F_dir_tot = None
        self.dev_F_dir_tot = None
        self.F_up_tot = None
        self.dev_F_up_tot = None
        self.F_down_tot = None
        self.dev_F_down_tot = None
        self.opac_band_lay = None
        self.dev_opac_band_lay = None
        self.scat_cross_lay = None
        self.dev_scat_cross_lay = None
        self.F_net = None
        self.dev_F_net = None
        self.F_net_diff = None
        self.dev_F_net_diff = None
        self.planckband_lay = None
        self.dev_planckband_lay = None
        self.planckband_int = None
        self.dev_planckband_int = None
        self.planck_opac_T_pl = None
        self.dev_planck_opac_T_pl = None
        self.ross_opac_T_pl = None
        self.dev_ross_opac_T_pl = None
        self.planck_opac_T_star = None
        self.dev_planck_opac_T_star = None
        self.ross_opac_T_star = None
        self.dev_ross_opac_T_star = None
        self.delta_tau_band = None
        self.dev_delta_tau_band = None
        self.trans_band = None
        self.dev_trans_band = None
        self.meanmolmass_lay = None
        self.dev_meanmolmass_lay = None
        self.c_p_lay = None
        self.dev_c_p_lay = None
        self.kappa_lay = None
        self.dev_kappa_lay = None
        self.kappa_int = None
        self.dev_kappa_int = None
        self.entropy_lay = None
        self.dev_entropy_lay = None
        self.trans_weight_band = None
        self.dev_trans_weight_band = None
        self.contr_func_band = None
        self.dev_contr_func_band = None
        self.cloud_opac_lay = None
        self.dev_cloud_opac_lay = None
        self.cloud_opac_int = None
        self.dev_cloud_opac_int = None
        self.cloud_scat_cross_lay = None
        self.dev_cloud_scat_cross_lay = None
        self.cloud_scat_cross_int = None
        self.dev_cloud_scat_cross_int = None
        self.g_0_tot_lay = None
        self.dev_g_0_tot_lay = None
        self.g_0_tot_int = None
        self.dev_g_0_tot_int = None
        self.delta_z_lay = None
        self.dev_delta_z_lay = None
        self.z_lay = None
        self.dev_z_lay = None
        self.phase_number_lay = None
        self.dev_phase_number_lay = None
        self.test_arr = None
        self.dev_test_arr = None

        # arrays exclusively used on the GPU
        # these are defined directly on the GPU and stay there. No copying required.
        self.dev_T_int = None
        self.dev_delta_t_prefactor = None
        self.dev_T_store = None
        self.dev_planckband_grid = None
        self.dev_opac_int = None
        self.dev_scat_cross_int = None
        self.dev_opac_wg_lay = None
        self.dev_opac_wg_int = None
        self.F_up_wg = None
        self.dev_F_up_wg = None
        self.F_down_wg = None
        self.dev_F_down_wg = None
        self.F_dir_wg = None
        self.dev_F_dir_wg = None
        self.Fc_down_wg = None
        self.dev_Fc_down_wg = None
        self.Fc_up_wg = None
        self.dev_Fc_up_wg = None
        self.Fc_dir_wg = None
        self.dev_Fc_dir_wg = None
        self.dev_trans_wg = None
        self.dev_trans_wg_upper = None
        self.dev_trans_wg_lower = None
        self.dev_delta_tau_wg = None
        self.dev_delta_tau_wg_upper = None
        self.dev_delta_tau_wg_lower = None
        self.dev_meanmolmass_int = None
        self.dev_M_term = None
        self.dev_N_term = None
        self.dev_P_term = None
        self.dev_M_upper = None
        self.dev_N_upper = None
        self.dev_P_upper = None
        self.dev_M_lower = None
        self.dev_N_lower = None
        self.dev_P_lower = None
        self.dev_G_plus = None
        self.dev_G_minus = None
        self.dev_G_plus_upper = None
        self.dev_G_minus_upper = None
        self.dev_G_plus_lower = None
        self.dev_G_minus_lower = None
        self.dev_w_0 = None
        self.dev_w_0_upper = None
        self.dev_w_0_lower = None
        self.dev_opac_h2o_wg_lay = None
        self.dev_opac_co2_wg_lay = None
        self.dev_opac_co_wg_lay = None
        self.dev_opac_ch4_wg_lay = None
        self.dev_opac_nh3_wg_lay = None
        self.dev_opac_hcn_wg_lay = None
        self.dev_opac_c2h2_wg_lay = None
        self.dev_opac_tio_wg_lay = None
        self.dev_opac_vo_wg_lay = None
        self.dev_opac_h2s_wg_lay = None
        self.dev_opac_na_wg_lay = None
        self.dev_opac_k_wg_lay = None
        self.dev_opac_cia_h2h2_wg_lay = None
        self.dev_opac_cia_h2he_wg_lay = None
        self.contr_h2o = None
        self.dev_contr_h2o = None
        self.contr_co2 = None
        self.dev_contr_co2= None
        self.contr_co = None
        self.dev_contr_co = None
        self.contr_ch4 = None
        self.dev_contr_ch4 = None
        self.contr_nh3 = None
        self.dev_contr_nh3 = None
        self.contr_hcn = None
        self.dev_contr_hcn = None
        self.contr_c2h2 = None
        self.dev_contr_c2h2 = None
        self.contr_h2s = None
        self.dev_contr_h2s = None
        self.contr_na = None
        self.dev_contr_na = None
        self.contr_k = None
        self.dev_contr_k = None
        self.contr_cia_h2h2 = None
        self.dev_contr_cia_h2h2 = None
        self.contr_cia_h2he = None
        self.dev_contr_cia_h2he = None
        self.contr_rayleigh = None
        self.dev_contr_rayleigh = None
        self.contr_cloud = None
        self.dev_contr_cloud = None

        # only used by Vmod
        self.dev_f_h2o_int = None
        self.dev_f_co2_int = None
        self.dev_f_co_int = None
        self.dev_f_ch4_int = None
        self.dev_f_nh3_int = None
        self.dev_f_hcn_int = None
        self.dev_f_c2h2_int = None
        self.dev_f_tio_int = None
        self.dev_f_vo_int = None
        self.dev_f_h2_int = None
        self.dev_f_he_int = None
        self.dev_opac_h2o_wg_int = None
        self.dev_opac_co2_wg_int = None
        self.dev_opac_co_wg_int = None
        self.dev_opac_ch4_wg_int = None
        self.dev_opac_nh3_wg_int = None
        self.dev_opac_hcn_wg_int = None
        self.dev_opac_c2h2_wg_int = None
        self.dev_opac_tio_wg_int = None
        self.dev_opac_vo_wg_int = None
        self.dev_opac_h2s_wg_int = None
        self.dev_opac_na_wg_int = None
        self.dev_opac_k_wg_int = None
        self.dev_opac_cia_h2h2_wg_int = None
        self.dev_opac_cia_h2he_wg_int = None

    def convert_input_list_to_array(self, Vmod):
        """ converts lists of quantities to arrays """

        self.p_lay = np.array(self.p_lay, self.fl_prec)
        self.p_int = np.array(self.p_int, self.fl_prec)
        self.delta_colmass = np.array(self.delta_colmass, self.fl_prec)
        self.delta_col_upper = np.array(self.delta_col_upper, self.fl_prec)
        self.delta_col_lower = np.array(self.delta_col_lower, self.fl_prec)
        self.ktemp = np.array(self.ktemp, self.fl_prec)
        self.kpress = np.array(self.kpress, self.fl_prec)
        self.entr_temp = np.array(self.entr_temp, self.fl_prec)
        self.entr_press = np.array(self.entr_press, self.fl_prec)
        self.opac_k = np.array(self.opac_k, self.fl_prec)
        self.opac_y = np.array(self.opac_y, self.fl_prec)
        self.gauss_weight = np.array(self.gauss_weight, self.fl_prec)
        self.opac_wave = np.array(self.opac_wave, self.fl_prec)
        self.opac_deltawave = np.array(self.opac_deltawave, self.fl_prec)
        self.opac_interwave = np.array(self.opac_interwave, self.fl_prec)
        self.opac_scat_cross = np.array(self.opac_scat_cross, self.fl_prec)
        self.opac_meanmass = np.array(self.opac_meanmass, self.fl_prec)
        self.opac_kappa = np.array(self.opac_kappa, self.fl_prec)
        self.opac_entropy = np.array(self.opac_entropy, self.fl_prec)
        self.entr_kappa = np.array(self.entr_kappa, self.fl_prec)
        self.entr_c_p = np.array(self.entr_c_p, self.fl_prec)
        self.entr_entropy = np.array(self.entr_entropy, self.fl_prec)
        self.entr_phase_number = np.array(self.entr_phase_number, self.fl_prec)
        self.starflux = np.array(self.starflux, self.fl_prec)
        self.T_lay = np.array(self.T_lay, self.fl_prec)
        self.abs_cross_cloud = np.array(self.abs_cross_cloud, self.fl_prec)
        self.scat_cross_cloud = np.array(self.scat_cross_cloud, self.fl_prec)
        self.g_0_cloud = np.array(self.g_0_cloud, self.fl_prec)

        # used for Vmod and molecular transmission function
        self.opac_k_h2o = np.array(self.opac_k_h2o, self.fl_prec)
        self.opac_k_co2 = np.array(self.opac_k_co2, self.fl_prec)
        self.opac_k_co = np.array(self.opac_k_co, self.fl_prec)
        self.opac_k_ch4 = np.array(self.opac_k_ch4, self.fl_prec)
        self.opac_k_nh3 = np.array(self.opac_k_nh3, self.fl_prec)
        self.opac_k_hcn = np.array(self.opac_k_hcn, self.fl_prec)
        self.opac_k_c2h2 = np.array(self.opac_k_c2h2, self.fl_prec)
        self.opac_k_tio = np.array(self.opac_k_tio, self.fl_prec)
        self.opac_k_vo = np.array(self.opac_k_vo, self.fl_prec)
        self.opac_k_h2s = np.array(self.opac_k_h2s, self.fl_prec)
        self.opac_k_na = np.array(self.opac_k_na, self.fl_prec)
        self.opac_k_k = np.array(self.opac_k_k, self.fl_prec)
        self.opac_k_cia_h2h2 = np.array(self.opac_k_cia_h2h2, self.fl_prec)
        self.opac_k_cia_h2he = np.array(self.opac_k_cia_h2he, self.fl_prec)

        if Vmod.V_iter_nr > 0:
            self.f_h2o_lay = np.array(self.f_h2o_lay, self.fl_prec)
            self.f_co2_lay = np.array(self.f_co2_lay, self.fl_prec)
            self.f_co_lay = np.array(self.f_co_lay, self.fl_prec)
            self.f_ch4_lay = np.array(self.f_ch4_lay, self.fl_prec)
            self.f_nh3_lay = np.array(self.f_nh3_lay, self.fl_prec)
            self.f_hcn_lay = np.array(self.f_hcn_lay, self.fl_prec)
            self.f_c2h2_lay = np.array(self.f_c2h2_lay, self.fl_prec)
            self.f_tio_lay = np.array(self.f_tio_lay, self.fl_prec)
            self.f_vo_lay = np.array(self.f_vo_lay, self.fl_prec)
            self.f_h2s_lay = np.array(self.f_h2s_lay, self.fl_prec)
            self.f_h2_lay = np.array(self.f_h2_lay, self.fl_prec)
            self.f_he_lay = np.array(self.f_he_lay, self.fl_prec)
            self.meanmolmass_lay = np.array(self.meanmolmass_lay, self.fl_prec)

        # DISCONTINUED --- used for molecular transmission function
        # self.f_h2o_tab = np.array(self.f_h2o_tab, self.fl_prec)
        # self.f_co2_tab = np.array(self.f_co2_tab, self.fl_prec)
        # self.f_co_tab = np.array(self.f_co_tab, self.fl_prec)
        # self.f_ch4_tab = np.array(self.f_ch4_tab, self.fl_prec)
        # self.f_nh3_tab = np.array(self.f_nh3_tab, self.fl_prec)
        # self.f_hcn_tab = np.array(self.f_hcn_tab, self.fl_prec)
        # self.f_c2h2_tab = np.array(self.f_c2h2_tab, self.fl_prec)
        # self.f_h2s_tab = np.array(self.f_h2s_tab, self.fl_prec)
        # self.f_h2_tab = np.array(self.f_h2_tab, self.fl_prec)
        # self.f_he_tab = np.array(self.f_he_tab, self.fl_prec)
        # self.f_na_tab = np.array(self.f_na_tab, self.fl_prec)
        # self.f_k_tab = np.array(self.f_k_tab, self.fl_prec)

    def dimensions(self):
        """ create the correct dimensions of the grid from input parameters """

        self.nlayer_nbin = np.int32(self.nlayer * self.nbin)
        self.nlayer_plus2_nbin = np.int32((self.nlayer+2) * self.nbin)
        self.ninterface_nbin = np.int32(self.ninterface * self.nbin)
        self.ninterface_wg_nbin = np.int32(self.ninterface * self.ny * self.nbin)
        self.nlayer_wg_nbin = np.int32(self.ninterface * self.ny * self.nbin)
        self.nplanck_grid = np.int32((self.plancktable_dim+1) * self.nbin)

    def create_zero_arrays(self, Vmod):
        """ creates zero arrays of quantities to be used on the GPU with the correct length/dimension """

        self.F_up_band = np.zeros(self.ninterface_nbin, self.fl_prec)
        self.F_down_band = np.zeros(self.ninterface_nbin, self.fl_prec)
        self.F_dir_band = np.zeros(self.ninterface_nbin, self.fl_prec)
        self.F_up_wg = np.zeros(self.ninterface_wg_nbin, self.fl_prec)
        self.F_down_wg = np.zeros(self.ninterface_wg_nbin, self.fl_prec)
        self.F_dir_wg = np.zeros(self.ninterface_wg_nbin, self.fl_prec)
        self.Fc_up_wg = np.zeros(self.ninterface_wg_nbin, self.fl_prec)
        self.Fc_down_wg = np.zeros(self.ninterface_wg_nbin, self.fl_prec)
        self.Fc_dir_wg = np.zeros(self.ninterface_wg_nbin, self.fl_prec)
        self.F_up_tot = np.zeros(self.ninterface, self.fl_prec)
        self.F_down_tot = np.zeros(self.ninterface, self.fl_prec)
        self.F_dir_tot = np.zeros(self.ninterface, self.fl_prec)
        self.opac_band_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.scat_cross_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.F_net = np.zeros(self.ninterface, self.fl_prec)
        self.F_net_diff = np.zeros(self.nlayer, self.fl_prec)
        self.planckband_lay = np.zeros(self.nlayer_plus2_nbin, self.fl_prec)
        self.planckband_int = np.zeros(self.ninterface_nbin, self.fl_prec)
        self.planck_opac_T_pl = np.zeros(self.nlayer, self.fl_prec)
        self.ross_opac_T_pl = np.zeros(self.nlayer, self.fl_prec)
        self.planck_opac_T_star = np.zeros(self.nlayer, self.fl_prec)
        self.ross_opac_T_star = np.zeros(self.nlayer, self.fl_prec)
        self.trans_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.delta_tau_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.abort = np.zeros(self.nlayer + 1, np.int32) # including "ghost layer" below
        self.c_p_lay = np.zeros(self.nlayer, self.fl_prec)
        self.test_arr = np.zeros(self.nlayer, self.fl_prec)
        self.kappa_lay = np.zeros(self.nlayer, self.fl_prec)
        self.kappa_int = np.zeros(self.ninterface, self.fl_prec)
        self.entropy_lay = np.zeros(self.nlayer, self.fl_prec)
        self.phase_number_lay = np.zeros(self.nlayer, self.fl_prec)
        self.trans_weight_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.contr_func_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.cloud_opac_lay = np.zeros(self.nlayer, self.fl_prec)
        self.cloud_opac_int = np.zeros(self.ninterface, self.fl_prec)  # to have zeros when reading in later
        self.cloud_scat_cross_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.cloud_scat_cross_int = np.zeros(self.ninterface_nbin, self.fl_prec)  # to have zeros when reading in later
        self.g_0_tot_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.g_0_tot_int = np.zeros(self.ninterface_nbin, self.fl_prec)  # to have zeros when reading in later
        self.delta_z_lay = np.zeros(self.nlayer, self.fl_prec)
        self.z_lay = np.zeros(self.nlayer, self.fl_prec)
        self.contr_h2o = np.zeros(self.nbin, self.fl_prec)
        self.contr_co2 = np.zeros(self.nbin, self.fl_prec)
        self.contr_co = np.zeros(self.nbin, self.fl_prec)
        self.contr_ch4 = np.zeros(self.nbin, self.fl_prec)
        self.contr_nh3 = np.zeros(self.nbin, self.fl_prec)
        self.contr_hcn = np.zeros(self.nbin, self.fl_prec)
        self.contr_c2h2 = np.zeros(self.nbin, self.fl_prec)
        self.contr_h2s = np.zeros(self.nbin, self.fl_prec)
        self.contr_na = np.zeros(self.nbin, self.fl_prec)
        self.contr_k = np.zeros(self.nbin, self.fl_prec)
        self.contr_cia_h2h2 = np.zeros(self.nbin, self.fl_prec)
        self.contr_cia_h2he = np.zeros(self.nbin, self.fl_prec)
        self.contr_rayleigh = np.zeros(self.nbin, self.fl_prec)
        self.contr_cloud = np.zeros(self.nbin, self.fl_prec)

        if Vmod.V_iter_nr == 0: ## otherwise already filled with values
            self.meanmolmass_lay = np.zeros(self.nlayer, self.fl_prec)
            self.f_h2o_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_co2_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_co_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_ch4_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_nh3_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_hcn_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_c2h2_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_tio_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_vo_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_h2s_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_na_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_k_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_h2_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
            self.f_he_lay = np.zeros(self.nlayer_nbin, self.fl_prec)

        # arrays to be used purely on the CPU
        self.conv_layer = np.zeros(self.nlayer + 1, np.int32)

    def copy_host_to_device(self, Vmod):
        """ copies relevant host arrays to device """

        # input arrays
        self.dev_p_lay = gpuarray.to_gpu(self.p_lay)
        self.dev_p_int = gpuarray.to_gpu(self.p_int)
        self.dev_delta_colmass = gpuarray.to_gpu(self.delta_colmass)
        self.dev_delta_col_upper = gpuarray.to_gpu(self.delta_col_upper)
        self.dev_delta_col_lower = gpuarray.to_gpu(self.delta_col_lower)
        self.dev_ktemp = gpuarray.to_gpu(self.ktemp)
        self.dev_kpress = gpuarray.to_gpu(self.kpress)
        self.dev_entr_temp = gpuarray.to_gpu(self.entr_temp)
        self.dev_entr_press = gpuarray.to_gpu(self.entr_press)
        self.dev_entr_kappa = gpuarray.to_gpu(self.entr_kappa)
        self.dev_entr_c_p = gpuarray.to_gpu(self.entr_c_p)
        self.dev_entr_phase_number = gpuarray.to_gpu(self.entr_phase_number)
        self.dev_entr_entropy = gpuarray.to_gpu(self.entr_entropy)
        self.dev_opac_k = gpuarray.to_gpu(self.opac_k)
        self.dev_opac_y = gpuarray.to_gpu(self.opac_y)
        self.dev_gauss_weight = gpuarray.to_gpu(self.gauss_weight)
        self.dev_opac_wave = gpuarray.to_gpu(self.opac_wave)
        self.dev_opac_deltawave = gpuarray.to_gpu(self.opac_deltawave)
        self.dev_opac_interwave = gpuarray.to_gpu(self.opac_interwave)
        self.dev_opac_scat_cross = gpuarray.to_gpu(self.opac_scat_cross)
        self.dev_opac_meanmass = gpuarray.to_gpu(self.opac_meanmass)
        self.dev_starflux = gpuarray.to_gpu(self.starflux)
        self.dev_T_lay = gpuarray.to_gpu(self.T_lay)
        self.dev_abs_cross_cloud = gpuarray.to_gpu(self.abs_cross_cloud)
        self.dev_scat_cross_cloud = gpuarray.to_gpu(self.scat_cross_cloud)
        self.dev_g_0_cloud = gpuarray.to_gpu(self.g_0_cloud)

        # used for Vmod
        if Vmod.V_coupling == 1:
            self.dev_opac_k_h2o = gpuarray.to_gpu(self.opac_k_h2o)
            self.dev_opac_k_co2 = gpuarray.to_gpu(self.opac_k_co2)
            self.dev_opac_k_co = gpuarray.to_gpu(self.opac_k_co)
            self.dev_opac_k_ch4 = gpuarray.to_gpu(self.opac_k_ch4)
            self.dev_opac_k_nh3 = gpuarray.to_gpu(self.opac_k_nh3)
            self.dev_opac_k_hcn = gpuarray.to_gpu(self.opac_k_hcn)
            self.dev_opac_k_c2h2 = gpuarray.to_gpu(self.opac_k_c2h2)
            self.dev_opac_k_tio = gpuarray.to_gpu(self.opac_k_tio)
            self.dev_opac_k_vo = gpuarray.to_gpu(self.opac_k_vo)
            # self.dev_opac_k_h2s = gpuarray.to_gpu(self.opac_k_h2s)
            # self.dev_opac_k_na = gpuarray.to_gpu(self.opac_k_na)
            # self.dev_opac_k_k = gpuarray.to_gpu(self.opac_k_k)
            self.dev_opac_k_cia_h2h2 = gpuarray.to_gpu(self.opac_k_cia_h2h2)
            self.dev_opac_k_cia_h2he = gpuarray.to_gpu(self.opac_k_cia_h2he)

            # used for Vmod
            self.dev_f_h2o_lay = gpuarray.to_gpu(self.f_h2o_lay)
            self.dev_f_co2_lay = gpuarray.to_gpu(self.f_co2_lay)
            self.dev_f_co_lay = gpuarray.to_gpu(self.f_co_lay)
            self.dev_f_ch4_lay = gpuarray.to_gpu(self.f_ch4_lay)
            self.dev_f_nh3_lay = gpuarray.to_gpu(self.f_nh3_lay)
            self.dev_f_hcn_lay = gpuarray.to_gpu(self.f_hcn_lay)
            self.dev_f_c2h2_lay = gpuarray.to_gpu(self.f_c2h2_lay)
            self.dev_f_tio_lay = gpuarray.to_gpu(self.f_tio_lay)
            self.dev_f_vo_lay = gpuarray.to_gpu(self.f_vo_lay)
            # self.dev_f_h2s_lay = gpuarray.to_gpu(self.f_h2s_lay)
            self.dev_f_h2_lay = gpuarray.to_gpu(self.f_h2_lay)
            self.dev_f_he_lay = gpuarray.to_gpu(self.f_he_lay)
            self.dev_meanmolmass_lay = gpuarray.to_gpu(self.meanmolmass_lay)
            # self.dev_f_na_lay = gpuarray.to_gpu(self.f_na_lay)
            # self.dev_f_k_lay = gpuarray.to_gpu(self.f_k_lay)

        # DISCONTINUED -- used for mol. transmission
        # self.dev_f_h2o_tab = gpuarray.to_gpu(self.f_h2o_tab)
        # self.dev_f_co2_tab = gpuarray.to_gpu(self.f_co2_tab)
        # self.dev_f_co_tab = gpuarray.to_gpu(self.f_co_tab)
        # self.dev_f_ch4_tab = gpuarray.to_gpu(self.f_ch4_tab)
        # self.dev_f_nh3_tab = gpuarray.to_gpu(self.f_nh3_tab)
        # self.dev_f_hcn_tab = gpuarray.to_gpu(self.f_hcn_tab)
        # self.dev_f_c2h2_tab = gpuarray.to_gpu(self.f_c2h2_tab)
        # self.dev_f_h2s_tab = gpuarray.to_gpu(self.f_h2s_tab)
        # self.dev_f_h2_tab = gpuarray.to_gpu(self.f_h2_tab)
        # self.dev_f_he_tab = gpuarray.to_gpu(self.f_he_tab)
        # self.dev_f_na_tab = gpuarray.to_gpu(self.f_na_tab)
        # self.dev_f_k_tab = gpuarray.to_gpu(self.f_k_tab)



        # zero arrays (copying anyway to obtain the gpuarray functionality)
        # those arrays will be copied to host at the end of computation or need to be zero-filled
        self.dev_F_up_band = gpuarray.to_gpu(self.F_up_band)
        self.dev_F_down_band = gpuarray.to_gpu(self.F_down_band)
        self.dev_F_dir_band = gpuarray.to_gpu(self.F_dir_band)
        self.dev_F_down_wg = gpuarray.to_gpu(self.F_down_wg)
        self.dev_F_up_wg = gpuarray.to_gpu(self.F_up_wg)
        self.dev_F_dir_wg = gpuarray.to_gpu(self.F_dir_wg)
        self.dev_F_up_tot = gpuarray.to_gpu(self.F_up_tot)
        self.dev_F_down_tot = gpuarray.to_gpu(self.F_down_tot)
        self.dev_F_dir_tot = gpuarray.to_gpu(self.F_dir_tot)
        self.dev_opac_band_lay = gpuarray.to_gpu(self.opac_band_lay)
        self.dev_scat_cross_lay = gpuarray.to_gpu(self.scat_cross_lay)
        self.dev_F_net = gpuarray.to_gpu(self.F_net)
        self.dev_F_net_diff = gpuarray.to_gpu(self.F_net_diff)
        self.dev_planckband_lay = gpuarray.to_gpu(self.planckband_lay)
        self.dev_planck_opac_T_pl = gpuarray.to_gpu(self.planck_opac_T_pl)
        self.dev_ross_opac_T_pl = gpuarray.to_gpu(self.ross_opac_T_pl)
        self.dev_planck_opac_T_star = gpuarray.to_gpu(self.planck_opac_T_star)
        self.dev_ross_opac_T_star = gpuarray.to_gpu(self.ross_opac_T_star)
        self.dev_trans_band = gpuarray.to_gpu(self.trans_band)
        self.dev_delta_tau_band = gpuarray.to_gpu(self.delta_tau_band)
        self.dev_abort = gpuarray.to_gpu(self.abort)
        self.dev_meanmolmass_lay = gpuarray.to_gpu(self.meanmolmass_lay)
        self.dev_c_p_lay = gpuarray.to_gpu(self.c_p_lay)
        self.dev_kappa_lay = gpuarray.to_gpu(self.kappa_lay)
        self.dev_kappa_int = gpuarray.to_gpu(self.kappa_int)
        self.dev_entropy_lay = gpuarray.to_gpu(self.entropy_lay)
        self.dev_phase_number_lay = gpuarray.to_gpu(self.phase_number_lay)
        self.dev_trans_weight_band = gpuarray.to_gpu(self.trans_weight_band)
        self.dev_contr_func_band = gpuarray.to_gpu(self.contr_func_band)
        self.dev_cloud_opac_lay = gpuarray.to_gpu(self.cloud_opac_lay)
        self.dev_cloud_opac_int = gpuarray.to_gpu(self.cloud_opac_int)
        self.dev_cloud_scat_cross_lay = gpuarray.to_gpu(self.cloud_scat_cross_lay)
        self.dev_delta_z_lay = gpuarray.to_gpu(self.delta_z_lay)
        self.dev_z_lay = gpuarray.to_gpu(self.z_lay)
        self.dev_g_0_tot_lay = gpuarray.to_gpu(self.g_0_tot_lay)
        # self.dev_test_arr = gpuarray.to_gpu(self.test_arr)

        # OUTDATED -- used for mol. contribution
        # self.dev_contr_h2o = gpuarray.to_gpu(self.contr_h2o)
        # self.dev_contr_co2 = gpuarray.to_gpu(self.contr_co2)
        # self.dev_contr_co = gpuarray.to_gpu(self.contr_co)
        # self.dev_contr_ch4 = gpuarray.to_gpu(self.contr_ch4)
        # self.dev_contr_nh3 = gpuarray.to_gpu(self.contr_nh3)
        # self.dev_contr_hcn = gpuarray.to_gpu(self.contr_hcn)
        # self.dev_contr_c2h2 = gpuarray.to_gpu(self.contr_c2h2)
        # self.dev_contr_h2s = gpuarray.to_gpu(self.contr_h2s)
        # self.dev_contr_na = gpuarray.to_gpu(self.contr_na)
        # self.dev_contr_k = gpuarray.to_gpu(self.contr_k)
        # self.dev_contr_cia_h2h2 = gpuarray.to_gpu(self.contr_cia_h2h2)
        # self.dev_contr_cia_h2he = gpuarray.to_gpu(self.contr_cia_h2he)
        # self.dev_contr_rayleigh = gpuarray.to_gpu(self.contr_rayleigh)
        # self.dev_contr_cloud = gpuarray.to_gpu(self.contr_cloud)

        if self.iso == 0:
            self.dev_planckband_int = gpuarray.to_gpu(self.planckband_int)
            self.dev_Fc_down_wg = gpuarray.to_gpu(self.Fc_down_wg)
            self.dev_Fc_up_wg = gpuarray.to_gpu(self.Fc_up_wg)
            self.dev_Fc_dir_wg = gpuarray.to_gpu(self.Fc_dir_wg)
            self.dev_g_0_tot_int = gpuarray.to_gpu(self.g_0_tot_int)
            self.dev_cloud_scat_cross_int = gpuarray.to_gpu(self.cloud_scat_cross_int)

    def copy_device_to_host(self):
        """ copies relevant device arrays to host """

        self.delta_colmass = self.dev_delta_colmass.get()
        self.F_up_band = self.dev_F_up_band.get()
        self.F_down_band = self.dev_F_down_band.get()
        self.F_dir_band = self.dev_F_dir_band.get()
        self.F_up_tot = self.dev_F_up_tot.get()
        self.F_down_tot = self.dev_F_down_tot.get()
        self.F_dir_tot = self.dev_F_dir_tot.get()
        self.opac_band_lay = self.dev_opac_band_lay.get()
        self.scat_cross_lay = self.dev_scat_cross_lay.get()
        self.F_net = self.dev_F_net.get()
        self.F_net_diff = self.dev_F_net_diff.get()
        self.p_lay = self.dev_p_lay.get()
        self.p_int = self.dev_p_int.get()
        self.T_lay = self.dev_T_lay.get()
        self.planckband_lay = self.dev_planckband_lay.get()
        self.planck_opac_T_pl = self.dev_planck_opac_T_pl.get()
        self.ross_opac_T_pl = self.dev_ross_opac_T_pl.get()
        self.planck_opac_T_star = self.dev_planck_opac_T_star.get()
        self.ross_opac_T_star = self.dev_ross_opac_T_star.get()
        self.trans_band = self.dev_trans_band.get()
        self.delta_tau_band = self.dev_delta_tau_band.get()
        self.meanmolmass_lay = self.dev_meanmolmass_lay.get()
        self.c_p_lay = self.dev_c_p_lay.get()
        self.kappa_lay = self.dev_kappa_lay.get()
        self.kappa_int = self.dev_kappa_int.get()
        self.entropy_lay = self.dev_entropy_lay.get()
        self.phase_number_lay = self.dev_phase_number_lay.get()
        self.trans_weight_band = self.dev_trans_weight_band.get()
        self.contr_func_band = self.dev_contr_func_band.get()
        self.cloud_opac_lay = self.dev_cloud_opac_lay.get()
        self.cloud_scat_cross_lay = self.dev_cloud_scat_cross_lay.get()
        self.g_0_tot_lay = self.dev_g_0_tot_lay.get()
        self.delta_z_lay = self.dev_delta_z_lay.get()
        self.z_lay = self.dev_z_lay.get()

        # OUTDATED -- molecular contribution function
        # self.contr_h2o = self.dev_contr_h2o.get()
        # self.contr_co2 = self.dev_contr_co2.get()
        # self.contr_co = self.dev_contr_co.get()
        # self.contr_ch4 = self.dev_contr_ch4.get()
        # self.contr_nh3 = self.dev_contr_nh3.get()
        # self.contr_hcn = self.dev_contr_hcn.get()
        # self.contr_c2h2 = self.dev_contr_c2h2.get()
        # self.contr_h2s = self.dev_contr_h2s.get()
        # self.contr_na = self.dev_contr_na.get()
        # self.contr_k = self.dev_contr_k.get()
        # self.contr_cia_h2h2 = self.dev_contr_cia_h2h2.get()
        # self.contr_cia_h2he = self.dev_contr_cia_h2he.get()
        # self.contr_rayleigh = self.dev_contr_rayleigh.get()
        # self.contr_cloud = self.dev_contr_cloud.get()

        if self.iso == 0:
            self.planckband_int = self.dev_planckband_int.get()

    def allocate_on_device(self, Vmod):
        """ allocate memory for arrays existing only on the GPU """

        # mem_alloc wants the number of bytes, which is 8 bytes per value for double precision (e.g. np.float64)
        # and 4 bytes per value for single precision (e.g. np.int32, np.float32)
        # save the correct space depending on precision

        size_ninterface = int(self.ninterface * self.nr_bytes)
        size_nlayer = int(self.nlayer * self.nr_bytes)
        size_nlayer_plus1 = int((self.nlayer+1) * self.nr_bytes)
        size_nlayer_nbin = int(self.nlayer_nbin * self.nr_bytes)
        size_ninterface_nbin = int(self.ninterface_nbin * self.nr_bytes)
        size_nlayer_wg_nbin = int(self.nlayer_wg_nbin * self.nr_bytes)
        size_ninterface_wg_nbin = int(self.ninterface_wg_nbin * self.nr_bytes)
        size_nplanckgrid = int(self.nplanck_grid * self.nr_bytes)

        # these arrays will never be copied between host and device.
        # Hence the normal mem_alloc functionality
        self.dev_T_int = cuda.mem_alloc(size_ninterface)
        self.dev_delta_t_prefactor = cuda.mem_alloc(size_nlayer_plus1)
        self.dev_T_store = cuda.mem_alloc(size_nlayer_plus1)
        self.dev_planckband_grid = cuda.mem_alloc(size_nplanckgrid)
        self.dev_opac_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_meanmolmass_int = cuda.mem_alloc(size_ninterface)
        self.dev_delta_tau_wg = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_trans_wg = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_w_0 = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_M_term = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_N_term = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_P_term = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_G_plus = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_G_minus = cuda.mem_alloc(size_nlayer_wg_nbin)

        if self.iso == 0:
            self.dev_opac_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
            self.dev_scat_cross_int = cuda.mem_alloc(size_ninterface_nbin)
            self.dev_delta_tau_wg_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_delta_tau_wg_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_trans_wg_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_trans_wg_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_M_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_N_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_P_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_M_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_N_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_P_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_w_0_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_w_0_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_G_plus_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_G_plus_lower = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_G_minus_upper = cuda.mem_alloc(size_nlayer_wg_nbin)
            self.dev_G_minus_lower = cuda.mem_alloc(size_nlayer_wg_nbin)

        if Vmod.V_coupling == 1:

            if Vmod.V_iter_nr > 0:

                self.dev_opac_h2o_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_co2_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_co_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_ch4_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_nh3_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_hcn_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_c2h2_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_tio_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_vo_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_h2s_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_na_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_k_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_cia_h2h2_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_opac_cia_h2he_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)

                if self.iso == 0:

                    self.dev_f_h2o_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_co2_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_co_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_ch4_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_nh3_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_hcn_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_c2h2_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_tio_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_vo_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_h2_int = cuda.mem_alloc(size_ninterface)
                    self.dev_f_he_int = cuda.mem_alloc(size_ninterface)
                    self.dev_opac_h2o_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_co2_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_co_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_ch4_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_nh3_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_hcn_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_c2h2_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_tio_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_vo_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_h2s_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_na_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_k_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_cia_h2h2_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)
                    self.dev_opac_cia_h2he_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)

if __name__ == "__main__":
    print("This module is for storing and allocating all the necessary quantities used in HELIOS"
          "...hopefully without memory bugs. ")
