# ==============================================================================
# Module for storing all the necessary quantities like parameters,
# arrays, input & output data, etc.
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
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray


class Store(object):
    """ class that stores parameters, quantities, arrays, etc., used in the HELIOS code """

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
        self.scat = None
        self.diffusivity = None
        self.convection = None
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
        self.w_0_scat_limit = None
        self.delta_tau_limit = None
        self.rad_convergence_limit = None
        self.global_limit = None
        self.n_plot = None
        self.energy_correction = None
        self.star_corr_factor = np.int32(1)
        self.input_dampara = None
        self.dampara = None
        self.F_intern = None
        self.adapt_interval = None
        self.smooth = None
        self.geom_zenith_corr = None
        self.scat_corr = None
        self.input_kappa_value = None
        self.approx_f = None
        self.tau_lw = 1
        self.planet_type = None
        self.F_sens = 0
        self.debug = None
        self.kappa_file_format = np.int32(0)
        self.i2s_transition = None
        self.flux_calc_method = None
        self.relaxed_criterion_trigger = 0
        self.clouds = None
        self.add_heating = None
        self.add_heating_path = None
        self.add_heating_file_header_lines = None
        self.add_heating_file_press_name = None
        self.add_heating_file_press_unit = None
        self.add_heating_file_data_name = None
        self.add_heating_file_data_conv_factor = None
        self.no_atmo_mode = np.int32(0)
        self.physical_tstep = None
        self.runtime_limit = None
        self.force_start_tp_from_file = None
        self.plancktable_dim = None
        self.plancktable_step = None
        self.kcoeff_mixing = None
        self.opacity_mixing = None
        self.coupling = None
        self.coupling_full_output = None
        self.coupling_speed_up = None
        self.coupling_iter_nr = None
        self.coupl_tp_write_interval = None
        self.coupl_convergence_limit = None
        self.max_nr_iterations = None


        # arrays/lists exclusively used on the CPU
        self.T_restart = []
        self.conv_unstable = None
        self.F_net_conv = []
        self.F_ratio = []
        self.marked_red = None
        self.converged = None
        self.add_heat_dens = None
        self.f_all_clouds_lay = None
        self.f_all_clouds_int = None
        self.species_list = []
        self.crit_relaxation_numbers = None

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
        self.opac_k = None
        self.dev_opac_k = None
        self.gauss_y = None
        self.dev_gauss_y = None
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
        self.entr_kappa = []
        self.dev_entr_kappa = None
        self.entr_c_p = []
        self.dev_entr_c_p = None
        self.entr_phase_number = []
        self.dev_entr_phase_number = None
        self.entr_entropy = []
        self.dev_entr_entropy = None
        self.starflux = None
        self.dev_starflux = None
        self.conv_layer = None
        self.dev_conv_layer = None
        self.surf_albedo = None
        self.dev_surf_albedo = None
        self.abs_cross_all_clouds_lay = None
        self.dev_abs_cross_all_clouds_lay = None
        self.abs_cross_all_clouds_int = None
        self.dev_abs_cross_all_clouds_int = None
        self.scat_cross_all_clouds_lay = None
        self.dev_scat_cross_all_clouds_lay = None
        self.scat_cross_all_clouds_int = None
        self.dev_scat_cross_all_clouds_int = None
        self.g_0_all_clouds_lay = None
        self.dev_g_0_all_clouds_lay = None
        self.g_0_all_clouds_int = None
        self.dev_g_0_all_clouds_int = None

        # arrays to be filled with species data in the on-the-fly opacity mixing mode (i.e., CPU --> GPU)
        self.dev_vmr_spec_lay = None
        self.dev_vmr_spec_int = None
        self.dev_opacity_spec_pretab = None
        self.dev_scat_cross_spec_lay = None
        self.dev_scat_cross_spec_int = None

        # arrays to be copied CPU --> GPU --> CPU
        # these are copied to GPU by "gpuarray" and copied back
        self.T_lay = []
        self.dev_T_lay = None
        self.abort = None
        self.dev_abort = None
        self.c_p_lay = None
        self.dev_c_p_lay = None
        self.kappa_lay = None
        self.dev_kappa_lay = None
        self.kappa_int = None
        self.dev_kappa_int = None

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
        self.dev_scat_cross_int = None
        self.opac_wg_lay = None
        self.dev_opac_wg_lay = None
        self.opac_wg_int = None
        self.dev_opac_wg_int = None
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
        self.meanmolmass_int = None
        self.dev_meanmolmass_int = None
        self.entropy_lay = None
        self.dev_entropy_lay = None
        self.trans_weight_band = None
        self.dev_trans_weight_band = None
        self.contr_func_band = None
        self.dev_contr_func_band = None
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
        self.T_int = None
        self.dev_T_int = None
        self.scat_trigger = None
        self.dev_scat_trigger = None
        self.delta_tau_all_clouds = None
        self.dev_delta_tau_all_clouds = None
        self.F_add_heat_lay = None
        self.dev_F_add_heat_lay = None
        self.F_add_heat_sum = None
        self.dev_F_add_heat_sum = None
        self.F_smooth = None
        self.dev_F_smooth = None
        self.F_smooth_sum = None
        self.dev_F_smooth_sum = None

        # arrays exclusively used on the GPU
        # these are defined directly on the GPU and stay there. No copying required.
        self.dev_delta_t_prefactor = None
        self.dev_T_store = None
        self.dev_planckband_grid = None
        self.dev_opac_int = None
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
        self.dev_delta_tau_all_clouds_upper = None
        self.dev_delta_tau_all_clouds_lower = None

        # for on-the-fly opacity mixing mode (only GPU arrays)
        self.dev_opac_spec_wg_lay = None
        self.dev_opac_spec_wg_int = None

        # for matrix method
        self.dev_alpha = None
        self.dev_beta = None
        self.dev_source_term_down = None
        self.dev_source_term_up = None
        self.dev_c_prime = None
        self.dev_d_prime = None

    def convert_input_list_to_array(self):
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
        self.gauss_y = np.array(self.gauss_y, self.fl_prec)
        self.gauss_weight = np.array(self.gauss_weight, self.fl_prec)
        self.opac_wave = np.array(self.opac_wave, self.fl_prec)
        self.opac_deltawave = np.array(self.opac_deltawave, self.fl_prec)
        self.opac_interwave = np.array(self.opac_interwave, self.fl_prec)
        self.opac_scat_cross = np.array(self.opac_scat_cross, self.fl_prec)
        self.opac_meanmass = np.array(self.opac_meanmass, self.fl_prec)
        self.entr_kappa = np.array(self.entr_kappa, self.fl_prec)
        self.entr_c_p = np.array(self.entr_c_p, self.fl_prec)
        self.entr_entropy = np.array(self.entr_entropy, self.fl_prec)
        self.entr_phase_number = np.array(self.entr_phase_number, self.fl_prec)
        self.starflux = np.array(self.starflux, self.fl_prec)
        self.T_lay = np.array(self.T_lay, self.fl_prec)
        self.surf_albedo = np.array(self.surf_albedo, self.fl_prec)
        self.abs_cross_all_clouds_lay = np.array(self.abs_cross_all_clouds_lay, self.fl_prec)
        self.abs_cross_all_clouds_int = np.array(self.abs_cross_all_clouds_int, self.fl_prec)
        self.scat_cross_all_clouds_lay = np.array(self.scat_cross_all_clouds_lay, self.fl_prec)
        self.scat_cross_all_clouds_int = np.array(self.scat_cross_all_clouds_int, self.fl_prec)
        self.g_0_all_clouds_lay = np.array(self.g_0_all_clouds_lay, self.fl_prec)
        self.g_0_all_clouds_int = np.array(self.g_0_all_clouds_int, self.fl_prec)

    def dimensions(self):
        """ create the correct dimensions of the grid from input parameters """

        self.nlayer_nbin = np.int32(self.nlayer * self.nbin)
        self.nlayer_plus2_nbin = np.int32((self.nlayer+2) * self.nbin)
        self.ninterface_nbin = np.int32(self.ninterface * self.nbin)
        self.ninterface_wg_nbin = np.int32(self.ninterface * self.ny * self.nbin)
        self.nlayer_wg_nbin = np.int32(self.ninterface * self.ny * self.nbin)
        self.wg_nbin = np.int32(self.ny * self.nbin)
        self.nplanck_grid = np.int32((self.plancktable_dim+1) * self.nbin)

    def create_zero_arrays(self):
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
        self.opac_wg_lay = np.zeros(self.nlayer_wg_nbin, self.fl_prec)
        self.opac_wg_int = np.zeros(self.ninterface_wg_nbin, self.fl_prec)
        self.scat_cross_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.scat_cross_int = np.zeros(self.ninterface_nbin, self.fl_prec)
        self.F_net = np.zeros(self.ninterface, self.fl_prec)
        self.F_net_diff = np.zeros(self.nlayer, self.fl_prec)
        self.meanmolmass_lay = np.zeros(self.nlayer, self.fl_prec)
        self.meanmolmass_int = np.zeros(self.ninterface, self.fl_prec)
        self.planckband_lay = np.zeros(self.nlayer_plus2_nbin, self.fl_prec)
        self.planckband_int = np.zeros(self.ninterface_nbin, self.fl_prec)
        self.planck_opac_T_pl = np.zeros(self.nlayer, self.fl_prec)
        self.ross_opac_T_pl = np.zeros(self.nlayer, self.fl_prec)
        self.planck_opac_T_star = np.zeros(self.nlayer, self.fl_prec)
        self.ross_opac_T_star = np.zeros(self.nlayer, self.fl_prec)
        self.trans_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.delta_tau_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.abort = np.zeros(self.nlayer + 1, np.int32)  # "nlayer + 1", because we include surface "layer"
        self.entropy_lay = np.zeros(self.nlayer, self.fl_prec)
        self.phase_number_lay = np.zeros(self.nlayer, self.fl_prec)
        self.trans_weight_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.contr_func_band = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.g_0_tot_lay = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.g_0_tot_int = np.zeros(self.ninterface_nbin, self.fl_prec)
        self.delta_z_lay = np.zeros(self.nlayer, self.fl_prec)
        self.z_lay = np.zeros(self.nlayer, self.fl_prec)
        self.T_int = np.zeros(self.ninterface, self.fl_prec)
        self.scat_trigger = np.zeros(self.wg_nbin, np.int32)
        self.delta_tau_all_clouds = np.zeros(self.nlayer_nbin, self.fl_prec)
        self.F_add_heat_lay = np.zeros(self.nlayer, self.fl_prec)
        self.F_add_heat_sum = np.zeros(self.nlayer, self.fl_prec)
        self.F_smooth = np.zeros(self.nlayer, self.fl_prec)
        self.F_smooth_sum = np.zeros(self.nlayer, self.fl_prec)

        # arrays to be used purely on the CPU
        self.conv_layer = np.zeros(self.nlayer + 1, np.int32)

    def copy_host_to_device(self):
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
        self.dev_opac_k = gpuarray.to_gpu(self.opac_k)
        self.dev_gauss_y = gpuarray.to_gpu(self.gauss_y)
        self.dev_gauss_weight = gpuarray.to_gpu(self.gauss_weight)
        self.dev_opac_wave = gpuarray.to_gpu(self.opac_wave)
        self.dev_opac_deltawave = gpuarray.to_gpu(self.opac_deltawave)
        self.dev_opac_interwave = gpuarray.to_gpu(self.opac_interwave)
        self.dev_opac_scat_cross = gpuarray.to_gpu(self.opac_scat_cross)
        self.dev_opac_meanmass = gpuarray.to_gpu(self.opac_meanmass)
        self.dev_entr_kappa = gpuarray.to_gpu(self.entr_kappa)
        self.dev_entr_c_p = gpuarray.to_gpu(self.entr_c_p)
        self.dev_entr_phase_number = gpuarray.to_gpu(self.entr_phase_number)
        self.dev_entr_entropy = gpuarray.to_gpu(self.entr_entropy)
        self.dev_c_p_lay = gpuarray.to_gpu(self.c_p_lay)
        self.dev_kappa_lay = gpuarray.to_gpu(self.kappa_lay)
        self.dev_starflux = gpuarray.to_gpu(self.starflux)
        self.dev_T_lay = gpuarray.to_gpu(self.T_lay)
        self.dev_surf_albedo = gpuarray.to_gpu(self.surf_albedo)
        self.dev_abs_cross_all_clouds_lay = gpuarray.to_gpu(self.abs_cross_all_clouds_lay)
        self.dev_scat_cross_all_clouds_lay = gpuarray.to_gpu(self.scat_cross_all_clouds_lay)
        self.dev_g_0_all_clouds_lay = gpuarray.to_gpu(self.g_0_all_clouds_lay)

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
        self.dev_opac_wg_lay = gpuarray.to_gpu(self.opac_wg_lay)
        self.dev_opac_wg_int = gpuarray.to_gpu(self.opac_wg_int)
        self.dev_scat_cross_lay = gpuarray.to_gpu(self.scat_cross_lay)
        self.dev_scat_cross_int = gpuarray.to_gpu(self.scat_cross_int)
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
        self.dev_meanmolmass_int = gpuarray.to_gpu(self.meanmolmass_int)
        self.dev_entropy_lay = gpuarray.to_gpu(self.entropy_lay)
        self.dev_phase_number_lay = gpuarray.to_gpu(self.phase_number_lay)
        self.dev_trans_weight_band = gpuarray.to_gpu(self.trans_weight_band)
        self.dev_contr_func_band = gpuarray.to_gpu(self.contr_func_band)
        self.dev_delta_z_lay = gpuarray.to_gpu(self.delta_z_lay)
        self.dev_z_lay = gpuarray.to_gpu(self.z_lay)
        self.dev_g_0_tot_lay = gpuarray.to_gpu(self.g_0_tot_lay)
        self.dev_T_int = gpuarray.to_gpu(self.T_int)
        self.dev_scat_trigger = gpuarray.to_gpu(self.scat_trigger)
        self.dev_delta_tau_all_clouds = gpuarray.to_gpu(self.delta_tau_all_clouds)
        self.dev_F_add_heat_lay = gpuarray.to_gpu(self.F_add_heat_lay)
        self.dev_F_add_heat_sum = gpuarray.to_gpu(self.F_add_heat_sum)
        self.dev_F_smooth = gpuarray.to_gpu(self.F_smooth)
        self.dev_F_smooth_sum = gpuarray.to_gpu(self.F_smooth_sum)

        if self.iso == 0:
            self.dev_planckband_int = gpuarray.to_gpu(self.planckband_int)
            self.dev_Fc_down_wg = gpuarray.to_gpu(self.Fc_down_wg)
            self.dev_Fc_up_wg = gpuarray.to_gpu(self.Fc_up_wg)
            self.dev_Fc_dir_wg = gpuarray.to_gpu(self.Fc_dir_wg)
            self.dev_g_0_tot_int = gpuarray.to_gpu(self.g_0_tot_int)
            self.dev_abs_cross_all_clouds_int = gpuarray.to_gpu(self.abs_cross_all_clouds_int)
            self.dev_scat_cross_all_clouds_int = gpuarray.to_gpu(self.scat_cross_all_clouds_int)
            self.dev_g_0_all_clouds_int = gpuarray.to_gpu(self.g_0_all_clouds_int)
            self.dev_kappa_int = gpuarray.to_gpu(self.kappa_int)

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
        self.entropy_lay = self.dev_entropy_lay.get()
        self.phase_number_lay = self.dev_phase_number_lay.get()
        self.trans_weight_band = self.dev_trans_weight_band.get()
        self.contr_func_band = self.dev_contr_func_band.get()
        self.g_0_tot_lay = self.dev_g_0_tot_lay.get()
        self.delta_z_lay = self.dev_delta_z_lay.get()
        self.z_lay = self.dev_z_lay.get()
        self.delta_tau_all_clouds = self.dev_delta_tau_all_clouds.get()
        self.F_add_heat_sum = self.dev_F_add_heat_sum.get()
        self.F_smooth_sum = self.dev_F_smooth_sum.get()

        if self.iso == 0:
            self.planckband_int = self.dev_planckband_int.get()
            self.kappa_int = self.dev_kappa_int.get()

    def allocate_on_device(self):
        """ allocate memory for arrays existing only on the GPU """

        # mem_alloc wants the number of bytes, which is 8 bytes per value for double precision (e.g. np.float64)
        # and 4 bytes per value for single precision (e.g. np.int32, np.float32)
        # save the correct space depending on precision

        size_nlayer_plus1 = int((self.nlayer+1) * self.nr_bytes)
        size_nlayer_nbin = int(self.nlayer_nbin * self.nr_bytes)
        size_nlayer_wg_nbin = int(self.nlayer_wg_nbin * self.nr_bytes)
        size_2_nlayer_wg_nbin = int(2 * self.nlayer_wg_nbin * self.nr_bytes)
        size_nplanckgrid = int(self.nplanck_grid * self.nr_bytes)
        size_ninterface_wg_nbin = int(self.ninterface_wg_nbin * self.nr_bytes)
        if self.iso == 1:
            size_nmatrix_wg_nbin = int(2 * self.ninterface_wg_nbin * self.nr_bytes)
        elif self.iso == 0:
            size_nmatrix_wg_nbin = int((4 * self.ninterface_wg_nbin - 2) * self.nr_bytes)

        # these arrays will never be copied between host and device.
        # Hence the normal mem_alloc functionality
        self.dev_delta_t_prefactor = cuda.mem_alloc(size_nlayer_plus1)
        self.dev_T_store = cuda.mem_alloc(size_nlayer_plus1)
        self.dev_planckband_grid = cuda.mem_alloc(size_nplanckgrid)
        self.dev_opac_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_delta_tau_wg = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_trans_wg = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_w_0 = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_M_term = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_N_term = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_P_term = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_G_plus = cuda.mem_alloc(size_nlayer_wg_nbin)
        self.dev_G_minus = cuda.mem_alloc(size_nlayer_wg_nbin)

        if self.opacity_mixing == "on-the-fly":
            self.dev_opac_spec_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin)

        if self.iso == 0:
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
            self.dev_delta_tau_all_clouds_upper = cuda.mem_alloc(size_nlayer_nbin)
            self.dev_delta_tau_all_clouds_lower = cuda.mem_alloc(size_nlayer_nbin)

            if self.opacity_mixing == "on-the-fly":
                self.dev_opac_spec_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin)

        # for matrix method
        if self.flux_calc_method == "matrix":
            if self.iso == 1:
                self.dev_alpha = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_beta = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_source_term_down = cuda.mem_alloc(size_nlayer_wg_nbin)
                self.dev_source_term_up = cuda.mem_alloc(size_nlayer_wg_nbin)
            elif self.iso == 0:
                self.dev_alpha = cuda.mem_alloc(size_2_nlayer_wg_nbin)
                self.dev_beta = cuda.mem_alloc(size_2_nlayer_wg_nbin)
                self.dev_source_term_down = cuda.mem_alloc(size_2_nlayer_wg_nbin)
                self.dev_source_term_up = cuda.mem_alloc(size_2_nlayer_wg_nbin)
            self.dev_c_prime = cuda.mem_alloc(size_nmatrix_wg_nbin)
            self.dev_d_prime = cuda.mem_alloc(size_nmatrix_wg_nbin)


if __name__ == "__main__":
    print("This module is for storing and allocating all the necessary quantities used in HELIOS"
          "...hopefully without memory bugs. ")
