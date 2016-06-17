# ==============================================================================
# Module for storing all the necessary quantities like parameters,
# arrays, input & output data, etc.
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
import pycuda.gpuarray as gpuarray
import phys_const as pc
import planets_and_stars as ps


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
        self.tabu = None
        self.singlewalk = None
        self.restart = None
        self.varying_tstep = None
        self.tstep = None
        self.scat = None
        self.direct = None
        self.diffusivity = None
        self.epsilon = None
        self.f_factor = None
        self.T_intern = None
        self.ntemp = None
        self.npress = None
        self.g_0 = None
        self.mu = None
        self.meanmolmass = None
        self.planet = None
        self.g = None
        self.a = None
        self.R_star = None
        self.T_star = None
        self.model = None
        self.real_star = np.int32(0)
        self.name = None
        self.foreplay = None
        self.fake_opac = None
        self.realtime_plot = None
        self.c_p = None
        self.iter_value = None
        self.ny = None
        self.nbin = None
        self.nlayer_nbin = None
        self.nlayer_plus2_nbin = None
        self.ninterface_nbin = None
        self.nlayer_wg_nbin = None
        self.ninterface_wg_nbin = None
        self.nplanck_grid = None
        self.ncapital_grid = None

        # arrays exclusively used on the CPU
        # these remain normal lists
        self.T_restart = []

        # input arrays to be copied CPU --> GPU
        # these need to be converted from lists to np.arrays of correct data format
        # and then copied to GPU with "gpuarray"
        self.ktemp = []
        self.dev_ktemp = None
        self.kpress = []
        self.dev_kpress = None
        self.opac_k = []
        self.dev_opac_k = None
        self.opac_y = []
        self.dev_opac_y = None
        self.opac_weight = None
        self.dev_opac_weight = None
        self.opac_wave = []
        self.dev_opac_wave = None
        self.opac_deltawave = []
        self.dev_opac_deltawave = None
        self.opac_interwave = []
        self.dev_opac_interwave = None
        self.cross_scat = []
        self.dev_cross_scat = None
        self.starflux = []
        self.dev_starflux = None

        # arrays to be copied CPU --> GPU --> CPU
        # these are copied to GPU by "gpuarray" and copied back
        self.T_lay = []
        self.dev_T_lay = None
        self.abort = None
        self.dev_abort = None

        # arrays to be copied GPU --> CPU
        # for these, zero arrays of correct size are created and then copied to GPU with "gpuarray" and copied back
        self.delta_colmass = None
        self.dev_delta_colmass = None
        self.Fup_band = None
        self.dev_Fup_band = None
        self.Fup_wg_band = None
        self.dev_Fup_wg_band = None
        self.Fdown_band = None
        self.dev_Fdown_band = None
        self.Fdown_wg_band = None
        self.dev_Fdown_wg_band = None
        self.Fup_tot = None
        self.dev_Fup_tot = None
        self.Fdown_tot = None
        self.dev_Fdown_tot = None
        self.opac_lay = None
        self.dev_opac_lay = None
        self.F_net = None
        self.dev_F_net = None
        self.p_lay = None
        self.dev_p_lay = None
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
        self.transmission = None
        self.dev_transmission = None

        # arrays exclusively used on the GPU
        # these are defined directly on the GPU and stay there. No copying required.
        self.dev_T_int = None
        self.dev_delta_colupper = None
        self.dev_delta_collower = None
        self.dev_p_int = None
        self.dev_delta_t_lay = None
        self.dev_delta_t_prefactor = None
        self.dev_delta_T_store = None
        self.dev_planckband_grid = None
        self.dev_opac_int = None
        self.dev_opac_wg_lay = None
        self.dev_opac_wg_int = None
        self.dev_Fc_down_wg_band = None
        self.dev_Fc_up_wg_band = None
        self.dev_Mterm_grid = None
        self.dev_Nterm_grid = None
        self.dev_Pterm_grid = None
        self.dev_Qterm_grid = None
        self.dev_Mterm = None
        self.dev_Nterm = None
        self.dev_Pterm = None
        self.dev_Qterm = None
        self.dev_Mterm_uppergrid = None
        self.dev_Nterm_uppergrid = None
        self.dev_Pterm_uppergrid = None
        self.dev_Qterm_uppergrid = None
        self.dev_Mterm_lowergrid = None
        self.dev_Nterm_lowergrid = None
        self.dev_Pterm_lowergrid = None
        self.dev_Qterm_lowergrid = None
        self.dev_Mterm_upper = None
        self.dev_Nterm_upper = None
        self.dev_Pterm_upper = None
        self.dev_Qterm_upper = None
        self.dev_Mterm_lower = None
        self.dev_Nterm_lower = None
        self.dev_Pterm_lower = None
        self.dev_Qterm_lower = None

    def convert_input_list_to_array(self):
        """ converts lists of quantities to arrays """

        self.ktemp = np.array(self.ktemp, np.float64)
        self.kpress = np.array(self.kpress, np.float64)
        self.opac_k = np.array(self.opac_k, np.float64)
        self.opac_y = np.array(self.opac_y, np.float64)
        self.opac_weight = np.array(self.opac_weight, np.float64)
        self.opac_wave = np.array(self.opac_wave, np.float64)
        self.opac_deltawave = np.array(self.opac_deltawave, np.float64)
        self.opac_interwave = np.array(self.opac_interwave, np.float64)
        self.cross_scat = np.array(self.cross_scat, np.float64)
        self.starflux = np.array(self.starflux, np.float64)
        self.T_lay = np.array(self.T_lay, np.float64)

    def dimensions(self):
        """ create the correct dimensions of the grid from input parameters """

        self.nbin = np.int32(len(self.opac_wave))
        self.ny = np.int32(len(self.opac_y))
        self.nlayer_nbin = np.int32(self.nlayer * self.nbin)
        self.nlayer_plus2_nbin = np.int32((self.nlayer+2) * self.nbin)
        self.ninterface_nbin = np.int32(self.ninterface * self.nbin)
        self.ninterface_wg_nbin = np.int32(self.ninterface * self.ny * self.nbin)
        self.nlayer_wg_nbin = np.int32(self.ninterface * self.ny * self.nbin)
        self.nplanck_grid = np.int32(402 * self.nbin)  # 402: grid in (10, 4000, 10) K + T_star, T_intern
        self.ncapital_grid = np.int32(190 * self.nbin * self.nlayer)  # 190: grid in opacity (1e-15, 1e4, log k = 0.1)

    def create_zero_arrays(self):
        """ creates zero arrays of quantities to be used on the GPU with the correct length/dimension """

        self.delta_colmass = np.zeros(self.nbin, np.float64)
        self.Fup_band = np.zeros(self.ninterface_nbin, np.float64)
        self.Fup_wg_band = np.zeros(self.ninterface_wg_nbin, np.float64)
        self.Fdown_band = np.zeros(self.ninterface_nbin, np.float64)
        self.Fdown_wg_band = np.zeros(self.ninterface_wg_nbin, np.float64)
        self.Fup_tot = np.zeros(self.ninterface, np.float64)
        self.Fdown_tot = np.zeros(self.ninterface, np.float64)
        self.opac_lay = np.zeros(self.nlayer_nbin, np.float64)
        self.F_net = np.zeros(self.nlayer, np.float64)
        self.p_lay = np.zeros(self.nlayer, np.float64)
        self.planckband_lay = np.zeros(self.nlayer_plus2_nbin, np.float64)
        self.planckband_int = np.zeros(self.ninterface_nbin, np.float64)
        self.planck_opac_T_pl = np.zeros(self.nlayer, np.float64)
        self.ross_opac_T_pl = np.zeros(self.nlayer, np.float64)
        self.planck_opac_T_star = np.zeros(self.nlayer, np.float64)
        self.ross_opac_T_star = np.zeros(self.nlayer, np.float64)
        self.transmission = np.zeros(self.nlayer_nbin, np.float64)
        self.abort = np.zeros(self.nlayer, np.int32)

    def copy_host_to_device(self):
        """ copies relevant host arrays to device """

        # input arrays
        self.dev_ktemp = gpuarray.to_gpu(self.ktemp)
        self.dev_kpress = gpuarray.to_gpu(self.kpress)
        self.dev_opac_k = gpuarray.to_gpu(self.opac_k)
        self.dev_opac_y = gpuarray.to_gpu(self.opac_y)
        self.dev_opac_weight = gpuarray.to_gpu(self.opac_weight)
        self.dev_opac_wave = gpuarray.to_gpu(self.opac_wave)
        self.dev_opac_deltawave = gpuarray.to_gpu(self.opac_deltawave)
        self.dev_opac_interwave = gpuarray.to_gpu(self.opac_interwave)
        self.dev_cross_scat = gpuarray.to_gpu(self.cross_scat)
        self.dev_starflux = gpuarray.to_gpu(self.starflux)
        self.dev_T_lay = gpuarray.to_gpu(self.T_lay)

        # zero arrays (copying anyway to obtain the gpuarray functionality)
        self.dev_delta_colmass = gpuarray.to_gpu(self.delta_colmass)
        self.dev_Fup_band = gpuarray.to_gpu(self.Fup_band)
        self.dev_Fup_wg_band = gpuarray.to_gpu(self.Fup_wg_band)
        self.dev_Fdown_band = gpuarray.to_gpu(self.Fdown_band)
        self.dev_Fdown_wg_band = gpuarray.to_gpu(self.Fdown_wg_band)
        self.dev_Fup_tot = gpuarray.to_gpu(self.Fup_tot)
        self.dev_Fdown_tot = gpuarray.to_gpu(self.Fdown_tot)
        self.dev_opac_lay = gpuarray.to_gpu(self.opac_lay)
        self.dev_F_net = gpuarray.to_gpu(self.F_net)
        self.dev_p_lay = gpuarray.to_gpu(self.p_lay)
        self.dev_planckband_lay = gpuarray.to_gpu(self.planckband_lay)
        self.dev_planckband_int = gpuarray.to_gpu(self.planckband_int)
        self.dev_planck_opac_T_pl = gpuarray.to_gpu(self.planck_opac_T_pl)
        self.dev_ross_opac_T_pl = gpuarray.to_gpu(self.ross_opac_T_pl)
        self.dev_planck_opac_T_star = gpuarray.to_gpu(self.planck_opac_T_star)
        self.dev_ross_opac_T_star = gpuarray.to_gpu(self.ross_opac_T_star)
        self.dev_transmission = gpuarray.to_gpu(self.transmission)
        self.dev_abort = gpuarray.to_gpu(self.abort)

    def copy_device_to_host(self):
        """ copies relevant device arrays to host """

        self.delta_colmass = self.dev_delta_colmass.get()
        self.Fup_band = self.dev_Fup_band.get()
        self.Fup_wg_band = self.dev_Fup_wg_band.get()
        self.Fdown_band = self.dev_Fdown_band.get()
        self.Fdown_wg_band = self.dev_Fdown_wg_band.get()
        self.Fup_tot = self.dev_Fup_tot.get()
        self.Fdown_tot = self.dev_Fdown_tot.get()
        self.opac_lay = self.dev_opac_lay.get()
        self.F_net = self.dev_F_net.get()
        self.p_lay = self.dev_p_lay.get()
        self.planckband_lay = self.dev_planckband_lay.get()
        self.planckband_int = self.dev_planckband_int.get()
        self.planck_opac_T_pl = self.dev_planck_opac_T_pl.get()
        self.ross_opac_T_pl = self.dev_ross_opac_T_pl.get()
        self.planck_opac_T_star = self.dev_planck_opac_T_star.get()
        self.ross_opac_T_star = self.dev_ross_opac_T_star.get()
        self.transmission = self.dev_transmission.get()

    def allocate_on_device(self):
        """ allocate memory for arrays existing only on the GPU """

        # mem_alloc wants the number of bytes, which is 8 bytes per value for double precision (e.g. np.float64)
        # and 4 bytes per value for single precision (e.g. np.int32)

        size_ninterface_double = int(self.ninterface * 8)
        size_nlayer_double = int(self.nlayer * 8)
        size_nlayer_single = int(self.nlayer * 4)  # array of integers
        size_nTstore_nlayer_double = int(6 * self.nlayer * 8)  # saving last six delta_T values
        size_ninterface_nbin_double = int(self.ninterface_nbin * 8)
        size_nlayer_wg_nbin_double = int(self.nlayer_wg_nbin * 8)
        size_ninterface_wg_nbin_double = int(self.ninterface_wg_nbin * 8)
        size_nplanckgrid_double = int(self.nplanck_grid * 8)
        size_ncapitalgrid_double = int(self.ncapital_grid * 8)
        
        self.dev_T_int = cuda.mem_alloc(size_ninterface_double)
        self.dev_delta_colupper = cuda.mem_alloc(size_nlayer_double)
        self.dev_delta_collower = cuda.mem_alloc(size_nlayer_double)
        self.dev_p_int = cuda.mem_alloc(size_ninterface_double)
        self.dev_delta_t_lay = cuda.mem_alloc(size_nlayer_double)
        self.dev_delta_t_prefactor = cuda.mem_alloc(size_nlayer_double)
        self.dev_delta_T_store = cuda.mem_alloc(size_nTstore_nlayer_double)
        self.dev_opac_int = cuda.mem_alloc(size_ninterface_nbin_double)
        self.dev_planckband_grid = cuda.mem_alloc(size_nplanckgrid_double)
        self.dev_opac_wg_lay = cuda.mem_alloc(size_nlayer_wg_nbin_double)
        self.dev_opac_wg_int = cuda.mem_alloc(size_ninterface_wg_nbin_double)
        self.dev_Fc_down_wg_band = cuda.mem_alloc(size_ninterface_wg_nbin_double)
        self.dev_Fc_up_wg_band = cuda.mem_alloc(size_ninterface_wg_nbin_double)

        if self.tabu == 1:
            self.dev_Mterm_grid = cuda.mem_alloc(size_ncapitalgrid_double)
            self.dev_Nterm_grid = cuda.mem_alloc(size_ncapitalgrid_double)
            self.dev_Pterm_grid = cuda.mem_alloc(size_ncapitalgrid_double)
            self.dev_Qterm_grid = cuda.mem_alloc(size_ncapitalgrid_double)
            self.dev_Mterm = cuda.mem_alloc(size_nlayer_wg_nbin_double)
            self.dev_Nterm = cuda.mem_alloc(size_nlayer_wg_nbin_double)
            self.dev_Pterm = cuda.mem_alloc(size_nlayer_wg_nbin_double)
            self.dev_Qterm = cuda.mem_alloc(size_nlayer_wg_nbin_double)

            if self.iso == 0:
                self.dev_Mterm_uppergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Nterm_uppergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Pterm_uppergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Qterm_uppergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Mterm_lowergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Nterm_lowergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Pterm_lowergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Qterm_lowergrid = cuda.mem_alloc(size_ncapitalgrid_double)
                self.dev_Mterm_upper = cuda.mem_alloc(size_nlayer_wg_nbin_double)
                self.dev_Nterm_upper = cuda.mem_alloc(size_nlayer_wg_nbin_double)
                self.dev_Pterm_upper = cuda.mem_alloc(size_nlayer_wg_nbin_double)
                self.dev_Qterm_upper = cuda.mem_alloc(size_nlayer_wg_nbin_double)
                self.dev_Mterm_lower = cuda.mem_alloc(size_nlayer_wg_nbin_double)
                self.dev_Nterm_lower = cuda.mem_alloc(size_nlayer_wg_nbin_double)
                self.dev_Pterm_lower = cuda.mem_alloc(size_nlayer_wg_nbin_double)
                self.dev_Qterm_lower = cuda.mem_alloc(size_nlayer_wg_nbin_double)

if __name__ == "__main__":
    print("This module is for storing all the necessary quantities, used in HELIOS.")
