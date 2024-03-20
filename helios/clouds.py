# ==============================================================================
# Module adding aerosol extinction to HELIOS
# Copyright (C) 2020 - 2022 Matej Malik
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
from scipy import interpolate as itp
from helios import tools as tls


class Cloud(object):
    """ class that reads in cloud parameters to be used in the HELIOS code """

    def __init__(self):
        self.nr_cloud_decks = None
        self.mie_path = None
        self.cloud_r_mode = None
        self.cloud_r_std_dev = None
        self.cloud_mixing_ratio_setting = None
        self.cloud_vmr_file = None
        self.cloud_vmr_file_header_lines = None
        self.cloud_file_press_name = None
        self.cloud_file_press_units = None
        self.cloud_file_species_name = None
        self.p_cloud_bot = None
        self.f_cloud_bot = None
        self.cloud_to_gas_scale_height = None
        self.lamda_mie = None
        self.abs_cross_one_cloud = None
        self.scat_cross_one_cloud = None
        self.g_0_one_cloud = None
        self.f_one_cloud_lay = None
        self.f_one_cloud_int = None

    @staticmethod
    def read_mie_file(mie_file):
        """ reads an LX-Mie output file """

        lamda_mie = []
        abs_cross_mie = []
        scat_cross_mie = []
        g_0_mie = []

        with open(mie_file, "r") as mfile:
            next(mfile)
            for line in mfile:
                column = line.split()
                lamda_mie.append(float(column[0]) * 1e-4)  # conversion micron -> cm
                scat_cross_mie.append(float(column[3]))
                abs_cross_mie.append(float(column[4]))
                g_0_mie.append(float(column[6]))

        return lamda_mie, scat_cross_mie, abs_cross_mie, g_0_mie

    @staticmethod
    def lognorm_pdf(r, r_mode, sigma):

        r_median = r_mode / np.exp(-np.log(sigma) ** 2)

        norm_factor = 1 / (r * np.log(sigma) * (2 * np.pi) ** 0.5)
        pdf = norm_factor * np.exp(-0.5 * (np.log(r / r_median) / np.log(sigma)) ** 2)

        return pdf

    def calc_weighted_cross_sections_with_pdf_and_interpolate_wavelengths(self, nr, quant):

        # reset for each cloud
        weighted_abs_cross_mie = []
        weighted_scat_cross_mie = []
        weighted_g_0_mie = []

        r_values = 10 ** np.arange(-2, 3.1, 0.1)  # WARNING: hardcoded particle sizes, WARNING: micron units

        delta_r = r_values * (10**0.05 - 10**-0.05)  # WARNING: micron units and hardcoded particle stepsize

        # calc pdf for these r values
        pdf = self.lognorm_pdf(r_values, self.cloud_r_mode[nr], self.cloud_r_std_dev[nr])

        # get lamda values
        self.lamda_mie, _, _, _ = self.read_mie_file(self.mie_path[nr] + "r{:.6f}.dat".format(r_values[0]))

        abs_cross_per_r = np.zeros((len(r_values), len(self.lamda_mie)))
        scat_cross_per_r = np.zeros((len(r_values), len(self.lamda_mie)))
        g_0_per_r = np.zeros((len(r_values), len(self.lamda_mie)))

        for r in range(len(r_values)):

            _, scat_cross_per_r[r, :], abs_cross_per_r[r, :], g_0_per_r[r, :] = self.read_mie_file(self.mie_path[nr] + "r{:.6f}.dat".format(r_values[r]))

        for l in range(len(self.lamda_mie)):

            abs_cross = sum(abs_cross_per_r[:, l] * pdf * delta_r)
            scat_cross = sum(scat_cross_per_r[:, l] * pdf * delta_r)
            g_0 = sum(scat_cross_per_r[:, l] * pdf * delta_r)

            weighted_abs_cross_mie.append(abs_cross)
            weighted_scat_cross_mie.append(scat_cross)
            weighted_g_0_mie.append(g_0)

        # interpolate to HELIOS wavelength grid
        self.abs_cross_one_cloud = tls.convert_spectrum(self.lamda_mie, weighted_abs_cross_mie, quant.opac_wave, int_lambda=quant.opac_interwave, type='log')
        self.scat_cross_one_cloud = tls.convert_spectrum(self.lamda_mie, weighted_scat_cross_mie, quant.opac_wave, int_lambda=quant.opac_interwave, type='log')
        self.g_0_one_cloud = tls.convert_spectrum(self.lamda_mie, weighted_g_0_mie, quant.opac_wave, int_lambda=quant.opac_interwave, type='linear')

    def create_cloud_deck(self, nr, quant):

        # reset for each cloud
        self.f_one_cloud_lay = np.zeros(quant.nlayer)
        self.f_one_cloud_int = np.zeros(quant.ninterface)

        # layer index of cloud bottom
        i_bot = 0

        if self.cloud_mixing_ratio_setting == "manual":

            for i in range(quant.nlayer):

                if quant.p_int[i] >= self.p_cloud_bot[nr] > quant.p_int[i + 1]:
                    self.f_one_cloud_lay[i] = self.f_cloud_bot[nr]

                    i_bot = i

                    break

            for i in range(i_bot + 1, quant.nlayer):
                self.f_one_cloud_lay[i] = self.f_cloud_bot[nr] * (quant.p_lay[i] / quant.p_lay[i_bot]) ** (1 / self.cloud_to_gas_scale_height[nr] - 1)

            if quant.iso == 0:

                for i in range(i_bot + 1, quant.ninterface):
                    self.f_one_cloud_int[i] = self.f_cloud_bot[nr] * (quant.p_int[i] / quant.p_lay[i_bot]) ** (1 / self.cloud_to_gas_scale_height[nr] - 1)

        elif self.cloud_mixing_ratio_setting == "file":

            cloud_file = np.genfromtxt(self.cloud_vmr_file, names=True, dtype=None, skip_header=self.cloud_vmr_file_header_lines)

            press_orig = cloud_file[self.cloud_file_press_name]

            if self.cloud_file_press_units == "Pa":

                press_orig *= 10

            elif self.cloud_file_press_units == "bar":

                press_orig *= 1e6

            f_cloud_orig = cloud_file[self.cloud_file_species_name[nr]]

            log_press_orig = [np.log10(p) for p in press_orig]
            log_p_lay = [np.log10(p) for p in quant.p_lay]

            cloud_interpol_function = itp.interp1d(log_press_orig, f_cloud_orig, kind='linear', bounds_error=False, fill_value=(f_cloud_orig[-1], f_cloud_orig[0]))

            self.f_one_cloud_lay = cloud_interpol_function(log_p_lay)

            if quant.iso == 0:

                log_p_int = [np.log10(p) for p in quant.p_int]

                self.f_one_cloud_int = cloud_interpol_function(log_p_int)

    def add_individual_cloud_decks_to_total(self, quant):

        # combine all cloud densities (this is really just for the cloud output file)
        for i in range(quant.nlayer):

            quant.f_all_clouds_lay[i] += self.f_one_cloud_lay[i]

            for x in range(quant.nbin):

                quant.abs_cross_all_clouds_lay[x + quant.nbin * i] += self.f_one_cloud_lay[i] * self.abs_cross_one_cloud[x]
                quant.scat_cross_all_clouds_lay[x + quant.nbin * i] += self.f_one_cloud_lay[i] * self.scat_cross_one_cloud[x]

                quant.g_0_all_clouds_lay[x + quant.nbin * i] += self.g_0_one_cloud[x] * self.f_one_cloud_lay[i] * self.scat_cross_one_cloud[x]

        if quant.iso == 0:

            for i in range(quant.ninterface):

                quant.f_all_clouds_int[i] += self.f_one_cloud_int[i]

                for x in range(quant.nbin):

                    quant.abs_cross_all_clouds_int[x + quant.nbin * i] += self.f_one_cloud_int[i] * self.abs_cross_one_cloud[x]
                    quant.scat_cross_all_clouds_int[x + quant.nbin * i] += self.f_one_cloud_int[i] * self.scat_cross_one_cloud[x]

                    quant.g_0_all_clouds_int[x + quant.nbin * i] += self.g_0_one_cloud[x] * self.f_one_cloud_int[i] * self.scat_cross_one_cloud[x]

    @staticmethod
    def normalize_g_0(quant):

        # have to normalize g_0 again to become dimensionless
        for i in range(quant.nlayer):

            for x in range(quant.nbin):

                if quant.scat_cross_all_clouds_lay[x + quant.nbin * i] > 0:

                    quant.g_0_all_clouds_lay[x + quant.nbin * i] /= quant.scat_cross_all_clouds_lay[x + quant.nbin * i]

        if quant.iso == 0:

            for i in range(quant.ninterface):

                for x in range(quant.nbin):

                    if quant.scat_cross_all_clouds_int[x + quant.nbin * i] > 0:

                        quant.g_0_all_clouds_int[x + quant.nbin * i] /= quant.scat_cross_all_clouds_int[x + quant.nbin * i]

    def cloud_pre_processing(self, quant):
        """ conducts the pre-processing of cloud data so it can be included in the RT calculation """

        quant.f_all_clouds_lay = np.zeros(quant.nlayer)
        quant.f_all_clouds_int = np.zeros(quant.ninterface)

        quant.abs_cross_all_clouds_lay = np.zeros(quant.nlayer * quant.nbin)
        quant.abs_cross_all_clouds_int = np.zeros(quant.ninterface * quant.nbin)

        quant.scat_cross_all_clouds_lay = np.zeros(quant.nlayer * quant.nbin)
        quant.scat_cross_all_clouds_int = np.zeros(quant.ninterface * quant.nbin)

        quant.g_0_all_clouds_lay = np.zeros(quant.nlayer * quant.nbin)
        quant.g_0_all_clouds_int = np.zeros(quant.ninterface * quant.nbin)

        if quant.clouds == 1:

            for nr in range(self.nr_cloud_decks):

                self.calc_weighted_cross_sections_with_pdf_and_interpolate_wavelengths(nr, quant)

                self.create_cloud_deck(nr, quant)

                self.add_individual_cloud_decks_to_total(quant)

            self.normalize_g_0(quant)


if __name__ == "__main__":
    print("This module takes care of the ugly cloud business. "
          "It is dusty and quite foggy in here. Enter upon own risk.")
