# ==============================================================================
# Module for implementing (maybe later even calculating) Aerosol extinction (module experimental!)
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


import sys
import h5py


class Cloud(object):
    """ class that reads in cloud parameters, which then can be used in the HELIOS code """

    def __init__(self):
        self.lamda_orig = []
        self.scat_cross_orig = []
        self.abs_cross_orig = []
        self.g_0_orig = []
        self.cloud_name = None

    @staticmethod
    def onetoleft(array, limit):
        """ small function to find the index of an array element whose value is closest to the left of a given limit """

        b = []
        for a in array:
            if a < limit:
                b.append(a)
        sloth = max(b)
        left = b.index(sloth)
        return left

    def read_cloud_dat(self, quant):
        """ reads the cloud file (courtesy of D. Kitzmann) """

        try:
            with open(quant.cloud_path, "r") as cloud_file:
                next(cloud_file)
                for line in cloud_file:
                    column = line.split()
                    self.lamda_orig.append(quant.fl_prec(column[0]) * 1e-4)  # conversion um -> cm
                    self.scat_cross_orig.append(quant.fl_prec(column[2]))
                    self.abs_cross_orig.append(quant.fl_prec(column[3]))
                    self.g_0_orig.append(quant.fl_prec(column[5]))

        except IOError:
            print("ABORT - cloud file not found!")
            raise SystemExit()


    # converts fine spectra to coarser grid
    def convert_to_new_grid(self, int_lambda, cent_lambda, orig_lambda, orig_value):
        """ generic function to convert values from one grid to another """

        print("Conversion started...")

        percentage = 0

        int_value = [0] * len(int_lambda)
        cent_value = []

        for i in range(len(int_lambda)):

            percentage_before = percentage
            percentage = int(i / len(int_lambda) * 100.0)
            if percentage != percentage_before:
                sys.stdout.write("pre-converting: %d%%   \r" % (percentage))
                sys.stdout.flush()

            if int_lambda[i] < orig_lambda[0]:
                continue
            if int_lambda[i] > orig_lambda[len(orig_lambda) - 1]:
                break
            else:
                p_bot = self.onetoleft(orig_lambda, int_lambda[i])
                interpol = orig_value[p_bot] * (orig_lambda[p_bot + 1] - int_lambda[i]) + orig_value[p_bot + 1] * (
                int_lambda[i] - orig_lambda[p_bot])
                interpol /= (orig_lambda[p_bot + 1] - orig_lambda[p_bot])
                int_value[i] = interpol

        print("Preconversion done...")

        percentage = 0

        for i in range(len(cent_lambda)):

            percentage_before = percentage
            percentage = int(i / len(cent_lambda) * 100.0)
            if percentage != percentage_before:
                sys.stdout.write("converting: %d%%   \r" % (percentage))
                sys.stdout.flush()

            if int_lambda[i + 1] < orig_lambda[0] or int_lambda[i] > orig_lambda[len(orig_lambda) - 1]:
                cent_value.append(0)
            else:
                if int_value[i] != 0:
                    p_bot = self.onetoleft(orig_lambda, int_lambda[i])
                    p_start = p_bot + 1
                    for p in range(p_start, len(orig_lambda)):
                        if p == p_start:
                            if orig_lambda[p_start] < int_lambda[i + 1]:
                                interpol = (int_value[i] + orig_value[p]) / 2.0 * (orig_lambda[p] - int_lambda[i])
                                if p_start == len(orig_lambda) - 1:
                                    interpol /= (orig_lambda[p] - int_lambda[i])
                                    break
                            else:
                                interpol = (int_value[i] + int_value[i + 1]) / 2.0
                                break
                        else:
                            if orig_lambda[p] < int_lambda[i + 1]:
                                interpol += (orig_value[p - 1] + orig_value[p]) / 2.0 * (orig_lambda[p] - orig_lambda[p - 1])
                                if p == len(orig_lambda) - 1:
                                    interpol /= (orig_lambda[p] - int_lambda[i])
                                    break
                            else:
                                interpol += (orig_value[p - 1] + int_value[i + 1]) / 2.0 * (int_lambda[i + 1] - orig_lambda[p - 1])
                                interpol /= (int_lambda[i + 1] - int_lambda[i])
                                break
                if int_value[i] == 0:
                    interpol = 0
                    for p in range(0, len(orig_lambda) - 1):
                        if orig_value[p + 1] < int_lambda[i + 1]:
                            interpol += (orig_value[p] + orig_value[p + 1]) / 2.0 * (orig_lambda[p] - orig_lambda[p - 1])
                        else:
                            interpol += (orig_value[p] + int_value[i + 1]) / 2.0 * (int_lambda[i + 1] - orig_lambda[p])
                            interpol /= (int_lambda[i + 1] - int_lambda[i])
                            break
                cent_value.append(interpol)

        print("Conversion done!")

        return cent_value

    def conversion_of_cloud_parameters(self, quant):
        """ converts the cloud parameters to the HELIOS wavelength grid """

        quant.scat_cross_cloud = self.convert_to_new_grid(quant.opac_interwave, quant.opac_wave, self.lamda_orig, self.scat_cross_orig)
        quant.abs_cross_cloud = self.convert_to_new_grid(quant.opac_interwave, quant.opac_wave, self.lamda_orig, self.abs_cross_orig)
        quant.g_0_cloud = self.convert_to_new_grid(quant.opac_interwave, quant.opac_wave, self.lamda_orig, self.g_0_orig)

    def write_cloud_h5(self, quant):
        """ creates a h5 container with the interpolated cloud parameters"""

        try:
            with h5py.File("." + self.cloud_name + "_" + str(len(quant.opac_wave)) + ".h5", "w") as h5_file:
                h5_file.create_dataset("scattering cross sections", data=quant.scat_cross_cloud)
                h5_file.create_dataset("absorption cross sections", data=quant.abs_cross_cloud)
                h5_file.create_dataset("asymmetry parameter", data=quant.g_0_cloud)
                h5_file.create_dataset("center wavelengths", data=quant.opac_wave)
                h5_file.create_dataset("interface wavelengths", data=quant.opac_interwave)
        except:
            print("ABORT - something wrong with writing the cloud.h5 file!")
            raise SystemExit()

    def read_cloud_h5(self, quant):
        """ reads int the cloud .h5 file """

        try:
            with h5py.File("." + self.cloud_name + ".h5", "r") as h5_file:
                print("\nReading cloud parameters...")
                for s in h5_file["scattering cross sections"][:]:
                    quant.scat_cross_cloud.append(s)
                for a in h5_file["absorption cross sections"][:]:
                    quant.abs_cross_cloud.append(a)
                for g in h5_file["asymmetry parameter"][:]:
                    quant.g_0_cloud.append(g)
        except:
            print("ABORT - something wrong with reading the cloud.h5 file!")
            raise SystemExit()

    def main_cloud_method(self, quant):
        """ main method to handle cloud input """

        if quant.clouds == 1:

            suffix = quant.cloud_path.split('.')[-1:][0]
            self.cloud_name = quant.cloud_path.split('.')[-2:-1][0]

            if suffix == "dat":

                self.read_cloud_dat(quant)
                self.conversion_of_cloud_parameters(quant)
                self.write_cloud_h5(quant)

            elif suffix == "h5":

                self.read_cloud_h5(quant)

            else:
                print("Unrecognized cloud file format. Aborting...")
                raise SystemExit()


if __name__ == "__main__":
    print("This module takes care of the ugly cloud business. "
          "It is dusty and quite foggy in here. Enter upon own risk.")
