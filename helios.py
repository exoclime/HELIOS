# ==============================================================================
# This is the main file of HELIOS.
# Copyright (C) 2018 Matej Malik
#
# To run HELIOS simply execute this file with Python 3.x
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


from source import read
from source import quantities as quant
from source import host_functions as hsfunc
from source import write
from source import computation as comp
from source import realtime_plotting as rt_plot
from source import clouds
from source import Vcoupling_modification as Vmod


def run_helios():
    """ runs a normal HELIOS run with standard I/O """

    reader = read.Read()
    keeper = quant.Store()
    computer = comp.Compute()
    writer = write.Write()
    plotter = rt_plot.Plot()
    cloudy = clouds.Cloud()
    Vmodder = Vmod.Vcoupling()

    # read input files and do preliminary calculations
    reader.read_param_file(keeper, Vmodder)
    reader.read_command_line(keeper, Vmodder)

    if Vmodder.V_coupling == 1:
        Vmodder.read_or_create_iter_count()
        Vmodder.read_species()
        Vmodder.read_molecular_opacities(keeper)
        Vmodder.read_layer_molecular_abundance(keeper)
    reader.read_opac_file(keeper, Vmodder)
    reader.read_entropy_table(keeper)
    cloudy.main_cloud_method(keeper)
    keeper.dimensions()
    reader.read_star(keeper)
    hsfunc.planet_param(keeper, reader)
    hsfunc.set_up_numerical_parameters(keeper)
    hsfunc.construct_grid(keeper)
    hsfunc.initial_temp(keeper, reader, Vmodder)
    if keeper.approx_f == 1:
        hsfunc.approx_f_from_formula(keeper, reader)
    hsfunc.calc_F_intern(keeper)


    # get ready for GPU computations
    keeper.create_zero_arrays(Vmodder)
    keeper.convert_input_list_to_array(Vmodder)
    keeper.copy_host_to_device(Vmodder)
    keeper.allocate_on_device(Vmodder)

    # conduct the GPU core computations
    computer.construct_planck_table(keeper)
    computer.correct_incident_energy(keeper)

    if Vmodder.V_coupling == 1:
        if Vmodder.V_iter_nr > 0:
            Vmodder.interpolate_f_molecule_and_meanmolmass(keeper)
            Vmodder.combine_to_scat_cross(keeper)

    computer.radiation_loop(keeper, writer, plotter, Vmodder)

    computer.convection_loop(keeper, writer, plotter, Vmodder)

    computer.integrate_optdepth_transmission(keeper)
    computer.calculate_contribution_function(keeper)
    computer.interpolate_entropy(keeper)
    computer.calculate_mean_opacities(keeper)
    computer.integrate_beamflux(keeper)

    # copy everything back to host and write to files
    keeper.copy_device_to_host()
    hsfunc.calculate_conv_flux(keeper)
    hsfunc.calc_F_ratio(keeper)
    writer.write_info(keeper, reader, Vmodder)
    writer.write_colmass_mu_cp_entropy(keeper, reader)
    writer.write_integrated_flux(keeper, reader)
    writer.write_downward_spectral_flux(keeper, reader)
    writer.write_upward_spectral_flux(keeper, reader)
    writer.write_TOA_flux_eclipse_depth(keeper, reader)
    writer.write_direct_spectral_beam_flux(keeper, reader)
    writer.write_planck_interface(keeper, reader)
    writer.write_planck_center(keeper, reader)
    writer.write_tp(keeper, reader)
    writer.write_tp_cut(keeper, reader)
    writer.write_opacities(keeper, reader)
    writer.write_Rayleigh_cross_sections(keeper, reader)
    writer.write_cloud_scat_cross_sections(keeper, reader)
    writer.write_cloud_absorption(keeper, reader)
    writer.write_g_0(keeper, reader)
    writer.write_transmission(keeper, reader)
    writer.write_opt_depth(keeper, reader)
    writer.write_trans_weight_function(keeper, reader)
    writer.write_contribution_function(keeper, reader)
    writer.write_mean_extinction(keeper, reader)
    writer.write_flux_ratio_only(keeper, reader)
    if Vmodder.V_coupling == 1:
        Vmodder.write_tp_VULCAN(keeper)
    if keeper.approx_f == 1:
        hsfunc.calc_tau_lw_sw(keeper, reader)

    # prints the success message - yay!
    hsfunc.success_message(keeper)

    if Vmodder.V_coupling == 1:
        Vmodder.test_coupling_convergence(keeper)


def main():
    """ runs the HELIOS RT computation if this file is executed """

    if __name__ == "__main__":

        run_helios()

main()
