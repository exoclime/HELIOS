# ==============================================================================
# This is the main file of HELIOS.
# Copyright (C) 2018 - 2022 Matej Malik
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


from helios import read
from helios import quantities as quant
from helios import host_functions as hsfunc
from helios import write
from helios import computation as comp
from helios import realtime_plotting as rt_plot
from helios import clouds
from helios import additional_heating as add_heat

# Depending on your CUDA installation, passing custom compiler
# flags might be necessary; the following was necessary to make
# it work on my laptop, but this might be totally different on
# your system!
NVCC_KWS: dict = {"arch":"sm_86"}

def run_helios():
    """ a full HELIOS run """

    reader = read.Read()
    keeper = quant.Store()
    computer = comp.Compute(nvcc_kws=NVCC_KWS)
    writer = write.Write()
    plotter = rt_plot.Plot()
    fogger = clouds.Cloud()

    # read input files and do preliminary calculations, like setting up the grid, etc.
    reader.read_param_file_and_command_line("param.dat", keeper, fogger)

    if keeper.opacity_mixing == "premixed":
        reader.load_premixed_opacity_table(keeper)

    elif keeper.opacity_mixing == "on-the-fly":
        reader.read_species_file(keeper)
        reader.read_species_opacities(keeper)
        reader.read_species_scat_cross_sections(keeper)
        reader.read_species_mixing_ratios(keeper)


    reader.read_kappa_table_or_use_constant_kappa(keeper)
    reader.read_or_fill_surf_albedo_array(keeper)
    keeper.dimensions()
    reader.read_star(keeper)
    hsfunc.planet_param(keeper, reader)
    hsfunc.set_up_numerical_parameters(keeper)
    hsfunc.construct_grid(keeper)
    hsfunc.initial_temp(keeper, reader)

    if keeper.approx_f == 1 and keeper.planet_type == "rocky":
        hsfunc.approx_f_from_formula(keeper, reader)

    hsfunc.calc_F_intern(keeper)
    add_heat.load_heating_terms_or_not(keeper)

    fogger.cloud_pre_processing(keeper)

    # create, convert and copy arrays to be used in the GPU computations
    keeper.create_zero_arrays()
    keeper.convert_input_list_to_array()
    keeper.copy_host_to_device()
    keeper.allocate_on_device()

    # conduct core computations on the GPU
    computer.construct_planck_table(keeper)
    computer.correct_incident_energy(keeper)

    computer.radiation_loop(keeper, writer, reader, plotter)

    computer.convection_loop(keeper, writer, reader, plotter)

    computer.integrate_optdepth_transmission(keeper)
    computer.calculate_contribution_function(keeper)
    if keeper.convection == 1:
        computer.interpolate_entropy(keeper)
        computer.interpolate_phase_state(keeper)
    computer.calculate_mean_opacities(keeper)
    computer.integrate_beamflux(keeper)

    # copy everything from the GPU back to host and write output quantities to files
    keeper.copy_device_to_host()
    hsfunc.calculate_conv_flux(keeper)
    hsfunc.calc_F_ratio(keeper)
    writer.create_output_dir_and_copy_param_file(reader, keeper)
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
    writer.write_cloud_mixing_ratio(keeper, reader)
    writer.write_cloud_opacities(keeper, reader)
    writer.write_Rayleigh_cross_sections(keeper, reader)
    writer.write_cloud_scat_cross_sections(keeper, reader)
    writer.write_g_0(keeper, reader)
    writer.write_transmission(keeper, reader)
    writer.write_opt_depth(keeper, reader)
    writer.write_cloud_opt_depth(keeper, reader)
    writer.write_trans_weight_function(keeper, reader)
    writer.write_contribution_function(keeper, reader)
    writer.write_mean_extinction(keeper, reader)
    writer.write_flux_ratio_only(keeper, reader)
    writer.write_phase_state(keeper, reader)
    writer.write_surface_albedo(keeper, reader)
    writer.write_criterion_warning_file(keeper, reader)

    if keeper.coupling == 1:
        writer.write_tp_for_coupling(keeper, reader)
        hsfunc.calculate_coupling_convergence(keeper, reader)

    if keeper.approx_f == 1:
        hsfunc.calc_tau_lw_sw(keeper, reader)

    # prints the success message - yay!
    hsfunc.success_message(keeper)


def main():
    """ runs the HELIOS RT computation if this file is executed """

    if __name__ == "__main__":

        run_helios()

main()
