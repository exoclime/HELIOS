import sys
sys.path.append("..")
import functions as fc

# Units are [temp]=K, [log_g]=log(cm s^-2), m = log[n(Fe)/n(H)]_star - log[n(Fe)/n(H)]_sun, with n(A) being number density of A

gj1214 = {
    "data_format": "phoenix",
    "name": "gj1214",
    "temp": 3026,
    "log_g": 4.944,
    "m": 0.39
} # reference: Harpsoe et al. (2013)

sun = {
    "data_format": "ascii",
    "source_file": "./input/ascii/sun_gueymard_2003.txt",
    "name": "sun",
    "w_conversion_factor": 1e-7,
    "flux_conversion_factor": 1e10,
    "temp": 5772
}

star_of_interest = {
    "data_format": "muscles",
    "name": "star_of_interest",
    "source_file" : "./input/downloaded_muscles_spectrum.fits",
    "w_conversion_factor": 1e-8,
    "flux_conversion_factor": 1e8,
    "distance_from_Earth": 4.67517, # in pc
    "R_star": 0.366999654557, # in R_sun
    "temp": 3293.7
}

# run the thing

fc.main_loop(gj1214,
             convert_to='r50_kdistr',
             opac_file_for_lambdagrid="../input/opacity/r50_kdistr/H2O_opac_kdistr.h5",
             output_file="star_2022.h5",
             plot_and_tweak='automatic',
             save_ascii='no',
             save_in_hdf5='yes')