###### function definitions for the stellar spectrum tool ###########

from astropy.io import fits
import numpy as np
import h5py
import matplotlib.pyplot as plt
import wget
import os
from source import tools as tls
from source import phys_const as pc


def save_to_dat(name, lamda, flux):

    with open("./" + name + ".dat", "w") as file:

        file.writelines("{:<15}{:<25}".format("lambda [um]", "flux [erg s^-1 cm^-3]"))
        for i in range(len(lamda)):
            file.writelines("\n{:<15.7e}{:<25.7e}".format(lamda[i], flux[i]))


def read_ascii_file(star):
    """ reads in a stellar_tool spectrum from an ascii file """

    lamda = []
    flux = []

    with open(star["source_file"], 'r') as file:

        next(file)
        next(file)
        next(file)
        next(file)
        next(file)
        next(file)
        next(file)
        next(file)

        for line in file:

            column = line.split()

            lamda.append(float(column[0]))
            flux.append(float(column[1]))

    lamda = [l * star["w_conversion_factor"] for l in lamda]
    flux = [f * star["flux_conversion_factor"] * (pc.AU / pc.R_SUN)**2 for f in flux]

    return lamda, flux

def read_muscles_file(star):
    """ reads in a stellar_tool spectrum from a MUSCLES fits file """

    contents = fits.getdata(star["source_file"], 1)

    lamda = contents['WAVELENGTH']
    flux = contents['FLUX']

    dist = star["distance_from_Earth"] * pc.PC
    rstar = star["R_star"] * pc.R_SUN

    lamda = [l * star["w_conversion_factor"] for l in lamda]
    flux = [f * star["flux_conversion_factor"] * (dist/rstar)**2 for f in flux]

    return lamda, flux

def read_btsettl_file(star):
    """ reads in a stellar_tool spectrum from a BT-Settl fits file """

    contents = fits.getdata(star["source_file"], 0)

    lamda = contents[0]
    flux = contents[1]

    lamda = [l * star["w_conversion_factor"] for l in lamda]
    flux = [f * star["flux_conversion_factor"] for f in flux]



    return lamda, flux

def read_fits(file):

    with fits.open(file) as fits_file:

        content = fits_file[0]

        flux = [f for f in content.data[:]]

    return flux


def interpol_phoenix_spectrum(name, teff, log_g, metal):

    if teff < 7000:
        tdown = int(100 * np.floor(teff / 100))
        tup = int(100 * np.ceil(teff / 100))
    else:
        tdown = int(200 * np.floor(teff / 200))
        tup = int(200 * np.ceil(teff / 200))

    gdown = 0.5 * np.floor(log_g / 0.5)
    gup = 0.5 * np.ceil(log_g / 0.5)

    if metal >= -2.0 and metal <= 1.0:
        mdown = 0.5 * np.floor(metal / 0.5)
        mup = 0.5 * np.ceil(metal / 0.5)
    else:
        print("ERROR: Metallicity out of bounds! Aborting...")

    for t in [tup, tdown]:

        for g in [gup, gdown]:

            for m in [mup, mdown]:

                print("\nLooking for PHOENIX file: {:05d}_{:.2f}_{:.1f}.fits".format(t, g, m))

                # checks wether file already exists. otherwise downloads it.
                if not os.path.exists("./input/phoenix/" + name + "/{:05d}_{:.2f}_{:.1f}.fits".format(t, g, m)):
                    print("File not found!")
                    print("Downloading PHOENIX file from the Goettingen server.")

                    if m <= 0:
                        link = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{:.1f}/lte{:05d}-{:.2f}-{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(abs(m), t, g, abs(m))
                    else:
                        link = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z+{:.1f}/lte{:05d}-{:.2f}+{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(m, t, g, m)

                    wget.download(link, out="./input/phoenix/"+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(t, g, m))

                else:
                    print("Found!")

    print("\n\nInterpolating PHOENIX model of "+name+".\n")

    flux_tup_gup_mup = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tup, gup, mup))

    try:
        flux_tdown_gup_mup = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tdown, gup, mup))
    except:
        pass

    try:
        flux_tup_gdown_mup = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tup, gdown, mup))
    except:
        pass

    try:
        flux_tdown_gdown_mup = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tdown, gdown, mup))
    except:
        pass

    try:
        flux_tup_gup_mdown = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tup, gup, mdown))
    except:
        pass

    try:
        flux_tdown_gup_mdown = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tdown, gup, mdown))
    except:
        pass

    try:
        flux_tup_gdown_mdown = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tup, gdown, mdown))
    except:
        pass

    try:
        flux_tdown_gdown_mdown = read_fits('./input/phoenix/'+name+"/{:05d}_{:.2f}_{:.1f}.fits".format(tdown, gdown, mdown))
    except:
        pass
 
    interpol_flux = []

    for i in range(len(flux_tup_gup_mup)):
        if tup != tdown and gup != gdown and mup != mdown:
            interpol =        flux_tup_gup_mup[i] * (teff - tdown) * (log_g - gdown) * (metal - mdown) \
                            + flux_tdown_gup_mup[i] * (tup - teff) * (log_g - gdown) * (metal - mdown) \
                            + flux_tup_gdown_mup[i] * (teff - tdown) * (gup - log_g) * (metal - mdown) \
                            + flux_tdown_gdown_mup[i] * (tup - teff) * (gup - log_g) * (metal - mdown) \
                            + flux_tup_gup_mdown[i] * (teff - tdown) * (log_g - gdown) * (mup - metal) \
                            + flux_tdown_gup_mdown[i] * (tup - teff) * (log_g - gdown) * (mup - metal) \
                            + flux_tup_gdown_mdown[i] * (teff - tdown) * (gup - log_g) * (mup - metal) \
                            + flux_tdown_gdown_mdown[i] * (tup - teff) * (gup - log_g) * (mup - metal)
            interpol /= ((tup-tdown)*(gup-gdown)*(mup-mdown))

        elif tup == tdown and gup == gdown and mup == mdown:
            interpol = flux_tup_gup_mup[i]

        elif tup == tdown and gup == gdown:
            interpol =        flux_tup_gup_mup[i] * (metal - mdown) \
                            + flux_tup_gup_mdown[i] * (mup - metal)
            interpol /= (mup-mdown)

        elif tup == tdown and mup == mdown:
            interpol =        flux_tup_gup_mup[i] * (log_g - gdown) \
                            + flux_tup_gdown_mup[i] * (gup - log_g)
            interpol /= (gup-gdown)

        elif mup == mdown:
            interpol =        flux_tup_gup_mup[i] * (teff - tdown) * (log_g - gdown) \
                            + flux_tdown_gup_mup[i] * (tup - teff) * (log_g - gdown) \
                            + flux_tup_gdown_mup[i] * (teff - tdown) * (gup - log_g) \
                            + flux_tdown_gdown_mup[i] * (tup - teff) * (gup - log_g)
            interpol /= ((tup-tdown)*(gup-gdown))

        elif gup == gdown:
            interpol =        flux_tup_gup_mup[i] * (teff - tdown) * (metal - mdown) \
                            + flux_tdown_gup_mup[i] * (tup - teff) * (metal - mdown) \
                            + flux_tup_gup_mdown[i] * (teff - tdown) * (mup - metal) \
                            + flux_tdown_gup_mdown[i] * (tup - teff) * (mup - metal)
            interpol /= ((tup-tdown)*(mup-mdown))

        elif tup == tdown:
            interpol =        flux_tup_gup_mup[i] * (log_g - gdown) * (metal - mdown) \
                            + flux_tup_gdown_mup[i] * (gup - log_g) * (metal - mdown) \
                            + flux_tup_gup_mdown[i] * (log_g - gdown) * (mup - metal) \
                            + flux_tup_gdown_mdown[i] * (gup - log_g) * (mup - metal)
            interpol /= ((gup-gdown)*(mup-mdown))

        interpol_flux.append(interpol)
        
    return interpol_flux


def h5_rm_create(f,dname,array):
    """ removes and creates an h5 data array

    :param f:
    :param dname:
    :param array:
    :return:
    """
    if dname in f:
        f.__delitem__(dname)
    f.create_dataset(dname, data=array)


def read_in_condition(f, dname, lambda_array):
    """ condition to use or not use this data array

    :param f:
    :param dname:
    :param lambda_array:
    :return:
    """

    condition=0
    if not dname in f:
        condition=1
    if dname in f:
        if len(lambda_array) != len(f[dname]):
            condition=1
    return condition


def read_or_interpol_phoenix(f, star, pho_lambda, force_interpol="no"):
    """ reads in the phoenix spectrum from h5 file or interpolates from downloaded data """

    if read_in_condition(f, "/original/phoenix/" + star["name"], pho_lambda) or force_interpol == "yes":

        print("\ncurrently working on: /original/phoenix/" + star["name"])
        print("\nstellar parameters: T_eff: {:05d}, log(g): {:.2f}, metallicity: {:.1f}".format(int(star["temp"]), star["log_g"], star["m"]))

        pho_interpolated = interpol_phoenix_spectrum(star["name"], star["temp"], star["log_g"], star["m"])
        h5_rm_create(f, "/original/phoenix/" + star["name"], pho_interpolated)
    else:
        array = [item for item in f["/original/phoenix/" + star["name"]][:]]
        pho_interpolated = array

    return pho_interpolated


def gen_int_lambda_values(lamda):
    """ generates interface wavelength values from wavelength bin values

    :param lamda:

    :return:
    """

    int_lambda = []

    int_lambda.append(lamda[0] - (lamda[1] - lamda[0]) / 2)
    for x in range(len(lamda) - 1):
        int_lambda.append((lamda[x + 1] + lamda[x]) / 2)
    int_lambda.append(lamda[-1] + (lamda[-1] - lamda[-2]) / 2)

    return int_lambda


def main_loop(star, convert_to, opac_file_for_lambdagrid, output_file, plot_and_tweak='no', save_ascii='no', save_in_hdf5='no', BB_temp=None):

    with h5py.File(opac_file_for_lambdagrid, "r") as file:

        try:
            new_lambda = [l for l in file["centre wavelengths"][:]]
            int_lambda = [l for l in file["interface wavelengths"][:]]

        except KeyError:
            try:
                new_lambda = [l for l in file["center wavelengths"][:]]
                int_lambda = [l for l in file["interface wavelengths"][:]]

            except KeyError:
                try:
                    new_lambda = [l for l in file["wavelengths"][:]]
                    int_lambda = None
                except:
                    raise IOError("ERROR: Unable to read wavelength data set!")

    if not os.path.exists("output"):
        os.makedirs("output")

    with h5py.File("./output/"+output_file, "a") as f:

        if star["data_format"] == "phoenix":

            if not os.path.exists("input/phoenix/"+star["name"]):
                os.makedirs("input/phoenix/"+star["name"])

            if not os.path.exists("input/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"):

                print("\nWavelength file does not exist yet. Downloading...")

                link = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS//WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

                wget.download(link, out="./input/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

            orig_lambda = read_fits('./input/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
            orig_lambda = [o * 1e-8 for o in orig_lambda]  # convert from Angstrom to cm

            orig_flux = read_or_interpol_phoenix(f, star, orig_lambda)

        elif star["data_format"] == "ascii":

            orig_lambda, orig_flux = read_ascii_file(star)

        elif star["data_format"] == "muscles":

            orig_lambda, orig_flux = read_muscles_file(star)

        elif star["data_format"] == "btsettl":

            orig_lambda, orig_flux = read_btsettl_file(star)

        else:
            raise IOError

        reply = None

        if BB_temp is None:
            BB_temp = star["temp"]

        if plot_and_tweak == 'yes' or plot_and_tweak == 'automatic':
            nr_iterations = 2
        else:
            nr_iterations = 1

        for iteration in range(nr_iterations):
            print("BB temp for final calculation is: {:.3f} K.".format(BB_temp))
            converted_flux = tls.convert_spectrum(old_lambda=orig_lambda,
                                                  old_flux=orig_flux,
                                                  new_lambda=new_lambda,
                                                  int_lambda=int_lambda,
                                                  extrapolate_with_BB_T=BB_temp
                                                  )

            if plot_and_tweak == 'yes' or plot_and_tweak == 'automatic':

                while reply != "yes":

                    # trying out a BB for extrapol
                    tryout_BB = []

                    if int_lambda == None:
                        int_lambda = gen_int_lambda_values(new_lambda)

                    print("\nCalculating BB for extrapolation.")

                    if plot_and_tweak == 'automatic':

                        print("Finding best BB extrapolation using Newton-Raphson with 10 iterations.")
                        extrapolation_necessary = False

                        # get last cell within old lambda range
                        for i in range(len(int_lambda)):

                            if int_lambda[i] > orig_lambda[-1]:

                                index = i - 2 # index of bin to be used for BB fitting. it is the last bin fully covered by original spectrum
                                extrapolation_necessary = True
                                break

                        if extrapolation_necessary:

                            # do 10 convergence iterations
                            for n in range(10):
                                if n == 0:
                                    BB_temp_before = BB_temp - 100
                                    BB_temp_now = BB_temp

                                else:
                                    BB_temp_before = BB_temp_now
                                    BB_temp_now = BB_temp_new

                                BB_value_before = np.pi * tls.calc_analyt_planck_in_interval(BB_temp_before, int_lambda[index], int_lambda[index + 1])
                                BB_value_now = np.pi * tls.calc_analyt_planck_in_interval(BB_temp_now, int_lambda[index], int_lambda[index + 1])

                                if BB_value_before != BB_value_now:
                                    BB_temp_new = BB_temp_now - (BB_value_now - converted_flux[index]) / (BB_value_now - BB_value_before) * (BB_temp_now - BB_temp_before)

                                else:
                                    BB_temp_new = BB_temp_now

                                print("Finding BB temp for extrapol... {:.3f}".format(BB_temp_new))

                            BB_temp = BB_temp_new

                    for i in range(len(new_lambda)):

                        tls.percent_counter(i, len(new_lambda))

                        tryout_BB.append(np.pi * tls.calc_analyt_planck_in_interval(BB_temp, int_lambda[i], int_lambda[i + 1]))

                    print("\nCalculation of BB extrapolation done!")

                    fig, ax = plt.subplots()

                    orig_lambda_plot = [p * 1e4 for p in orig_lambda]
                    new_lambda_plot = [n * 1e4 for n in new_lambda]

                    ax.plot(orig_lambda_plot, orig_flux, color='darkorange', linewidth=1.5, alpha=0.5, label='original')
                    ax.plot(new_lambda_plot, converted_flux, color='blue', linewidth=1.0, alpha=0.7, label='converted')
                    ax.scatter(new_lambda_plot, converted_flux, color='green', s=10, alpha=0.9)
                    ax.plot(new_lambda_plot, tryout_BB, color='red', linewidth=1, alpha=0.7, label='new BB extrapol.')

                    ax.set(xscale='log', yscale='log', xlim=[0.2, 30], xlabel='wavelength ($\mu$m)', ylabel='flux (erg s$^{-1}$ cm$^{-3}$)')

                    leg = ax.legend(loc='best', frameon=True, labelspacing=0.1, framealpha=0.8, fancybox=True, handlelength=1.5, handletextpad=0.2)
                    for line in leg.legendHandles:
                        line.set_linewidth(4)

                    plt.show()

                    reply = None

                    while reply not in ["yes", "no"]:

                        reply = input("\nDo you accept the new blackbody extrapolation? (BB temperature: {:.3f} K): (yes/no)\n\tEnter here:".format(BB_temp))

                        if reply == "no":

                            BB_temp = input("\n  I am sorry to read that. Please choose a new blackbody temperature and we try again: \n\tEnter here:")
                            plot_and_tweak = 'yes' # continuing manually from here, as automatic is not satisfactory

                            try:
                                BB_temp = float(BB_temp)
                            except ValueError:
                                print("Invalid choice for the blackbody temperature. Let's try again.")
                                reply = None

        if save_ascii == 'yes':

            new_lambda_micron = [lam * 1e4 for lam in new_lambda]
            orig_lambda_micron = [lam * 1e4 for lam in orig_lambda]

            # converted_flux = [o * 1e-8 for o in converted_flux]  # uncomment to convert to erg/s/cm2/A units

            if not os.path.exists("output"):
                os.makedirs("output")

            save_to_dat("./output/" + star["name"] + "_orig", orig_lambda_micron, orig_flux)
            save_to_dat("./output/" + star["name"] + "_" + convert_to, new_lambda_micron, converted_flux)

        if save_in_hdf5 == 'yes':

            h5_rm_create(f, "/" + convert_to + "/" + star["data_format"] + "/" + star["name"], converted_flux)

            if star["data_format"] == "phoenix":

                h5_rm_create(f, "/original/phoenix/lambda", orig_lambda)

            h5_rm_create(f, "/" + convert_to + "/lambda", new_lambda)

    print("Done :)")