import numpy as np
import sys
import bigfloat as bf
import phys_const as pc


def adjust_color(color, hue=0, light=0, sat=0):
    """
    Adjusts given color by changing the hue, lighting and sat values

    Hereby, 0 = original value
            1 = upper end limit
            -1 = lower end limit
            everything in-between is linear interpolation between the original value and the respective limit

    E.g., light = 1 return white, and light = -1 returns black
          light = -0.3 is darker, using 70% original lighting and 30% black
          analogously for hue and saturation
    """

    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color

    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    new_hue = c[0] + (hue >= 0) * hue * (1 - c[0]) + (hue < 0) * hue * c[0]
    new_light = c[1] + (light >= 0) * light * (1 - c[1]) + (light < 0) * light * c[1]
    new_sat = c[2] + (sat >= 0) * sat * (1 - c[2]) + (sat < 0) * sat * c[2]

    new_rgb = colorsys.hls_to_rgb(new_hue, new_light, new_sat)

    return new_rgb


def percent_counter(z, nz, y=0, ny=1, x=0, nx=1):
    """ displays percentage completed of a long operation (usually a for loop) for up to three intertwined parameters """

    percentage = float((x + nx * y + nx * ny * z) / (nx * ny * nz) * 100.0)
    sys.stdout.write("calculating: {:.1f}%\r".format(percentage))
    sys.stdout.flush()


def calculate_planck_pointwise(temp, lamda, w_unit='cm'):
    """ calculates the Planckian blackbody function at given wavelengths and temperature

    :param temp: float
                 the blackbody temperature

    :param lamda: float list or array
                  wavelength grid

    :return:     float list
                 blackbody values at the wavelength positions

    """

    if type(lamda) == float or int:

        if w_unit == 'micron':

            lamda *= 1e-4

        with bf.precision(128):

            bb_arr = 2 * pc.H * pc.C ** 2 / (lamda ** 5 * (bf.exp(pc.H * pc.C / (lamda * pc.K_B * temp)) - 1))

    else:

        bb_arr = []

        if w_unit == 'micron':
            lamda = [l * 1e-4 for l in lamda]

        with bf.precision(128):

            for l in lamda:

                bb_value = 2 * pc.H * pc.C ** 2 / (l ** 5 * (bf.exp(pc.H * pc.C / (l * pc.K_B * temp)) - 1))

                bb_arr.append(bb_value)

    return bb_arr


def calc_analyt_planck_in_interval(temp, lower_lambda, higher_lambda):
    """ calculates the planck function over a wavelength integral

    :param temp: float
                 the blackbody temperature

    :param lower_lambda: float
                         lower wavelength boundary (cm units!)

    :param higher_lambda: float
                          upper wavelength boundary (cm units!)

    :return: float
             the Planckian blackbody function integrated over a wavelength interval and averaged
    """

    d = 2.0 * (pc.K_B / pc.H)**3 * pc.K_B * temp**4 / pc.C**2
    y_top = pc.H * pc.C / (higher_lambda * pc.K_B * temp)
    y_bot = pc.H * pc.C / (lower_lambda * pc.K_B * temp)

    result = 0

    for n in range(1, 200):  # 200 found to be accurate enough.
        result += np.exp(-n*y_top) * (y_top**3/n + 3.0*y_top**2/n**2 + 6.0*y_top/n**3 + 6.0/n**4) \
                - np.exp(-n*y_bot) * (y_bot**3/n + 3.0*y_bot**2/n**2 + 6.0*y_bot/n**3 + 6.0/n**4)

    result *= d / (higher_lambda - lower_lambda)

    return result


def convert_spectrum(old_lambda, old_flux, new_lambda, int_lambda=None, type='linear', extrapolate_with_BB_T=0):

    """ converts a spectrum from one to another resolution. this method conserves energy. it is the real deal.

        :param old_lambda: list of float or numpy array
                           wavelength values of the old grid to be discarded. must be in ascending order!

        :param old_flux: list of float or numpy array
                         flux values of the old grid to be discarded.

        :param new_lambda: list of float or numpy array
                           wavelength values of the new output grid. must be in ascending order!

        :param int_lambda: (optional) list of float or numpy array
                           wavelength values of the interfaces of the new grid bins. must be in ascending order!
                           if not provided they are calculated by taking the middle points between the new_lambda values.

        :param type: (optional) 'linear' or 'log'
                     either linear interpolation or logarithmtic interpolation possible

        :param extrapolate_with_BB_T: (optional) float
                                      the out-of-boundary flux values will be extrapolated with a blackbody spectrum.
                                      set here the temperature. if not provided the out-of-boundary flux values are set to zero.

        :return: list of floats
                 flux values at the new wavelength grid points

        """

    if int_lambda is None:

        int_lambda = []

        int_lambda.append(new_lambda[0] - (new_lambda[1] - new_lambda[0]) / 2)

        for x in range(len(new_lambda) - 1):
            int_lambda.append((new_lambda[x + 1] + new_lambda[x]) / 2)

        int_lambda.append(new_lambda[-1] + (new_lambda[-1] - new_lambda[-2]) / 2)

    if extrapolate_with_BB_T > 0:

        extrapol_values = []

        print("\n\nPre-tabulating blackbody values with a temperature of " + str(extrapolate_with_BB_T) + " K ...\n")

        for i in range(len(new_lambda)):

            percent_counter(i, len(new_lambda))

            extrapol_values.append(np.pi * calc_analyt_planck_in_interval(extrapolate_with_BB_T, int_lambda[i], int_lambda[i+1]))

        print("Pre-tabulation done! \n")

    elif extrapolate_with_BB_T == 0:

        extrapol_values = np.zeros(len(new_lambda))

    else:
        raise ValueError("Error: extrapolation blackbody temperature cannot be negative.")

    print("Starting pre-conversion...\n")

    int_flux = [0] * len(int_lambda)
    new_flux = []
    old_lambda = np.array(old_lambda)  # conversion required by np.where

    if type == 'linear':

        for i in range(len(int_lambda)):

            percent_counter(i, len(int_lambda))

            if int_lambda[i] < old_lambda[0]:
                continue

            elif int_lambda[i] > old_lambda[len(old_lambda) - 1]:
                break

            else:
                p_bot = len(np.where(old_lambda < int_lambda[i])[0]) - 1

                interpol = old_flux[p_bot] * (old_lambda[p_bot + 1] - int_lambda[i]) + old_flux[p_bot + 1] * (int_lambda[i] - old_lambda[p_bot])
                interpol /= (old_lambda[p_bot + 1] - old_lambda[p_bot])
                int_flux[i] = interpol

        print("\n  Pre-conversion done!")

        print("\nStarting main conversion...\n")

        for i in range(len(new_lambda)):

            percent_counter(i, len(new_lambda))

            if int_flux[i] == 0 or int_flux[i+1] == 0:
                new_flux.append(extrapol_values[i])

            else:
                p_bot = len(np.where(old_lambda < int_lambda[i])[0]) - 1

                p_start = p_bot + 1

                for p in range(p_start, len(old_lambda)):

                    if p == p_start:
                        if old_lambda[p_start] < int_lambda[i + 1]:
                            interpol = (int_flux[i] + old_flux[p]) / 2.0 * (old_lambda[p] - int_lambda[i])

                        else:
                            interpol = (int_flux[i] + int_flux[i + 1]) / 2.0
                            break
                    else:
                        if old_lambda[p] < int_lambda[i + 1]:
                            interpol += (old_flux[p - 1] + old_flux[p]) / 2.0 * (old_lambda[p] - old_lambda[p - 1])

                        else:
                            interpol += (old_flux[p - 1] + int_flux[i + 1]) / 2.0 * (int_lambda[i + 1] - old_lambda[p - 1])
                            interpol /= (int_lambda[i + 1] - int_lambda[i])
                            break

                new_flux.append(interpol)

    elif type == 'log':

        for i in range(len(int_lambda)):

            percent_counter(i, len(int_lambda))

            if int_lambda[i] < old_lambda[0]:
                continue

            elif int_lambda[i] > old_lambda[len(old_lambda) - 1]:
                break

            else:
                p_bot = len(np.where(old_lambda < int_lambda[i])[0]) - 1

                interpol = old_flux[p_bot] ** (old_lambda[p_bot + 1] - int_lambda[i]) * old_flux[p_bot + 1] ** (int_lambda[i] - old_lambda[p_bot])
                interpol = interpol**(1/(old_lambda[p_bot + 1] - old_lambda[p_bot]))
                int_flux[i] = interpol

        print("\n  Pre-conversion done!")

        print("\nStarting main conversion...\n")

        for i in range(len(new_lambda)):

            percent_counter(i, len(new_lambda))

            if int_flux[i] == 0 or int_flux[i + 1] == 0:
                new_flux.append(extrapol_values[i])

            else:
                p_bot = len(np.where(old_lambda < int_lambda[i])[0]) - 1

                p_start = p_bot + 1

                for p in range(p_start, len(old_lambda)):

                    if p == p_start:
                        if old_lambda[p_start] < int_lambda[i + 1]:
                            interpol = (int_flux[i] * old_flux[p]) ** (0.5 * (old_lambda[p] - int_lambda[i]))

                        else:
                            interpol = (int_flux[i] * int_flux[i + 1]) ** 0.5
                            break
                    else:
                        if old_lambda[p] < int_lambda[i + 1]:
                            interpol *= (old_flux[p - 1] * old_flux[p]) ** (0.5 * (old_lambda[p] - old_lambda[p - 1]))

                        else:
                            interpol *= (old_flux[p - 1] * int_flux[i + 1]) ** (0.5 * (int_lambda[i + 1] - old_lambda[p - 1]))
                            interpol = interpol ** (1/(int_lambda[i + 1] - int_lambda[i]))
                            break

                new_flux.append(interpol)

    print("\n  Main conversion done!")

    return new_flux


def read_helios_tp(file):
    """ reads temperature pressure profile from a file, incl. potential convective zones

    :param file: string
                 name/path of file to be read

    :return 1: pressure array
    :return 2: temperature array
    :return 3: pressures in convective zone 1, if exists
    :return 4: temperatures in convective zone 1, if exists
    :return 5: pressures in convective zone 2, if exists
    :return 6: temperatures in convective zone 2, if exists
    :return 7: pressures in convective zone 3, if exists
    :return 8: temperatures in convective zone 3, if exists

    """

    temp = []
    press = []
    convective = []

    temp_conv0 = []
    press_conv0 = []
    temp_conv1 = []
    press_conv1 = []
    temp_conv2 = []
    press_conv2 = []
    temp_conv3 = []
    press_conv3 = []

    with open(file, "r") as file:
        next(file)
        next(file)
        for line in file:
            column = line.split()
            press.append(float(column[2]) * 1e-6)
            temp.append(float(column[1]))
            try:
                convective.append(float(column[6]))
            except (IndexError, ValueError):
                convective.append(0)

    i_break = len(press) - 2

    for i in range(len(press)-1):
        if convective[i] == 1:

            temp_conv0.append(temp[i])
            press_conv0.append(press[i])
            if convective[i + 1] == 0:
                i_break = i
                break

    i_break2 = len(press) - 2

    for i in range(i_break + 1, len(press)-1):
        if convective[i] == 1:
            temp_conv1.append(temp[i])
            press_conv1.append(press[i])
            if convective[i + 1] == 0 or i == len(press) - 2:
                i_break2 = i
                break

    i_break3 = len(press) - 2

    for i in range(i_break2 + 1, len(press)-1):
        if convective[i] == 1:
            temp_conv2.append(temp[i])
            press_conv2.append(press[i])
            if convective[i + 1] == 0 or i == len(press) - 2:
                i_break3 = i
                break

    for i in range(i_break3 + 1, len(press)-1):
        if convective[i] == 1:
            temp_conv3.append(temp[i])
            press_conv3.append(press[i])

    return press, temp, press_conv0, temp_conv0, press_conv1, temp_conv1, press_conv2, temp_conv2, press_conv3, temp_conv3


def read_helios_spectrum(file, type='emission'):
    """

    :param file: string
                 name/path of file to be read

    :param type: accepted input 'emission' or 'eclipse'
                 sets the type of the spectrum which should be reach

    :return 1: wavelength array
    :return 2: TOA emission spectrum or secondary eclipse spectrum array, depending on the 'type' choice
    """

    lamda = []
    spec = []

    with open(file, "r") as f:
        next(f)
        next(f)
        next(f)
        for line in f:
            column = line.split()
            lamda.append(float(column[1]))
            if type == 'emission':
                spec.append(float(column[5]))
            elif type == 'eclipse':
                spec.append(float(column[6]))
            else:
                raise ValueError("Unknown input for spectrum type!")

    return lamda, spec


def rebin_spectrum_to_resolution(old_lamda, old_flux, resolution=1000, w_unit='cm', type='linear'):
    """ rebins a given spectrum to a new resolution

    :param old_lambda: list of float or numpy array
                           wavelength values of the old grid to be discarded.
                           must be in ascending order!

    :param old_flux: list of float or numpy array
                     flux values of the old grid to be discarded.

    :param resolution: float
                       resolution (R=lamda/delta_lamda) of the new, rebinned wavelength grid

    :param w_unit: 'cm'/'micron'
                   the units of the old wavelength values. will be the output units as well.

    :return 1: wavelength values of the rebinned grid
    :return 2: flux values of the rebinned grid
    """

    if w_unit == 'micron':
        old_lamda = [l * 1e-4 for l in old_lamda]

    bot_limit = old_lamda[0]

    top_limit = old_lamda[-1]

    rebin_lamda = []
    l_point = bot_limit

    while l_point < top_limit:
        rebin_lamda.append(l_point)

        l_point *= (resolution + 1) / resolution

    rebin_flux = convert_spectrum(old_lamda, old_flux, rebin_lamda, type=type, extrapolate_with_BB_T=0)

    if w_unit == 'micron':
        rebin_lamda = [l * 1e4 for l in rebin_lamda]

    return rebin_lamda, rebin_flux


def read_helios_contribution(path):

    press = []
    lamda = []
    flux = []

    if path.endswith('/'):
        path = path[:-1]
    suffix = path.split("/")[-1]

    tp_file = path+"/"+suffix+"_tp.dat"
    flux_file = path+"/"+suffix+"_spec_upflux.dat"
    contri_file = path+"/"+suffix+"_contribution.dat"

    # read in the pressure array
    with open(tp_file, "r") as tpfile:
        next(tpfile)
        next(tpfile)
        for line in tpfile:
            column = line.split()
            press.append(float(column[2]) * 1e-6)

    # read in the TOA flux
    with open(flux_file, "r") as specfile:
        next(specfile)
        next(specfile)
        next(specfile)
        for line in specfile:
            column = line.split()
            lamda.append(float(column[1]))
            flux.append(float(column[-1]))

    nlayer = len(press)
    nbin = len(lamda)
    contri = np.zeros([nlayer, nbin])

    # read in the contribution function
    with open(contri_file, "r") as contrifile:
        next(contrifile)
        next(contrifile)
        index = 0
        for line in contrifile:
            column = line.split()
            for c in range(len(column)):
                if c > 3:
                    contri[c - 4][index] = (float(column[c]) / flux[index]) * 100

            index += 1

    return lamda, press, contri


def read_helios_transweight(path):

    press = []
    lamda = []

    if path.endswith('/'):
        path = path[:-1]
    suffix = path.split("/")[-1]

    tp_file = path + "/" + suffix + "_tp.dat"
    flux_file = path + "/" + suffix + "_spec_upflux.dat"
    contri_file = path + "/" + suffix + "_transweight.dat"

    # read in the pressure array
    with open(tp_file, "r") as tpfile:
        next(tpfile)
        next(tpfile)
        for line in tpfile:
            column = line.split()
            press.append(float(column[2]) * 1e-6)

    # read in lamda
    with open(flux_file, "r") as specfile:
        next(specfile)
        next(specfile)
        next(specfile)
        for line in specfile:
            column = line.split()
            lamda.append(float(column[1]))

    nlayer = len(press)
    nbin = len(lamda)
    transw = np.zeros([nlayer, nbin])

    # read in the transmission weighting function
    with open(contri_file, "r") as contrifile:
        next(contrifile)
        next(contrifile)
        index = 0
        for line in contrifile:
            column = line.split()
            for c in range(len(column)):
                if c > 3:
                    transw[c - 4][index] = float(column[c])

            index += 1

    return lamda, press, transw
