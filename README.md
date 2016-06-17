# HELIOS #

#### A GPU-ACCELERATED RADIATIVE TRANSFER CODE FOR EXOPLANETARY ATMOSPHERES ####

###### Copyright (C) 2016 Matej Malik ######

### Contents ###

1. Foreword

2. Requirements

3. Installation

4. Execution

5. Input Files

6. Directory Structure

7. Input Parameters

8. K-table Generator

9. Using an Own Opacity Table

10. License

11. Final Remarks

12. Acknowledgements

### 1. Foreword ###

HELIOS is an open-source radiative transfer code, which is constructed for studying exoplanetary atmospheres. In its initial version, the model atmospheres of HELIOS are one-dimensional and plane-parallel, and the equation of radiative transfer is solved in the two-stream approximation with non-isotropic scattering. 

The optimal application of HELIOS is in combination with the opacity calculator [HELIOS-K](https://github.com/exoclime/HELIOS-K/), which provides the necessary molecular opacities. To construct the opacity table from the output of HELIOS-K, we use a small k-table generator program, which combines the k-distribution functions of the individual opacity sources by weighing the molecular abundances with analytical chemistry formulae.

Naturally, HELIOS can be used alone without HELIOS-K as as a pure radiative transfer solver. However, in this case the opacities need to be provided in the correct HDF5 file format (see section "Input files" for details).

### 2. Requirements ###

#### Python ####

HELIOS's computational core is written in CUDA C++, but the user shell comes in Python modular format. To communicate between the Host and the Device the PyCUDA interface is used.

The following Python packages are required, which, depending on your Python distribution, might already be included. Otherwise they need to be installed manually, e.g. with the Python package manager pip.

- numpy
- matplotlib
- h5py
- PyCUDA (obtained from [here](https://mathema.tician.de/software/pycuda/) and all requirements therein)

Note that HELIOS was developed and tested with Python 3.4. It has not been tested with earlier version of Python.

#### CUDA ####

The NVIDIA CUDA API can be found [here](https://developer.nvidia.com/cuda-downloads).

#### OS ####

HELIOS was developed on a machine with Mac OS X 10.9 and also successfully tested on Ubuntu and Archlinux. It has not been tested to run on other OS, but should in principle work if the requirements are met.

### 3. Installation ###

For the installation you have two choices:

- Clone the GitHub repository to a local directory by typing

> git clone https://github.com/exoclime/HELIOS

- Download the ZIP archive, containing all the necessary files. Then unpack the ZIP to a local directory.

### 4. Execution ###

As HELIOS comes as a Python package it is not required to be compiled. Simply run the file **helios_main.py** with Python. For example, go to the installation directory and type the following into your console/terminal:

> python helios_main.py

Make sure that your Python distribution has access to all the required libraries and packages (see sect. Requirements).

### 5. Input Files ###

The following files are read into HELIOS.

##### Opacity table file (required) #####

The easiest way to obtain a ready-to-use opacity table is to run the k-table generator program, which is provided in the subdirectory **ktable**. For instructions how to use the k-table generator program, see section 8. However, if you want to use an own opacity table it becomes somewhat tricky, see section 9 for instructions.
                                          
##### Stellar spectrum file (optional) #####

*This option is still under development. Stay tuned for updates.* 

##### Restart temperature file (optional) #####

The restart temperature file is produced automatically by a HELIOS run and is named **"name"_restart_tp.dat**. If you want to start with this TP-profile simply set the path in **input_param.dat** accordingly.

### 6. Directory Structure ###

The parent directory consists of the main HELIOS files. The output files appear in the **output** subdirectory. The k-table generator program is located per default in the subdirectory **ktable**, but can be moved to other locations, as it works independently of the main HELIOS code. In general, it is recommended to leave the directory structure as it is when downloaded from GitHub.

### 7. Input Parameters ###

The input parameters are set in the file **input_param.dat**. The individual parameters are the following, with input type and suggested values in round brackets. The units are given in square brackets.

> isothermal layers (yes/no)

Determines whether the model uses a constant temperature across each layer (=isothermal layers) or expands the Planck function to first order (=non-isothermal layers). Usually for the temperature iteration non-isothermal layers are recommended, whereas for pure post-processing purposes isothermal layers are sufficient.

> number of layers (number; typically 50 - 200)

Sets the number of vertical layers in the grid. Usually a value around 100 provides a reasonable compromise between accuracy and computational effort. 

> TOA pressure \[10^-6 bar\] (number; typically 1)

The pressure value at the topmost simulated atmospheric layer.

> BOA pressure \[10^-6 bar\] (number; typically 1e9)

The pressure value at the bottommost simulated atmospheric layer. The model construct a grid between the BOA and TOA pressures equidistant in log P.

> pre-tabulate (yes/no)

HELIOS offers the possibility to pre-tabulate transmission function across opacity. This decreases significantly the running time of the model. It is recommended to set this value to "yes" when iterating for the temperature. For pure post-processing this can be set to "no", which causes the model to calculate the transmission in each layer on the fly. Note that setting "yes" uses substantially more GPU memory so depending on the hardware this option may lead to an insufficient memory error message.

> post-processing only (yes/no)

Sets whether the model proceeds with the temperature iteration routine ("no") or just propagates the fluxes once through the layers ("yes"). In the latter case, there is no iteration happening and the output temperature will be the same as the input temperature. Setting this option to “yes” is typically used to produce a high resolution spectrum from a given TP-profile.

> restart temperatures (yes/no)

Determines whether the initial temperature profile is read from **"name"_restart_tp.dat**. This causes the model to start from a given TP-profile. Otherwise, if this value is set to "no", the effective irradiated temperature of the planet is used initially. Usually, this option is used in compination with the “”post-processing only” option. WARNING: The pressures values in **"name"_restart_tp.dat** are not read into the code. The grid is constructed solely from the TOA and BOA pressure values. 

> path to restart temperatures ("system path to file")

Sets the path to the file **"name"_restart_tp.dat**. This parameter is irrelevant if the option "restart temperatures" is set to "no".

> varying timestep (yes/no)

Determines whether the model uses a fixed timestep ("no") or a varying, adaptive timestep ("yes") to advance during the temperature iteration process. It is recommended to set this parameter always to "yes". If "post-processing only" is set to "yes", this parameter is irrelevant.

> timestep \[s\] (number; typically 1e2 - 1e3)

Sets the length of the fixed timestep. Is irrelevant if "varying timestep" is set to "yes".

> scattering (yes/no)

Determines whether Rayleigh scattering is used in the model. For this to work the opacity file needs to possess the Rayleigh scattering cross-sections.

> exact solution (yes/no)

Use the exact solution to the two-stream equations. This only works if scattering is disabled. Hence if this parameter is set to "yes", the parameter "scattering" is overwritten to "no” - also the value of the diffusivity factor becomes irrelevant. As a rule of thumb, if you want to model absorption only, then the exact solution provides the most accurate results.

> path to opacity file ("system path to file")

Sets the path to the opacity table file.

> diffusivity factor (number; typically 1.5 - 2)

Sets the value of the diffusivity factor. If you are not sure, pick 2.

> f factor (number; typically 0.6667 = dayside-redistribution or 1 = no redistribution or 0.25 = full redistribution)

The f factor determines the heat redistribution efficiency in the atmosphere. For day-side emission spectra one typically assumes f = 2/3 = 0.6667. 

> internal temperature \[K\] (number; typically 0 - 300)

The internal temperature determines the strength of the internal heating. If internal heating is negligible on the resulting spectrum (e.g. strongly irradiated planets) it is safe to assume this parameter as zero.

> asymmetry factor g_0 (number; between -1 and 1)

Determines the scattering orientation. As Rayleigh scattering is mostly isotropic, it is recommended to choose zero. A positive value implies forward scattering and a negative value backward scattering. 

> mean molecular weight \[m_p\] (number; typically 2.3 or 2.4 for H-dominated atmospheres)

Sets the mean molecular weight of the atmosphere in units of the proton mass. This value depends on the atmospheric composition. 

> pre-tabulated planets (pick a name from the provided list, or manual)

The planetary parameters can be either specified manually or picked from the pre-tabulated parameter lists. The pre-tabulated numbers are provided WITHOUT ANY WARRANTY for accuracy. You are welcome to change/update these numbers in the file **planets_and_stars.py**. Naturally, you can also add more planets.

> surface gravity \[cm s^-2\] (number)

> orbital distance \[AU\] (number)

> radius star \[R_sun\] (number)

> temperature star \[K\] (number) 

Manual entry for the parameters. See above description.

> model (name; possible inputs: blackbody, kurucz, phoenix)

Sets the model for the stellar irradiation. Simplest approach is to use a blackbody shape with the stellar temperature. For this choice no additional input is required. If a sophisticated stellar spectrum is desired, the spectrum needs to be provided in a HDF5 file with the correct format.

*The implementation of stellar spectra is still under development. This feature may be added in future. Please use the blackbody option for the moment.*

> path to stellar model file ("system path to file")

Sets the path to the file with the stellar spectrum. If chosen blackbody for the stellar irradiation this parameter is irrelevant. 

> name (name; only your imagination limits this input)

The output files will begin with this string.

> number of run-in timesteps (number; typically 0)

Number of iteration steps before the temperature starts to evolve. Usually used only for video producing purposes. *This is still an experimental feature.*

> artificial shortw. opacity (number; typically 0)

For exploration of the effects of additional shortwave absorption there is the possibility to add artifcial opacity for wavelengths < 1 micron. This opacity acts as a continuum absorption at the given value. 

> realtime plotting (yes/no)

This parameter activates a realtime plotting routine showing the evolution of the TP-profile during the temperature iteration process. This parameter is irrelevant if "post-processing only" is activated.

#### Recommended settings ####

Here are some recommended settings specifically for the following purposes.

##### Temperature iteration #####

> isothermal layers = no

> pre-tabulate = yes

> post-processing only = no

> restart temperatures = no

> varying timestep = yes

##### Spectrum generation (post-processing only) #####

> isothermal layers = yes

> pre-tabulate = no

> post-processing only = yes

> restart temperatures = yes

### 8. K-table Generator ###

The provided k-table generator is a (optional) tool that converts the HELIOS-K output to an opacity table (k-table), which can be used in HELIOS.

#### Execution ####

Run the file **generate_ktable.py** with Python. For example, in the **ktable** directory type into your console/terminal

> python generate_ktable.py .

#### Input ####

##### HELIOS-K Output Format #####

You need to use the standard output format of HELIOS-K. For a suite of HELIOS-K calculations applied to a grid of temperatures and pressures you set in the **param.dat** file (in the HELIOS-K directory) the parameters "doResampling" = 1 and "ReplaceFile" = 0. Doing this, you obtain one(!) "Info_\*_.dat" and one (!) "Out_\*_cbin.dat" for each molecule. These two files can be read by the k-table generator.

##### Input Parameters #####

The input parameters are set in the file **input_ktable.dat**. The individual parameters are the following with input type and suggested values in round brackets. The units are given in square brackets.

> Helios-k output path ("system path to parent directory")

The path to the HELIOS-K output. It is sufficient to define the path to a common parent directory of subdirectories containing the individual HELIOS-K output files.

> output format (number; typically 1)

The format of the HELIOS-K output. If following the instructions from above, set this parameter to 1.

> elemental oxygen abundance  (number; solar value ca. 5e-4)
                 
> elemental carbon abundance (number; solar value ca. 2.5-4)

Sets the elemental abundances of oxygen and carbon in the atmosphere. These values serve as the basis for the chemical abundance calculations. One may want to start with values of the solar photosphere.

> mean molecular weight \[m_p\] (number; typically 2.3 or 2.4 for H-dominated atmospheres)

Sets the mean molecular weight of the atmosphere in units of the proton mass. This value depends on the atmospheric composition. 

#### Output ####

Output files are written into the **output** subdirectory. There is first a HDF5 container built for each opacity source separately and then a final container **mixed_opacities.h5** with the combined opacities according to their mixing ratios. This latter file is the opacity table intended to be used in HELIOS. Point the "path to opacity file" in the HELIOS **input_param.dat** to this file. 

For more information on the structure of the k-table generator output, see the produced file **ktable_info.txt**.

### 9. Using an Own Opacity Table ###

At this moment (unfortunately), using an own opacity table is somewhat tricky. HELIOS can only read a specific format! Hence, each opacity file, which is to be used by HELIOS, needs first to be converted to a HDF5 file possessing the following datasets.

> pressures                            

Array of pressure values used for the calculation of the opacities in ascending order

> temperatures
                         
Array of temperature values used for the calculation of the opacities in ascending order

> interface wavelengths

Array of wavelengths at the interfaces of the spectral bin

> centre wavelengths                   

Array of wavelengths of the bin centers

> wavelength width of bins

Array of the width of the bins

> ypoints

Abszissa points for the 20th order Gauss-Legendre quadrature rule (= roots of the 20th order Legendre Polynomial), applied to the interval [0,1]. At these points the k-distribution function is evaluated.

> kpoints                              

opacity values in the format: opacity\[Y-point, Lambda, Press, Temp\] = kpoints\[y + n_y*l + n_y*n_l*p + n_y*n_l*n_p*t\], where n_* the length of the according list and y, l, p, t are the indices in the according lists, e.g. Temp = temperatures\[t\], Lambda = centre wavelengths\[l\], etc.

> cross rayleigh

(Rayleigh) scattering cross sections in the format: Rayleigh cross section\[Lambda\] = cross_rayleigh\[l\], where Lambda = centre wavelengths\[l\].

WARNING: This early version of HELIOS only supports opacities that are given in the k-distribution format, with 20 opacity values located at Gauss-Legendre points (Y-points) within each wavelength bin.

### 10. License ###

HELIOS is distributed under the terms of the GNU General Public License (GPL) license. For more information see either the file **license.txt** in the main HELIOS directory or see [here](http://www.gnu.org/licenses/).

### 11. Final Remarks ###

HELIOS is part of the exoclime simulation platform ([ESP](http://www.exoclime.net)), which also incorporates the [THOR](https://github.com/exoclime/THOR/) and [VULCAN](https://github.com/exoclime/VULCAN/) projects. HELIOS-K, the fast GPU based opacity calculator, is found [here](https://github.com/exoclime/HELIOS-K/).

Any questions, issues or bug reports are appreciated and can be reported to *matej.malik@csh.unibe.ch*. Thank you for considering HELIOS!

### 12. Acknowledgements ###

Thanks to ...

- the [PyCUDA](https://mathema.tician.de/software/pycuda/) developers to allow us to use the efficiency of GPU computations and still remain at the Pythonic *no syntax* level.

- NVIDIA in particular and the whole Video Gaming industry/community in general for enabling large advances in GPU technology in the past years.

