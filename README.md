# HELIOS #

#### A GPU-ACCELERATED RADIATIVE TRANSFER CODE FOR EXOPLANETARY ATMOSPHERES ####

###### Copyright (C) 2018 Matej Malik ######

### About ###

HELIOS is an open-source radiative transfer code, which is constructed for studying exoplanetary atmospheres in their full variety. The model atmospheres are one-dimensional and plane-parallel, and the equation of radiative transfer is solved in the hemispheric two-stream approximation with non-isotropic scattering. For given opacities and planetary parameters, HELIOS finds the atmospheric temperature profile in radiative-convective equilibrium and the corresponding planetary emission spectrum.

HELIOS is part of the Exoclimes Simulation Platform ([ESP](http://www.exoclime.net)).

If you use HELIOS for your own work, please cite its two method papers: [Malik et al. 2017](http://adsabs.harvard.edu/abs/2017AJ....153...56M) and *Malik et al. 2018, AJ, under review.*.

Any questions, issues or bug reports are appreciated and can be sent to *matej.malik@csh.unibe.ch*. 

Thank you for considering HELIOS!

### Documentation ###

A detailed documentation of HELIOS can be found [here](https://heliosexo.readthedocs.io/).


































### 4. Execution ###

As HELIOS comes as a Python package it is not required to be compiled. Simply run the file **helios_main.py** with Python. For example, go to the installation directory and type the following into your console/terminal:

    python helios_main.py

Make sure that your Python distribution has access to all the required libraries and packages (see sect. Requirements).

### 8. K-table Generator ###

The provided k-table generator is a (optional) tool that converts the HELIOS-K output to an opacity table (k-table), which can be used in HELIOS.

#### Execution ####

Run the file **generate_ktable.py** with Python. For example, in the **ktable** directory type into your console/terminal

    python generate_ktable.py .

#### Input ####

##### HELIOS-K Output Format #####

You need to use the standard output format of HELIOS-K. For a suite of HELIOS-K calculations applied to a grid of temperatures and pressures you set in the **param.dat** file (in the HELIOS-K directory) the parameters "doResampling" = 1 and "ReplaceFile" = 0. Doing this, you obtain one(!) **Info_\*_.dat** and one(!) **Out_\*_cbin.dat** for each molecule. These two files can be read by the k-table generator.

##### Input Parameters #####

The input parameters are set in the file **input_ktable.dat**. The individual parameters are the following with input type and suggested values in round brackets. The units are given in square brackets.

    Helios-k output path ("system path to parent directory")

The path to the HELIOS-K output. It is sufficient to define the path to a common parent directory of subdirectories containing the individual HELIOS-K output files.

    output format (number; typically 1)

The format of the HELIOS-K output. If following the instructions from above, set this parameter to 1.

    elemental oxygen abundance  (number; solar value ca. 5e-4)
    elemental carbon abundance (number; solar value ca. 2.5-4)

Sets the elemental abundances of oxygen and carbon in the atmosphere. These values serve as the basis for the chemical abundance calculations. One may want to start with values of the solar photosphere.

    mean molecular weight \[m_p\] (number; typically 2.3 or 2.4 for H-dominated atmospheres)

Sets the mean molecular weight of the atmosphere in units of the proton mass. This value depends on the atmospheric composition.

#### Output ####

Output files are written into the **output** subdirectory. There is first a HDF5 container built for each opacity source separately and then a final container **mixed_opacities.h5** with the combined opacities according to their mixing ratios. This latter file is the opacity table intended to be used in HELIOS. Point the "path to opacity file" in the HELIOS **input_param.dat** to this file.

For more information on the structure of the k-table generator output, see the produced file **ktable_info.txt**.


