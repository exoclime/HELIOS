==================
**ktable Program**
==================

General Info
============

The ktable program is an included tool that converts HELIOS-K output opacity files into opacity tables that can be used in by HELIOS. The most straightforward way is to use the pre-calculated HELIOS-K opacity that can be downloaded from the `online opacity database <https://dace.unige.ch/opacityDatabase/>`_ at the University of Geneva. Always download the whole temperature and pressure ranges and extract all the files into a directory. Each species should have a separate directory.

The ktable program is run by typing::

	python3 ktable.py

while being in the ``ktable`` directory. All parameters are set in the parameter file, per default named ``param_ktable.dat``. (The file name can be changed via the command-line option '--parameter_file'.) The source code files are located in the ``source_ktable`` subdirectory, and additional input files are conveniently in the input directory (though all input file paths can be modified). All output paths are set in the parameter file.

As with the main HELIOS code, most input parameters can be set via command-line. Parameters that have provide this option are marked as (CL:Y) and those that do not as (CL:N). The command-line option has the **same name** as the respective parameter given in ``param_ktable.dat`` with the following limitations:

- small letters only
- all spaces and dash symbols are replaced by an underscore
- without the square brackets and their content
- no dependency information (parameter name starts after the arrow)

Main Workflow
-------------

The ktable program works in **two stages**.

In the first stage, for each species the HELIOS-K output files are converted to a single HDF5 file containing the pre-tabulated opacity. As opacity format, HELIOS supports **'opacity sampling'** and the **'k-distribution method'**. If sampling is selected, the high-resolution opacity is merely interpolated to the HELIOS wavelength grid. For the 'k-distribution method' the high-resolution opacity is converted to k-coefficients with chosen bins and number of Gaussian points (note that only 20 Gaussian points are currently supported by the RO method in HELIOS).

In the second stage, the individual opacities are interpolated to a common temperature-pressure grid, weighted by the respective molecule's mixing ratio and combined to the final mixed opacity table. This final mixed opacity table can then be used in HELIOS when the 'premixed' setting is selected. For 'on-the-fly' opacity mixing, the individual opacity files are used. Note that since the individual opacities have to be on the same temperature-pressure grid, the **interpolated files have to be used** for that purpose, i.e., the files that have '_ip_' in their name. (Obviously, the opacities have to be on the same wavelength grid as well.)

Parameter File
==============

Below a detailed explanation of the input parameters as found in the parameter file.

First Stage
-----------

   ``individual species calculation   [yes, no]   (CL: Y)``

This determines whether the first stage will be executed. If not, the program starts directly at the second stage. Set 'yes', when starting from HELIOS-K output and you need to produce the individual opacity files. Set 'no', if you already have the individual files and just want to produce a new mixed file.

   ``format   [k-distribution, sampling]   (CL: Y)``

HELIOS supports opacity tables in two formats: sampling and k-distribution. The k-distribution approach is more accurate when calculating the global energy budget of the atmosphere and the goal is finding the equilibrium T-P profile. The opacity sampling approach allows for a higher resolution in wavelength than the k-distribution method for given hardware costs (because only 1 opacity value per wavelength point instead of 20) and thus in order to generate a planetary spectrum with many spectral points 'sampling' is the way to go.

   ``HELIOS-K output format   [binary, text]   (CL: Y)``

The format of the HELIOS-K output files. The files from the online database come in binary format (to reduce their size). Per default though, HELIOS-K generates output files in text (ASCII) format. Files of different format cannot be mixed in the same directory.

   ``path to individual species file   [file path]   (CL: Y)``

Path to the file which lists all species to be used for the production of individual opacity files (= first stage calculation).

   ``grid format   [fixed_resolution, file]   (CL: Y)``

For the opacities and the HELIOS calculation, either a fixed resolution grid in wavelength can be used, or specific wavelengths can be read from a file. Fixed resolution means that R = delta_lambda / lambda is constant throughout the grid.

   ``wavelength grid   [resolution, lower limit, upper limit [micron]]   (CL: N)``

This defines the wavelength grid to be used. First parameter is the resolution, R = delta_lambda / lambda, followed by the lower and upper wavelength limits of the grid. The limits are in micron. Note that if opacity sampling is used those limits set the first and last wavelength points. If the k-distribution method is used, those limits set the lower interface of the first wavelength bin and the upper interface of the last bin. *This parameter is only used if grid format is set to 'fixed_resolution'.*

   ``path to grid file   [path to file]   (CL: Y)``

Path to the a file with the wavelength grid. The format is a text file with a single column listing the wavelengths in cm(!). Note that if opacity sampling is used, the listed values directly set the wavelength points. However, if the k-distribution method is used, the listed values set the wavelength bin interfaces. *This parameter is only used if grid format is set to 'file'.*

   ``number of Gaussian points   [number > 1]   (CL: Y)``

Number of Gaussian points in a wavelength bin. Important: currently the RO method is *hard-coded to require 20 points*. If not using RO, this number can be anything > 1.

   ``directory with individual files   [directory path]   (CL: Y)``

This sets the directory where the individual opacity files are written, i.e., the output directory of the first stage calculation.

Second Stage
------------

   ``mixed table production   [yes, no]   (CL: Y)``

Determines whether the second stage calculation will be executed. If set to 'no', the ktable program stops after producing the individual opacity files without combining them.

   ``path to final species file   [file path]   (CL: Y)``

This sets the path to the file which lists all the species to be included in the final, combined opacity table.

   ``path to FastChem output   [directory path]   (CL: Y)``

This sets the path to the directory with the FastChem output files. Only necessary if at least one species obtains its mixing ratio from FastChem.

   ``mixed table output directory   [directory path]   (CL: Y)``

This sets the path to the directory where the final, mixed opacity table is written. If all goes well and the whole ktable program runs through,  either ``mixed_opac_kdistr.h5`` or ``mixed_opac_sampling.h5`` will appear in that directory, depending on the opacity format used.

   ``units of mixed opacity table   [CGS, MKS]   (CL: Y)``

This sets the units of the opacity in the final, mixed table. For HELIOS, always use 'CGS'. However, if using the table for another RT code that employs MKS units, there is an option for that too.

Input Files Format
==================

The installation comes with reference examples for all the required input files.

Individual Species File
-----------------------

There is an example file ``ktable/input/individual_species.dat`` included in the HELIOS installation (just make a copy of the file and modify it for your own purpose.)

For each species that is to be processed (= an opacity file is produced), one first sets the name and then the respective path to the directory with the HELIOS--K output files. The name of the species can be set quite arbitrarily, as it simply determines how the output files are named.

.. _final-species-file:

Final Species File
------------------ 

There is an example file ``ktable/input/final_species.dat`` included in the HELIOS installation (just make a copy the file and modify for your own purpose.).

First, the chosen name in this file needs to coincide with the name of the opacity file for this species. Then, one sets whether this species should be included as absorber of scatterer in the final table. Lastly, one needs to choose how the mixing ratio is included. Two options exist, 'FastChem' and a numerical value. If 'FastChem' is set, the FastChem output is read (see next parameter which sets the file path for that). If a number is inserted, a constant mixing ratio of this value is assumed. 

For CIA opacity, if setting a constant mixing ratio, one needs to include a value for each collision pair and so two numbers have to be given, separated by a '&'. For instance, 0.9&0.1 is a valid input.

Note that **each species in the species file has to exist in the species database** ``source/species_database.py`` because the properties are pulled from there. Most of the common species should already be pre-defined. If an error is returned that there is no such entry in ``species_database.py`` a new one has to be manually created. When creating a new entry just follow the format of the existing ones. The FastChem name can be looked up in the FastChem output file. The weight parameter is the species' molecular weight in AMU (or the molar weight in g). For CIA pairs, it is the weight of the secondly-listed molecule.

Not every species can be included as scatterer. At the moment, the Rayleigh cross-sections for the following species are included (plus references):

   * H2: Cox 2000
   * He: Sneep & Ubachs 2005, Thalman et al. 2014
   * H: Lee & Kim 2004
   * H2O: Murphy 1977, Wagner & Kretzschmar 2008
   * CO: Sneep & Ubachs 2005
   * CO2: Sneep & Ubachs 2005, Thalman et al. 2014
   * O2: Sneep & Ubachs 2005, Thalman et al. 2014
   * N2: Sneep & Ubachs 2005, Thalman et al. 2014
   * e--: Thomson scattering cross-section from 'astropy.constants' package.


Lastly, the bound-free and free-free absorption of H- and the free-free absorption of He- can be included. If including H-, the free-free and bound-free contributions have to be listed as two separate species, i.e., H-_ff and H-_bf. If using constant mixing ratios, the mixing ratio of H- is set for H-_bf (because the electron is bound = H-) and the mixing ratios of e- and H are set for H-_ff (because here the e- is unbound around a neutral H) separated by a '&', analogously to the CIA pairs.

No additional files have to be provided when including H-_ff, H-_bf and He- because these opacities are calculated directly using the approximations from `John 1988 <https://ui.adsabs.harvard.edu/abs/1988A%2526A...193..189J>`_ and `John 1994 <https://ui.adsabs.harvard.edu/abs/1994MNRAS.269..871J>`_. Note that there is a typo in John 1988. The value for alpha, in the line underneath Eq. (3), should be 1.439e4 instead of 1.439e8. (Actually, this value is never used here because the mixing ratio of H- is taken from FastChem, which is more accurate than the Saha equation approximation of John 1988.)

.. _ktable-code-structure:

ktable Code Structure
=====================

In the main directory there is:

- ``ktable.py``: the main run file

- ``param_ktable.dat``: main parameter file. This file can be renamed and, if renamed, included via the command-line option '-parameter_file'.

The ``source_ktable`` directory contains the source code with the files:

- ``param.py``: reads the parameter file and command-line options

- ``build_individual_opacities.py``: generates the individual opacity files from HELIOS-K output

- ``combination.py``: interpolates the individual opacities, adds scattering, weights with the respective mixing ratios and combines everything to a final, mixed opacity table

- ``continuous.py``: calculates the continuous opacities of the H- and He- ions

- ``rayleigh.py``: calculates the Rayleigh scattering cross sections for all included species

- ``information.py``: writes a text file next to the final opacity table describing the contents and format.

Lastly, input data and files are usually included in the ``input`` subdirectory, though all paths can be in the freely chosen in the parameter file.
