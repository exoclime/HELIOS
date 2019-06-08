k-Table Generator
=================

This script called the "k-Table Generator" is a small tool to convert the HELIOS-K opacity output into opacity tables, which can be read in by HELIOS. 

To combine the individual opacities it further requires access to chemical abundances given by FastChem. Pre-generated HELIOS-K opacities may also be found `here <https://chaldene.unibe.ch/>`_.

Getting Started
---------------

The main program is run by typing:: 

	python3 ktable.dat

in the main directory. Additional source files (the "code") are located in the source subdirectory, and some input files in the input directory. The output may come conveniently to the output directory. However, every path can modified in the parameter file ``param_ktable.dat``. The parameters are described further below.

The opacity table generation comes in *two stages*. In the first stage, opacity files for the individual absorbers are created from the HELIOS-K output. In the second stage, the individual opacities are weighted by the respective molecule's equilibrium mixing ratio, determined by FastChem, and combined to the final mixed opacity table. This final product is then used in HELIOS.

Parameter File
--------------

The parameter file first contains the parameters setting the correct format of the opacities.

``format (ktable, sampling)``

This determines whether k-distribution tables (ktable) or tables with per-wavelength opacities (sampling) shall be created during the first stage of the process.

``individual species calculation (yes, no)``

Sets whether the files for individual species shall be calculated (aka the first stage of the process). If starting from the HELIOS-K output, this should be set to "yes". If the individual files have already been generated, this step can be skipped with a "no".

``path to HELIOS-K output``

Sets the path to the HELIOS-K output directory. The individual "cbin" files can be in subdirectories. This parameter is **only relevant in the "ktable" setting**.

``path to sampling param file``

Sets the path to the files which lists all the species and their respective paths for the sampling calculation. 

``sampling wavelength grid (R=3000, 8k)``

Sets the wavelength grid to be used for the opacity downsampling process. At the moment only two choices exist. "R=3000" creates a wavelength grid with a constant resolution of 3000 across the whole range. This grid has around 13'000 wavelength points between 0.34 micron and 30 micron. In contrast "8k" creates a wavelength grid with a focus on the near-infrared regime. It contains around 8000 points and ranges from 0.33 micron to 200 micron.  It is recommed to use the "R=3000" setting. Other wavelength grids can be defined in the ``build_opac_sampling.py`` if desired.

``path to sampling output``

This sets the path to the individual opacity tables, created in stage one.

``path to species file``

Sets the file containing the species included in the final mixed table.

``path to FastChem output``

Sets the path to the FastChem output. In the current version the output needs to come in two files named ``chem_high.dat`` and ``chem_low.dat``, containing the chemical abundance for a temperature pressure grid, whereas the first file contains the higher temperature regime and the second one the lower temperatures. Preferably, the grid structure [p + n_p * t] is used. (This will be made more user-friendly in a future update.)

As reference, I use a grid in P=[1e-6,1e3,Delta_P=1/3dex] and T=[100, 6000, Delta_T=100].

``path to final output (mixed ktable)``

Sets the path to the final mixed opacity table and corresponding temporary and info files.

Note: Many temporary files are being generated to save intermediate calculations, e.g., the weighted opacities weighted by the mixing ratios. This allows for a large speed-up the next time the same species and abundances are employed.

Input Files
-----------

``species_for_sampling.dat``

This lists the species to be used in the sampling calculation. Depending on their location either Opacity2 or Opacity3 will determine the correct file format by the program. The first column is the name of the species. It can be chosen theoretically at will, but it is recommended to chose the usual chemical notation.

``species_for_final_output.dat``

This lists the species to be included in the final mixed opacity table. Their name must be equal to the name chosen for the individual sampling. The FastChem name further allows the program to find the correct abundances in the FastChem output. Finally the program needs the mass of the species to convert from volume to mass mixing ratios.

Reference input files are provided with the installation.

ktable vs sampling
------------------

**ktable**

If the first parameter in ``param_ktable.dat`` is set to "ktable", the program will calculate the k-distribution tables for the individual species, using the Chebyshev coefficients from the HELIOS-K output. For this it needs to access the "cbin" files containing the Chebyshev coefficients. It will generature individual opacity containers with 20 Gaussian points per wavelength bin. This is required in HELIOS. The number of wavelength bins is given by the resolution of the "cbin" files. See the HELIOS-K ReadMe for more info on those. The species used in this process are simply are species present in the directory (or subdirectories) set in the parameter file.

.. figure:: ../figures/cbin_files.png
   :scale: 60 %
   :alt: map to buried treasure

   *Figure: cbin files produced by HELIOS-K.*

**sampling**

If set to "sampling", the program will sample the opacity functions from HELIOS-K for individual species at the the wavelength grid, as specified in the parameter file. The species to be sampled are set in the ``species_for_sampling.dat`` file. The files should be located in the HELIOS-K output directories, e.g. as given the Opacity2 main directory. 

.. figure:: ../figures/Opacity2.png
   :scale: 60 %
   :alt: map to buried treasure

   *Figure: Directory with calculated opacities by HELIOS-K. This is a good input for the opacity table generation using the "sampling" method.*