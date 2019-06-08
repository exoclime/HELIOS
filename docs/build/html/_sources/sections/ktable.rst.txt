k-Table Generator
=================

This script called the "k-Table Generator" is a small tool to convert the HELIOS-K opacity output into opacity tables, which can be read in by HELIOS. 

To combine the individual opacities it further requires access to chemical abundances given by FastChem. Pre-generated HELIOS-K opacities may also be found `here <https://chaldene.unibe.ch/>`_.


Getting Started
---------------

The main program is run by typing:: 

	python3 ktable.dat

in the main directory. Additional source files (the "code") are located in the source subdirectory, and additional input files in the input directory. The output goes per default to the output subdirectory. Note, every path can modified in the parameter file ``param_ktable.dat``. Further parameters are described below.

The opacity table generation comes in *two stages*. In the first stage, opacity files for each molecule are individually created from the HELIOS-K output. In the second stage, the individual opacities are weighted by the respective molecule's equilibrium mixing ratio, determined by FastChem, and combined to the final mixed opacity table. This final pre-mixed opacity table is then used in HELIOS.


Parameter File and Input
------------------------

The parameter file ``param_ktable.dat`` contains the parameters for the correct reading and writing of the opacities. The parameters are as follows.

``format (ktable, sampling)``

This determines the type of opacity tables at hand. Either k-distribution tables (ktable) or opacity sampling tables (=simple per-wavelength opacities) are possible.

``individual species calculation (yes, no)``

Sets whether the files for individual molecules should be calculated (aka the first stage of the process). If starting from the HELIOS-K output, this should be set to "yes". If the individual files have already been generated previously and you simply want to combine the gases differently (e.g., different chemical abundances), set this to "no".

``path to HELIOS-K output``

Sets the path to the HELIOS-K output directory. The individual "cbin" files can be in subdirectories. This parameter is **only considered in the "ktable" setting**.

``path to sampling param file``

Sets the path to the file which lists all the species and their respective paths for the sampling calculation. This is usually the HELIOS-K output directory. Depending on their location, having either "Opacity2" or "Opacity3" in the path name will determine the correct file format to be read in by the program. The first column is the name of the species. It can be chosen theoretically at will, but it is recommended to choose the usual chemical notation.

``sampling wavelength grid (resolution, limits in micron [lower, upper])``

Sets the wavelength grid to be used for the opacity sampling process. Three numbers specify the sampling process. The first number sets the sampling resolution in wavelength. The second and third numbers give the lower and upper wavelength limits in micron, respectively. It is recommended to use something around "R=3000" for post-processing (spectrum generation) purposes.

``path to sampling output``

Sets the path to the individual opacity files. Those files generated in stage 1 will appear in this directory.

``path to species file``

Sets the path to the file, which contains all the species to be included in the final mixed table. Their name must be equal to the name chosen for the individual sampling. The FastChem name further allows the program to find the correct abundances in the FastChem output. Finally the program needs the mass of the species to convert from volume to mass mixing ratios.

``path to FastChem output``

Sets the path to the FastChem output. In the current version the output needs to come in two files named ``chem_high.dat`` and ``chem_low.dat``, containing the chemical abundance for a temperature pressure grid, whereas the first file contains the higher temperature regime and the second one the lower temperatures. The grid structure [p + n_p * t] is used. Check out the example file included in the input/chemistry subdirectory for reference. (This will be made more user-friendly in a future update.)

As example, I used a grid in P=[1e-6,1e3,Delta_P=1/3dex] and T=[100, 6000, Delta_T=100] for my own calculations.

``path to condensation curves``

Sets the path to condensation curve data. Some species are removed from the gas phase if below their condensation (stability) curves. Those species are TiO, VO, SiO (due to MgSiO3), Na (due to Na2S) and K (due to KCl). The species removal procedure is described in detail in App. B of `Malik et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019AJ....157..170M/>`_. Pre-calculated reference for solar abundances are provided with the installation.

``path to final output (mixed table)``

Sets the path to the final mixed opacity table and corresponding temporary and info files.

Note: Many temporary files are being generated to save intermediate calculations, e.g., the weighted opacities weighted by the mixing ratios. This allows for a large speed-up the next time the same species and abundances are employed. These temporary files can be safely deleted, should they clog up too much storage space.


Sample Files for Reference
--------------------------

The installation comes with reference examples for all the required input files.


ktable vs sampling
------------------

**ktable**

If the first parameter in ``param_ktable.dat`` is set to "ktable", the program will calculate the k-distribution tables for the individual species, using the Chebyshev coefficients from the HELIOS-K output. For this it needs to access the "cbin" files containing the Chebyshev coefficients. It will generature individual opacity containers with 20 Gaussian points per wavelength bin. This is required in HELIOS. The number of wavelength bins is given by the resolution of the "cbin" files. See the HELIOS-K ReadMe for more info on those. The species considered in this process are simply the species present in the directory (or subdirectories) set in the parameter file.

.. figure:: ../figures/cbin_files.png
   :scale: 60 %
   :alt: map to buried treasure

   *Figure: cbin files produced by HELIOS-K.*

**sampling**

If the first parameter in ``param_ktable.dat`` is set to "sampling", the program will sample the opacity output from HELIOS-K at the wavelength grid, as specified in the parameter file and convert it into individual molecular opacity files. The species to be sampled are set in the "sampling param file" (see above). The files should be located in the HELIOS-K output directories, e.g. as given in the Opacity2 or Opacity3 directories on the University of Bern server.

.. figure:: ../figures/Opacity2.png
   :scale: 60 %
   :alt: map to buried treasure

   *Figure: Directory with calculated opacities by HELIOS-K. This is a good input for the opacity table generation using the "sampling" method.*