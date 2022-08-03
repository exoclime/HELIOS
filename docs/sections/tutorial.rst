============
**Tutorial**
============

Welcome! I assume this is the first time you are looking at HELIOS. Below, I will guide you through the installation and the first successful run of HELIOS. I am providing some sample input files you can use for your initial attempts with HELIOS. I will then explain how to calculate chemistry files for varying abundances, get more opacities and stellar spectra to be used with HELIOS, how to run HELIOS with mixing opacities on--the--fly, how one can couple to a photochemical kinetics code, include clouds, etc.

Installation
============

Setting Up
----------

In the following I merely expect that you possess an NVIDIA GPU and are either on a Linux or Mac Os X system (sorry, Windows users. You are on your own).


1. Install the newest version of the CUDA toolkit from `NVIDIA <https://developer.nvidia.com/cuda-downloads>`_. To ascertain a successful installation, type ``which nvcc``. This should provide you with the location of the nvcc compiler. If not, something went wrong with the installation. 

Make sure that the library and program paths are exported. On Mac Os x you should have the following entries in your .bash_profile file (shown for version 10.0 of CUDA) ::

	export PATH=/Developer/NVIDIA/CUDA-10.0/bin:$PATH
	export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-10.0/lib:$DYLD_LIBRARY_PATH

On a Linux machine you should have the following entries ::

	export PATH=/usr/local/cuda-10.0/bin:$PATH
	export DYLD_LIBRARY_PATH=/usr/local/cuda-10.0/lib:$DYLD_LIBRARY_PATH


2. Now, let us set up Python. Most straightforward is to download and install the Anaconda package as it contains most of the required libraries already. Get it from `here <https://www.anaconda.com/distribution/#download-section>`_. Pick Python version 3.x. Also, the command line installer is better if you are installing it on a remote server (click 'Get Additional Installers'). Alternatively, all of the Python libraries can be downloaded and installed manually with 'pip' as well. See :doc:`requirements` for more info on that.


3. Now we set up a virtual environment for Python. In a terminal or shell window type ::

	conda create -n MyBubble

  where MyBubble is just a naming suggestion. Type ::

	conda activate MyBubble 

  to activate the environment. Then we need to uninstall and reinstall pycuda package (the pre--installed version does not work usually). Type ::

	pip uninstall pycuda

  (type yes, when asked) and then type ::

	pip install pycuda

Get HELIOS
----------

For the installation of HELIOS there are two choices:

* Either, using the terminal, type::

    git clone https://github.com/exoclime/HELIOS.git "insert_target_dir"

  to clone into the GitHub repository. In addition to downloading the required files, this gives you full Git functionality. In case you don't have Git installed on your machine you can get it e.g. from `here <https://git-scm.com/downloads>`_.

* Or, if you'd like to bypass Git, simply download the `ZIP archive <https://github.com/exoclime/HELIOS/archive/refs/heads/master.zip>`_, which contains all the necessary files. Unpack the ZIP to a local directory and you are ready to go!

Download Input Files
--------------------

Many of the input files for HELIOS are too large to be stored on GitHub. They have to be downloaded from the University of Bern server. They are located `here <https://chaldene.unibe.ch/data/helios/input_files/>`_. For convenience, the installation includes a bash script that automatically downloads all files and puts them where they belong. Just run the included script with::

    bash install_input_files.bash

Once all files are downloaded (may take a while since opacity files are large), you are done. Congratulations, you have successfully installed HELIOS!

Note that those downloaded files are often only example files. If using HELIOS for more than just a handful of initial models, a larger variety of input files will have to be produced. See below sections on how to do that.

Here is a list of of the downloaded input files and what they are good for:

``r50_kdistr_solar_eq.h5``: A pre-mixed opacity table with the most common absorbers for a typical sub-Neptune atmosphere, weighted by solar equilibrium mixing ratios. A resolution of R = 50 and k-distribution format is used. This table can be used to calculate T--P profiles in radiative-convective equilibrium for aforementioned conditions. The opacity file is given in HDF5 format. See `h5py <http://www.h5py.org/>`_ or `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ for more info on this format and how to use it. Type ``h5dump -d included\ molecules r50_kdistr_solar_eq.h5`` to see which species are included in this table. See :doc:`ktable` on how to produce opacity tables.

``delad_example.dat``: Example file from which kappa/delad and c_p can be read and used in HELIOS. If you're curious, this file is made for a 100% H2O atmosphere. See `Using Pre-Tabulated kappa and c_p` for more info.

``star_2022.h5``: Example file that contains a GJ1214 PHOENIX spectrum that can be used in HELIOS. See `Creating a Stellar Spectrum File` on how to add more stellar spectra to this file.

``chemistry``: Directory containing equilibrium chemistry files generated with FastChem for solar, 10x solar and 100x solar elemental abundances based on Lodders 2009. These files can be used with the ktable program or HELIOS in the on--the--fly mixing mode. See `Use Different Chemistry` on how to produce more chemistry files.

``opacity``: Directory that contains opacity files that can be used with HELIOS in the on--the--fly mixing mode. They have the resolution R=50 and k-distribution format. Included are the most relevant absorbers for sub-Neptune atmospheres. More opacity files can be produced with the included ktable program, see :doc:`ktable` for more info.

``cloud_files``: Directory that contains extinction coefficient for a wealth of aerosol types, calculated using the `LX--MIE code <https://github.com/exoclime/LX-MIE>`_. These files can be used when including clouds in HELIOS. See `Including Clouds` for more info.


First Run
=========

Go to the HELIOS main directory and type:: 

	python3 helios.py

HELIOS is pre--configured to execute a generic run of the sub--Neptune GJ 1214b. If HELIOS ran successfully, you should be now in possession of the dayside temperature profile in radiative--convective equilibrium and the corresponding emission and secondary eclipse spectra of this planet. Feel free to explore the ``output`` directory. 

For a first quick analysis of your output, the installation includes two plotting scripts ``plot_tp.py`` and ``plot_spectrum.py``, located in the ``plotting`` subdirectory. Those scripts allow you to inspect your very first generated temperature--profile profile and emission spectrum. 

Well done! You can now go on and modify the parameters in the ``param.dat`` file or simulate another planet or model other atmospheric conditions, etc. The doors to the world of 1D RT modeling are open!


Use Different Chemistry
=======================

New Method
----------

To run a model with different chemical abundances, FastChem is the preferred choice to generate the chemistry files needed to combine molecular opacities and construct the final opacity table. It is **recommended** to install `FastChem <https://github.com/exoclime/FastChem>`_. How to install and run FastChem is described in ``manual/fastchem_manual.pdf`` in the FastChem repository. 

Back in the HELIOS repository, in ``input/chemistry`` there are subdirectories that contain FastChem output files for a number of different solar metallicities. These files can be used to generate opacity tables or mix opacities on--the--fly with HELIOS and so their format acts as reference. These files can be generated with FastChem quite straightforwardly. The P--T grid file that I use to produce the chemistry files is provided in ``input/chemistry/fastchem_input/`` and named ``pt.dat``. A different P--T grid can be used too, provided it is over a large temperature and pressure range with sufficiently small step sizes. Step in temperature has to be linear and step in pressure has to be constant in logarithm--space.

Going back to FastChem, in the ``config.input`` file:

- As Atmospheric profile, pick the ``pt.dat`` file.
- Name the abundance output file ``chem.dat``. Make sure the output directory exists or FastChem will simply not produce the file. It is also useful to produce a monitor file and inspect it to see whether everything went OK. (There should be no 'fail' entry.)
- Set the output to mixing ratios by setting the last option to 'MR'.

In the ``parameters.dat`` file:

- Set the correct elemental abundance file.
- The species data file should be ``logK.dat``.

Finally, we need to conduct two tiny source code tweaks. First, in ``model_src/model_main.cpp``, check that line 91 is active and line 92 is commented out. If not, make it so. We need the order to be pressure first, temperature second (because this is the format of ``pt.dat``). Second, in ``model_src/save_output.h`` between lines 49 and 53, remove **all** spaces in **all** of the header names. For example '#P (bar)' should become '#P(bar)'. (Otherwise Python's 'numpy.genfromtxt' function will complain that the number of headers is not equal to the number of columns.) After modifying the source code, **FastChem needs to be compiled again.**

With these settings FastChem should now generate the chemistry files exactly in the format that HELIOS and the ktable program require.


Old Method (still compatible)
-----------------------------

**Choose this method if including ions leads to issues for very low temperatures.**

Previously, two chemistry files were needed because at low temperatures FastChem (at least in earlier versions) had to be run without ions, but at high temperatures ions are obviously important. That means for each elemental composition, FastChem had to be run twice, each using a different P--T grid, once without and once with ions. The two P--T grid files for that are provided in ``input/chemistry/fastchem_input/`` too and named ``pt_low.dat`` and ``pt_high.dat``. In this case, in the ``config.input`` file:

- As Atmospheric profile file pick the ``pt_low.dat`` or ``pt_high.dat``, respectively.
- Name the abundance output file ``chem_low.dat`` or ``chem_high.dat``, respectively. Make sure the output directory exists or FastChem will simply not produce the files. It is also useful to produce the monitor files and inspect them to see whether everything went OK. (There should be no 'fail' entry.)
- Set the output to mixing ratios by setting the last option to 'MR'

In the ``parameters.dat`` file:

- Set the correct elemental abundance file
- The species data file should be ``logK_wo_ions.dat`` for the lower temperature grid and ``logK.dat`` for the higher temperature grid.

Finally, we need to conduct two tiny source code tweaks. First, in ``model_src/model_main.cpp``, check that line 91 is active and line 92 is commented out. If not, make it so. We need the order to be pressure first, temperature second (because this is the format of ``pt.dat``). Second, in ``model_src/save_output.h`` between lines 49 and 53, remove **all** spaces in **all** of the header names. For example '#P (bar)' should become '#P(bar)'. (Otherwise Python's 'numpy.genfromtxt' function will complain that the number of headers is not equal to the number of columns.) After modifying the source code, **FastChem needs to be compiled again.**

With these settings FastChem should now generate the chemistry files exactly in the format that HELIOS and the ktable program require.


Include More Opacities
======================

Almost all of the opacities that HELIOS uses have been calculated with `HELIOS-K <https://github.com/exoclime/helios-k>`_. However, it is **not necessary** to install HELIOS-K in order to include more opacities in HELIOS. A vast number of pre--calculated opacities can be found in the `online opacity database <https://dace.unige.ch/opacityDatabase/>`_ at the University of Geneva.

- For a given species, download the files for the whole temperature and pressure ranges provided. 
- Put the files of each species in a separate directory on your local machine. 
- Include the paths to these directories in the ktable program parameter file (per default ``param_ktable.dat``) and the ktable program will take over from here. See :doc:`ktable` for more info on how to proceed from here.

If you don't find the species you need in the online database, or because you'd like to vary some opacity parameters, then you will probably have to run HELIOS--K yourself (or have a student do it for you). How to install and run HELIOS--K is described in the `online documentation <https://helios-k.readthedocs.io/en/latest/>`_ file. 

If running HELIOS--K yourself, note that the ktable program that generates the opacity table for HELIOS needs the HELIOS--K output files to have **exactly the same format** (i.e., file names and contents) as the files found in the `online database <https://dace.unige.ch/opacityDatabase/>`_. Only leeway is that the files can be in ASCII instead of binary format. In this case the file should have two columns, with the wavenumber listed in the first one (in ascending order) and the opacity in the second one.

Creating a Stellar Spectrum File
================================

In addition to using the blackbody with the stellar temperature for the external irradiation, one can read in a stellar spectrum. General requirements are: (i) The spectrum has to be on the same wavelength grid as the opacities. (ii) Spectrum has to be provided as flux measured on the stellar surface in cgs units, i.e., erg s :math:`^{-1}` cm :math:`^{-3}`. (iii) The spectrum has to be read from an HDF5 file.

A reference file is provided with the installation. It contains the spectrum of GJ 1214 downloaded from the `PHOENIX online database <http://phoenix.astro.physik.uni-goettingen.de/>`_ interpolated to the stellar parameters given in `Harpsoe et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013A%2526A...549A..10H/>`_ and sampled to R=50, same as the opacity files provided.

In the directory ``star_tool`` a script is included that allows for the generation of more stellar spectra to be used in HELIOS. To run the script, go to ``star_tool`` and type::

   python run.py

Everything is set up in the way to produce the ``star_2022.h5`` that comes with the installation. I use the naming structure /“wavelength_grid”/“database”/“star”, visible by typing::

   hfls -r star_2022.h5

All parameters and settings are found in ``run.py`` (it basically acts as the 'config' as well as 'run' file). 

In the top of ``run.py``, one creates a star with the desired properties and in the bottom one runs the script, setting the star as first parameter.

The further parameters of main_loop are:

   ``convert_to [data set name]``

This sets the data set name within the HDF5 file. I usually name the data set after the resolution I use, but this is not necessary.

   ``opac_file_for_lambdagrid [path to opacity file]``

Access to an opacity file is needed which will be used to read the wavelength grid (in order to guarantee that the opacity and the star are on the same grid). It does not matter what kind of opacity file it is. It just needs to exhibit the correct wavelength grid.

   ``output_file [file name]``

Name of the output file. It will appear in the ``output`` subdirectory. If the file already exists, a the new spectrum will be included in this file.

   ``plot_and_tweak [automatic, yes, no]``

Here one sets how the spectrum should be extrapolated if necessary. The spectrum is extrapolated with a blackbody (BB) and the question is which temperature should be used for that. It is recommended to choose the 'automatic' feature, which will find the best extrapolation by applying the Newton--Raphson method. A pop--up window will open showing the spectrum with the extrapolation and ask whether one accepts it. If not, another temperature can be chosen by hand. Similarly, if one sets 'yes', one can choose the extrapolation temperature manually by looking at the pop--up window and choosing the best fitting value by eye. If one sets 'no', one has to define a BB temperature later with the 'BB_temp' option.

   ``save_ascii [yes, no]``

This sets whether the spectrum is written to an ASCII (text) file. This is not exclusive to saving to an HDF5 file.

   ``save_in_hdf5 [yes, no]``

This sets whether to write the spectrum to the HDF5 file given earlier. If the file does not exist yet, it is created.

Note there is a final optional paramater called ``BB_temp``. If earlier one sets ``plot_and_tweak=no``, the BB temperature can be defined here directly.

There are three options on how to include a spectrum from an online library. For PHOENIX models, the script downloads all files automatically from the `Göttingen library <https://phoenix.astro.physik.uni-goettingen.de/?page_id=15>`_ and interpolates to the given stellar parameters. For MUSCLES one has to download the files manually from `their page <https://archive.stsci.edu/prepds/muscles/>`_ and convert wavelengths and fluxes to cgs units manually via included parameters. Lastly, an ASCII file can be used as well. Examples for each case are given in the top of ``run.py``.

The file “functions.py” contains the function definitions if one would like to modify the script.

.. _kappa-file-format:

Using Pre-Tabulated kappa and c_p
=================================

Instead of using a constant value for the adiabatic coefficient, kappa (or delad), it is also possible to read pre-tabulated values from a text file. With the HELIOS installation the example file ``input/delad_example.dat`` is included to show the necessary format of the file. This file format is hard-coded and a **user--supplied file needs to match exactly the example file** including providing the heat capacity c_p column because the c_p is needed by the convective adjustment scheme. The entropy *can* be provided as well but does not have to be, because it is not used in the RT calculation. The temperature and pressure grids may be different from the example file, but the **temperature steps have to be constant in linear space and the pressure steps constant in log space**. Note that the c_p is per unit mole and the entropy (if given) is per unit gram!

Using Vertical Chemistry Profiles or On--the--fly Opacity Mixing
================================================================

Instead of using a single, premixed opacity table it is with version 3 now possible to mix the individual gaseous opacities on the fly. The advantage of that is twofold. First, the mixing can be made more accurate by using random overlap (RO) instead of assuming perfect correlation between species absorption bands. (In theory one could also use RO for premixed tables as well -- that is a missing functionality in the ktable program that perhaps an avid user can implement in the future). Second, instead of being limited to equilibrium abundances, one can read in vertical chemical profiles and by doing so post--process non--equilibrium chemistry models with HELIOS, or even couple the RT with the chemistry self--consistently (see next section for the latter approach).

To use on--the--fly opacity mixing, one needs to provide 

   (i) chemical abundances for all included species

   (ii) opacities for all included species

The chemical abundances can come (a) from a file listing the vertical mixing ratios, or (b) equilibrium abundances in FastChem format, or (c) a constant mixing ratio can be set as well. The chemistry source has to be specified for each included species in the species file. There is a reference species file called ``species.dat`` included in the ``input`` directory. Equilibrium abundances are given by the same FastChem output files as used by the ktable program and explained in `Use Different Chemistry`_. The molecular opacity files are simply the individual opacity files that are produced as by--product when running the ktable program and constructing the premixed table. 

Note that **each species in the species file has to exist in the species database** ``source/species_database.py`` because the properties are pulled from there. Most of the common species should already be pre--defined. If an error is returned that there is no such entry in ``species_database.py`` a new one has to be manually created. When creating a new entry just follow the format of the existing ones. The FastChem name can be looked up in the FastChem output file. The weight parameter is the species' molecular weight in AMU (or the molar weight in g). For CIA pairs, it is the weight of the secondly--listed molecule.

In the parameter file (default ``param.dat``) one then sets the according parameters in the 'Opacity Mixing' section. See :doc:`parameters` for more info on these parameters.

Scattering, Ions and CIA
------------------------

Analogously to the molecular opacities, the scattering cross--sections are pre--calculated with the ktable program. It is possible to include Rayleigh scattering cross--sections for H2, He, H, H2O, CO, CO2, O2, N2 and e-- Thompson scattering. See :ref:`final-species-file` for further info. Apart from H2O the scattering cross--sections are provided by the file ``scat_cross_sections.h5`` that is generated with the ktable program. It is included with the HELIOS installation for R=50 (in ``input/r50_kdistr`` directory), but has to be generated with the ktable program for other resolutions. Including H2O Rayleigh scattering is special, because its cross--section is dependent on the atmospheric H2O abundance. It is thus not provided via file but directly calculated during the HELIOS run.

The H-- opacity is divided in bound--free and free--free contributions, each provided in a separate file. For He-- only the free--free opacity can be included. These files are generated with the ktable program. Currently, using constant mixing ratios for H-- and He-- opacity is not possible (this mixing ratio can only be supplied via file or FastChem).

Collision--induced absorption (CIA) is listed separately in the species file as CIA_H2H2 or CIA_CO2CO2, etc., and provided by own files too. When using constant mixing ratio for the CIA pairs, the mixing ratio for each pair has to be included, separated by a '&'.

Coupling to Photochemical Kinetics
==================================

HELIOS offers the interface for *sequential coupling* to a chemistry code. This means that HELIOS and the chemistry code alternately run in sequence until a converged solution in terms of radiative transfer and chemical composition is found. A standard procedure looks as follows. HELIOS is run first using equilibrium chemistry until an equilibrium T--P profile is obtained. This profile is inserted in the chemistry code which then calculates the corresponding vertical chemical abundances including the desired non--equilibrium processes. The vertical abundances are then used as input for HELIOS which in turn calculates the equilibrium T--P profile. This cycle continues until convergence is found. Currently the convergence is tested in HELIOS by comparing the last two T--P profiles and convergence is proclaimed if the relative difference in temperature in each atmospheric layer is < 1e--4 (i.e. difference < 0.1 K for T = 1000 K). This convergence limit can be changed in the HELIOS parameter file.

A template bash script for the coupling called ``coupling_template.bash`` is included in the HELIOS installation. The relevant commands for the chemistry code are included as comments and just need to be replaced with the *real deal*. Obviously, the HELIOS part of the script can and probably has to be adjusted to one own's needs. See also the 'Photochemical Kinetics Coupling' Section in the parameter file for further options.

Including Clouds
================

HELIOS offers the option to include multiple cloud decks. If setting the vertical cloud distribution manually, each cloud can be parameterized by the pressure at the bottom of the cloud, the volume mixing ratio at that location and and the cloud--to--gas scale height. This gives the cloud a simple shape that exponentially decays with altitude -- faster, slower or as much as the surrounding gas. Alternatively, the vertical cloud distributions can be read from a file. See section 'clouds' in the parameter file for more info on the settings.

The extinction coefficients for a large number of aerosol types are included with the HELIOS installation. These have been calculated with the `LX--MIE code <https://github.com/exoclime/LX-MIE>`_, and the Mie data provided therein. Other cloud files can be calculated, but the format of the files needs to be the same. Also, each file corresponds to a certain particle radius and currently these radii **are hardcoded**. That means when including other clouds the supplied files also have to be for radii from --2 to 3 with 0.1 stepsize in log10(r[micron]) space.
