Structure & I/O
===============

Directories
-----------

The root (or parent, or main) directory contains the main HELIOS run file, and the parameter, readme and license files. In addition, the following subdirectories are found in the HELIOS installation:

* ``root``: contains the main HELIOS, readme, and license files.

* ``input``: usually contains the input files, like the opacity table, planet data, stellar spectrum, etc. In principle, as the path of any input file can be set in the parameter file, they don't need to be here though. 

* ``output``: files magically appear here after a finished calculation.

* ``source``: contains the source code files.

* ``docs``: contains the `Sphinx <http://www.sphinx-doc.org/en/master/>`_ files, used to create the documentation (the one you're reading right now). 

* ``ktable``: contains the k-table generator program. This directory can be moved (or deleted) in principle, as it works independently of the main HELIOS code. In general, it is recommended to leave the directory structure as it is.

Files
-----

In the following the included files are briefly described. They are ordered from a user perspective from most to least interesting (from a developer perspective the order is probably the other way around).

* ``param.dat``: the main parameter/configuration file. That's the file altered on a daily basis. See the :doc:`parameters` for more info.

* ``read.py``: contains the class and the methods responsible for reading data. Check here if you get "reading" errors.

* ``write.py``: contains the class and the methods responsible for reading data. Check here if you get "writing" errors or if you want to modify the written files.

* ``realtime_plotting.py``: contains the matplotlib script for the realtime plotting. Alter this if you don't like the aesthetics.

* ``helios.py``: main run file. It calls the other files and exhibits the chronological workflow. This is the conductor of HELIOS.

* ``host_functions.py``: contains the functions and short scripts executed on the CPU (aka host). If you want to include a short feature, which is not computation-heavy, you probably want to include it here.

* ``quantities.py``: contains all scalar variables and arrays. It is responsible for data management, like copying arrays between the host and the device (GPU), and allocating memory. 

* ``computation.py``: calls and co-ordinates the device kernels, i.e., functions living on the GPU. This is the brain of HELIOS.

* ``kernels.cu``: contains the detailed computations, executed on the device. This is the workhorse of HELIOS.

* ``phys_const.py``: contains the physical constants. It purely exists to convert long names to shorter ones.

The k-table generator files are explained in :doc:`ktable`.

Mandatory Input
---------------

The following input is required to be present and needs to be in the correct format.

opacity table
^^^^^^^^^^^^^

The opacity table is best produced with the k-table generator program, using the output of HELIOS-K. If you want to use your own opacity table, the following format needs to met. First of all, it needs to be in HDF5 format. See `h5py <http://www.h5py.org/>`_ or `HDF5 <https://www.hdfgroup.org/>`_ for more info on this format and how to use it.

The opacity table may come in two versions. For the temperature iteration, the k-distribution method. The integration over one wavelength bin is performed via Gaussian quadrature. For pure-postprocessing an opacity table is used in pure opacity sampling format, i.e., one opacity value per wavelength. 

----

In both cases, the opacity table has to come as HDF5 file with the following datasets::

   pressures
   temperatures
   weighted Rayleigh cross-sections
   meanmolmass

The first two sets list the PT-grid on which the opacities are pre-calculated. Both the pressure and temperature values need to be in ascending order. The pressures need to be uniformly spaced in log10 and the temperatures linearly. 

The ``weighted Rayleigh cross-sections`` set gives the weighted scattering cross-sections as function of wavelength, pressure and temperature in the following format:

cross-sect[wavelength, press, temp] = cross-sect[x + n_x*p + n_x*n_p*t], 

where n_* is the length of the according list and x, p, t are the indices in the according list. For instance, temp = temperatures[t], wavelength = wavelengths[x] and press = pressures[p]. They can be a combination of variation of different cross-section sources, but they need to be weighted by their respective volume mixing ratio.

The ``meanmolmass`` set lists the mean molecular mass of the gas particles as function of pressure and temperature as:

meanmolmass[press, temp] = meanmolmass[p + n_p*t], 

with the same denomination as above.

----

If opacity sampling is used, the following entry needs to be present as well::

   wavelengths
   kpoints

The ``wavelengths`` set lists the wavelengths used for the opacity calculation in ascending order. Those are also the wavelengths used for the radiative transfer calculation.

``kpoints`` lists the opacities as a function of wavelength, pressure and temperature. The same format as for ``weighted Rayleigh cross-sections`` is used.

----

If the k-distribution method is used, these datasets are required::

   center wavelengths
   interface wavelengths
   wavelength width of bins
   ypoints
   kpoints

The ``center wavelengths`` set lists the central wavelength values for the wavelength bins in ascending order. These values are only used for plotting reasons and are not used in the radiative transfer calculation.

The ``interface wavelengths`` set lists the interface wavelength values between the wavelength bins in ascending order.

The ``wavelength width of bins`` set lists the width of the wavelength bins in ascending order.

The ``ypoints`` set lists the abscissa point values for the Gaussian quadrature integration. The standard approach is to use 20th order Gaussian quadrature. Note, that those values need to be rescaled to lie within [0,1].

The ``kpoints set`` lists the opacities as a function of y-point, wavelength, pressure and temperature. The format is analogously to before,

kpoints[y-point, wavelength, press, temp] = kpoints[y + n_y*l + n_y*n_l*p + n_y*n_l*n_p*t], 

where n_* is the length of the according list and y, l, p, t are the indices in the according lists, e.g. y-point = ypoints[y], etc.


Optional Input
--------------

The following input is optional, and only needed when certain options are set in the :doc:`parameters`.

temperature profile
^^^^^^^^^^^^^^^^^^^

An iterative run of HELIOS does not require an input temperature profile as it will iterate and find the radiative-convective solution. However, in the case of pure post-processing a given temperature profile is used to create the corresponding emission spectrum. The temperature file should be in ASCII form, with the temperatures and pressures in the first and second column. Both formats "TP" or "PT" can be set in the :doc:`parameters`. Usually the pressure is assumed to be in cgs units. Should the pressure be in bar, an additional "bar" needs to be written after "TP" or "PT".
The standard output of HELIOS may also be used as an input profile. 

Practically, the read-in temperature profile is linearly interpolated to the HELIOS pressure grid, set by the top and bottom of atmosphere pressures and the number of layers.

adiabatic coefficient
^^^^^^^^^^^^^^^^^^^^^

To enable the convective adjustment, the adiabatic coefficient as function of temperature and pressure needs to be known. In the simplest case, a constant value for the the adiabatic coefficient can be set manually.

If a file is to be read in, it should be in ASCII format, with the adiabatic coefficient listed as function of pressure and temperature, with log10 temperature being on the smaller loop and log10 pressure on the larger one, i.e. from top to down we get kappa[t+n_t*p], with the pressure index p and the temperature index t, and n_t the number of temperature values. If the corresponding entropy is listed as well, its layer values will be given out as output as well.

The format of the file should be:

.. figure:: ../figures/adiabat.png
   :scale: 60 %
   :alt: map to buried treasure

   *Figure: Format of the adiabatic coefficient/entropy file.*

planet parameters
^^^^^^^^^^^^^^^^^

The planetary, stellar and orbital parameters may be pre-tabulated for convenience. The format of this ASCII file should be:

.. figure:: ../figures/planets.png
   :scale: 60 %
   :alt: map to buried treasure

   *Figure: Format of the planet file.*

The name in the first column can then be used in the :doc:`parameters` making the corresponding values to be read automatically. The surface gravity can be given either in (dex cgs) or in (cgs) units. See the figure for the correct units of the other parameters.

A sample planet file is provided with the installation. No guarantee is made about the correctness of the data within.

If no planet file can be bypassed by setting the planetary parameters manually in the :doc:`parameters`.

stellar spectrum
^^^^^^^^^^^^^^^^

In addition to using the blackbody with the stellar temperature for the external irradiation, one can read in a stellar spectrum. The spectrum has to exhibit the same wavelength grid as the opacities. The spectral flux needs to come with an HDF5 file in cgs units of erg s :math:`^{-1}` cm :math:`^{-3}`. 

A sample file is provided with the installation. It contains the spectrum of HD 189733 downloaded from the `PHOENIX online database <http://phoenix.astro.physik.uni-goettingen.de/>`_, once in original resolution and once downsampled to 300 wavelength bins with the corresponding wavelength values.

VULCAN mixing ratios
^^^^^^^^^^^^^^^^^^^^

There are ongoing tests to couple HELIOS with the chemical kinetics code `VULCAN <https:github.com/exoclime/vulcan>`_. To this end, the calculated chemical abundances obtained with VULCAN are used for each radiative transfer run. 

*This feature is still in development. Thank you for your patience.*

Command Line Options
--------------------

In addition to the parameter file, the most important parameters can also be set as command line options or console arguments. These options are ::

	-name: 			name of output
	-outputdir:		root output directory
        -isothermal: 		isothermal layers?
        -postprocess: 		pure post-processing?
        -nlayers: 		number of layers in the grid
        -plot: 			realtime plotting?
        -ptoa: 			pressure at the TOA
        -pboa: 			pressure at the BOA
        -temperaturepath: 	path to the temperature file
        -opacitypath: 		path to the opacity table file
        -energycorrection: 	include correction for global incoming energy?
        -tintern: 		internal flux temperature [K]
        -angle: 		zenith angle measured from the vertical [deg]
        -planet: 		name of the planet (manual or entry in planet data file)
	-g: 			surface gravity [cm s^-2]
        -a: 			orbital distance [AU]
        -rstar: 		stellar radius [R_sun]
        -tstar: 		stellar temperature [K]
        -f: 			f heat redistribution factor
        -star: 			spectral model of the star
        -kappa: 		adiabatic coefficient, kappa = (ln T / ln P)_S
	-Vfile: 		path to the file with VULCAN mixing ratios
	-Viter: 		VULCAN coupling iteration step nr.
	-tau_lw: 		longwave optical depth (used for f approximation)


Output
------

The output files should be self-explanatory. If not, additional information will be given here (later).
