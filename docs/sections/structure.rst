==================
**Code Structure**
==================

Directories
===========

The root (or parent, or main) directory contains the main HELIOS run file ``helios.py``, and the parameter, readme and license files. In addition, the following subdirectories are found in the HELIOS installation:

* ``input``: The standard location for all kinds of input files, like opacity table(s), stellar spectrum file, chemistry files, etc. In principle, as the path of any input files can be set in the parameter file, they don't need to be here though. 

* ``output``: The default output directory. Files magically appear here after a finished (and successful) calculation. The output directory can be changed in the parameter file.

* ``source``: contains the source code files.

* ``docs``: contains the `Sphinx <http://www.sphinx-doc.org/en/master/>`_ files, used to create the documentation (the one you're reading right now). 

* ``ktable``: contains the ktable program and all its files. In theory, this is a separate code from HELIOS, however, it accesses some source files within the HELIOS directory and so the ktable directory should be kept where it is.

Files
=====

In the following each of the files is briefly described. (They are ordered from most to least interesting from a user perspective.)

* ``param.dat``: the main parameter/configuration file. That's the file altered on a daily basis. It can be renamed and, if renamed, included via command-line option '-parameter_file'. See :doc:`parameters` for more info.

* ``planet_database.py``: It is called planet database. It is a database of planets. It stores parameters. Of planets. Feel free to add more planets.

* ``read.py``: responsible for reading data. Check here if you get "reading" errors, if you want to modify the format of read files or would like to include more functions to read stuff.

* ``write.py``: responsible for writing the code output. Check here if you get "writing" errors, if you want to modify the written files or would like to write more output.

* ``realtime_plotting.py``: contains the matplotlib script for the realtime plotting. Alter this if you don't like the aesthetics or would like to alter the quantities plotted.

* ``helios.py``: main run file. It calls the other files and run through the chronological workflow. Explore this file if you would like to understand how HELIOS works on the most top level.

* ``host_functions.py``: contains the functions and short scripts executed on the CPU (aka host). If you want to include a short feature, which is not computation-heavy, you probably want to include it here.

* ``quantities.py``: contains all scalar variables and arrays. It is responsible for data management, like copying arrays between the host and the device (GPU), and allocating memory. 

* ``computation.py``: calls and co-ordinates the device kernels, i.e., functions living on the GPU. If you write a new GPU functionality (=kernel) include it here.

* ``kernels.cu``: contains the detailed computations, executed on the GPU/device. Write new kernel functions or alter existing ones here.

* ``clouds.py``: runs the cloud pre-processing, like converting Mie files to absorption and scattering coefficients and creating the cloud deck(s).

* ``species_database.py``: stores the FastChem names and weights of the most common molecules. Feel free to add more species.

* ``tools.py``: includes some neat helper functions that are generally useful.

* ``additional_heating.py``: reads and includes the additional heating terms from a file.

* ``phys_const.py``: contains the physical constants. It purely exists to convert long names to shorter ones.

The ktable program files are explained in :ref:`ktable-code-structure`.