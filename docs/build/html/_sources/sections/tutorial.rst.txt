Tutorial
========


Welcome! I assume this is the first time you are looking at HELIOS. Thus, I will guide you here in the most straightforward way towards the first run of the code on your machine. In order to keep the tutorial short and clear, in-depths explanations are omitted. Please refer to the appropriate sections in the in-depth documentation for more info.

In the following I merely expect that you possess an NVIDIA GPU and are either on a Linux or Mac Os X system (sorry, Windows users. You are on your own).



Step-by-step Installation
-------------------------


1. Install the newest version of the CUDA toolkit from `NVIDIA <https://developer.nvidia.com/cuda-downloads>`_. To ascertain a successful installation, type ``which nvcc``. This should provide you with the location of the nvcc compiler. If not, something went wrong with the installation. 

  Make sure that the library and program paths are exported. On Mac Os x you should have the following entries in your .bash_profile file (shown for version 10.0 of CUDA) ::

	export PATH=/Developer/NVIDIA/CUDA-10.0/bin:$PATH
	export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-10.0/lib:$DYLD_LIBRARY_PATH

  On a Linux machine you should have the following entries ::

	export PATH=/usr/local/cuda-10.0/bin:$PATH
	export DYLD_LIBRARY_PATH=/usr/local/cuda-10.0/lib:$DYLD_LIBRARY_PATH


2. Now we set up Python. Most straightforward is to download and install the Anaconda package as it contains most of the required libraries already. Get it from `here <https://www.anaconda.com/distribution/#download-section>`_. Pick Python version 3.x. Also, the command line installer is better if you are installing it on a remote server.


3. Now we set up a virtual environment for Python. In a terminal or shell window type ::

	conda create -n MyBubble

  where MyBubble is just a naming suggestion. Type ::

	conda activate MyBubble 

  to activate the environment. Then we need to uninstall and reinstall pycuda package (the pre-installated version does not work usually). Type ::

	pip uninstall pycuda

  (type yes, when asked) and then type ::

	pip install pycuda


4. Now we get Helios from its Github repository by typing :: 

	git clone https://github.com/exoclime/HELIOS.git helios

  The last "helios" is the name of the directory to be created. 

  Congratulations, you have successfully installed HELIOS!

First Run
---------

Go to the HELIOS main directory and type:: 

	python3 helios.py

HELIOS is pre-configured to conduct a generic run of the hot Jupiter HD 189733b. If HELIOS ran successfully, you should be now in possession of the dayside temperature profile in radiative-convective equilibrium and the corresponding emission and secondary eclipse spectra of this planet. Feel free to explore the ``output`` directory. 

For a first quick analysis of your output, the installation includes two plotting scripts ``plot_tp.py`` and ``plot_spectrum.py``, located in the ``tools`` subdirectory. Those scripts allow you to inspect your very first generated temperature-profile profile and emission spectrum. Well done! You can now go on and modify the parameters in the ``param.dat`` file or simulate another planet. Please refer to the rest of the documentation for more info.

Sample Files
------------

In the ``input`` subdirectory, three sample files are included to help with the first runs of HELIOS.

``opac_sample.dat``

This is a sample file for the opacity table. It contains pre-calculated opacities for the main typical hot Jupiter (or brown dwarf) absorbers, pre-mixed with solar elemental abundances according to Asplund et al. 2009. The included absorbers are H2O, CO2, CO, CH4, NH3, HCN, C2H2, PH3, H2S and the CIA H2-H2, H2-He opacities. It also contains the Rayleigh cross-sections for H2, H, He and H2O. To keep the size of the table manageable, the opacities are sampled at a resolution of R=50 between 0.3 and 200 micron. 

Note, this table is good for obtaining a *first order estimate* of hot Jupiter (or brown dwarf) atmospheric temperatures and their spectra. For accurate results, a higher resolution with more absorbing species is recommended. See :doc:`ktable` how to generate an own, improved opacity table.

See `h5py <http://www.h5py.org/>`_ or `HDF5 <https://www.hdfgroup.org/>`_ for more info on this format and how to use it.

``stellar_sample.dat``

This is a sample file on how to include a realistic stellar spectrum. As the opacity, this is in HDF5 format. Pre-set is the PHOENIX stellar spectrum of HD 189733. If you wish to employ an own spectrum, create a new data set analogously to the existing one. In order for HELIOS to work, the wavelength grid of the stellar spectrum needs to be consistent with the wavelength grid of the opacities.

``planet_data.dat``

Here you can save planetary parameters for later use, which can be loaded in the parameter files under the entry ``planet``. Two planets are pre-saved, HD 198733b and WASP-43b (no guarantee for the correctness of the parameters). See :doc:`structure`, section "planet parameters" for more info.

