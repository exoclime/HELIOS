================
**Requirements**
================

HELIOS was developed on a machine with Mac OS X 10.9 -- 11.6 and also successfully tested on Ubuntu and Archlinux. It has not been tested to run on other operating systems (like Windows), but should in principle work if the following requirements are met.

Hardware
========

HELIOS is a GPU-accelerated software developed with parts written in CUDA. It thus requires an NVIDIA graphics card (GPU) to operate on. Any GeForce or Tesla card manufactured since 2013 and with 2 GB VRAM or more should suffice to run standard applications of HELIOS.

CUDA
====

CUDA is the NVIDIA API responsible for the communication between the graphics card (aka device) and the CPU (aka host). The software package consists of the core libraries, development utilities and the NVCC compiler to interpret C/C++ code. The CUDA toolkit can be downloaded from `here <https://developer.nvidia.com/cuda-downloads>`_.

HELIOS has been tested with CUDA versions 7.x -- 11.x and should, in principle, also be compatible with any newer version.

Python
======

HELIOS's computational core is written in CUDA C++, but the user shell comes in Python modular format. To communicate between the host and the device the PyCUDA wrapper is used.

The following Python packages are required to run HELIOS.

* numpy
* scipy
* astropy
* matplotlib
* h5py
* PyCUDA
* wget (to prepare stellar spectra yourself)

Some of them may be already included in the python distribution (e.g., Anaconda). Otherwise they can be installed with the Python package manager pip. To install, e.g., PyCUDA type::

   pip install pycuda

This may fail, if you don't have admin permissions on the machine you are trying to install software. For this case, so-called *virtual environments* exist which embed your Python installation in a user-manageable 'bubble'. See, e.g., `here <https://docs.python.org/3/tutorial/venv.html>`_, `here <https://realpython.com/python-virtual-environments-a-primer/>`__ or `there <https://docs.python-guide.org/dev/virtualenvs/>`_ for tutorials on virtual environments.

Note that HELIOS has been tested with Python versions 3.5.x -- 3.8.x. It should be compatible with newer versions but may not be compatible with earlier versions. **HELIOS does not run with Python 2.**
