Requirements
============

HELIOS was developed on a machine with Mac OS X 10.9 - 10.13 and also successfully tested on Ubuntu and Archlinux. It has not been tested to run on other operating systems, but should in principle work if the following requirements are met.

Hardware
--------

HELIOS is a GPU-accelerated program developed with CUDA. It thus requires an NVIDIA graphics card (GPU) to operate on. Any relatively new GeForce or Tesla card should be sufficient.

CUDA
----

CUDA is the NVIDIA API responsible for the communication between the graphics card (aka device) and the CPU (aka host). The software package consists of the core libraries, development utilities and the NVCC compiler to interpret C/C++ code. The CUDA toolkit can be downloaded from `here <https://developer.nvidia.com/cuda-downloads>`_.

HELIOS has been tested with CUDA 7.5 and 8.0. In principle, it should be compatible with any newer version.

Python
------

HELIOS's computational core is written in CUDA C++, but the user shell comes in Python modular format. To communicate between the host and the device the PyCUDA wrapper is used.

The following Python packages are required.

* numpy
* scipy
* astropy
* matplotlib
* h5py
* PyCUDA

Some of them may be already included in the python distribution. Otherwise they can be installed with the Python package manager pip. To install e.g. PyCUDA type::

   pip install pycuda

This may fail, if you don't have admin permissions on the machine you are trying to install software. For this case, so-called *virtual environments* exist which embed your Python installation in a user-manageable frame. See e.g. `this <https://docs.python-guide.org/dev/virtualenvs/>`_ for a nice tutorial on that.

Note that HELIOS has been tested with Python 3.5.x and 3.6.x. It may not be compatible with earlier versions. HELIOS does not run with Python 2.
