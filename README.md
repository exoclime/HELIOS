# HELIOS
*Radiative transfer code optimized for GPUs*


HELIOS is an upcoming open-soure code intended to calculate the radiative transfer in exoplanet atmospheres. The code is written in CUDA C++ and optimized for execution on GPUs.

The code is using a two-stream approximation. First versions will study the pure absorption limit including isothermal and non-isothermal horizontal layers. Later, isotropic and non-isotropic scattering will be added.

HELIOS is part of the exoclime simulation platform ([ESP][1]), which also incorporates the THOR and VULCAN projects. 

Please note that HELIOS is currently in its early stages of development. First versions are expected to come out by the end of 2015.

[1]:http://www.exoclime.net
