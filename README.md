# HELIOS
*Radiative transfer code optimized for GPUs*


HELIOS is an upcoming open-soure code intended to calculate the radiative transfer in exoplanet atmospheres. The code is written in CUDA C++ and optimized for execution on GPUs.

The code is based on a two-stream approximation. First versions will study the pure absorption limit including isothermal and non-isothermal horizontal layers. Later, isotropic and non-isotropic scattering will be added.

HELIOS is part of the exoclime simulation platform ([ESP][1]), which also incorporates the THOR and VULCAN projects. 

HELIOS-K a fast opacity calculator has been released.

## Getting the source
All codes in HELIOS can be cloned with a single

    git clone --recursive https://github.com/exoclime/HELIOS

Please note the `--recursive`, updating source is done by the two steps

    git pull
    git submodule foreach git pull

## Reporting issues
Please use the issue tracker of the respective submodule:
  - [HELIOS-K][2] for issues in the k-calculator

[1]:http://www.exoclime.net
[2]:https://github.com/exoclime/HELIOS-K/issues/new
