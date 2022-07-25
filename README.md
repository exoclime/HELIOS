# HELIOS v3.0 #

#### A GPU-ACCELERATED RADIATIVE TRANSFER CODE FOR EXOPLANETARY ATMOSPHERES ####

###### Copyright (C) 2018 - 2022 Matej Malik ######

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

### About ###

HELIOS is an open-source radiative transfer code, which is constructed for studying exoplanetary atmospheres in their full variety. The model atmospheres are one-dimensional and plane-parallel, and the equation of radiative transfer is solved in the hemispheric two-stream approximation with non-isotropic scattering. For given opacities and planetary parameters, HELIOS finds the atmospheric temperature profile in radiative-convective equilibrium and the corresponding planetary emission spectrum.

HELIOS is part of the Exoclimes Simulation Platform ([ESP](http://www.exoclime.org)).

If you use HELIOS for your own work, please cite its two method papers: [Malik et al. 2017](http://ui.adsabs.harvard.edu/abs/2017AJ....153...56M) and [Malik et al. 2019a](https://ui.adsabs.harvard.edu/abs/2019AJ....157..170M). If you use the solid surface feature, please also cite the papers describing its implementation, namely [Malik et al. 2019b](https://ui.adsabs.harvard.edu/abs/2019ApJ...886..142M) and [Whittaker et al. 2022](https://arxiv.org/abs/2207.08889).

Any questions, issues or bug reports are appreciated and can be sent to *malik@umd.edu*.

Thank you for considering HELIOS!

### New version 3.0 --- released July 2022 ###

New features include:

- option to add a non-gray surface albedo, self-consistently incorporated into the atmospheric RT (with option to run a bare-rock case without atmosphere).
- option to mix opacities on-the-fly instead of using a premixed opacity table.
- option to read vertical mixing ratios instead of (or in addition to) equilibrium chemistry.
- option to couple to photochemical kinetics codes.
- option to use Random Overlap (RO) to mix opacities of individual species (previously only correlated-k possible).
- option to include multiple, parameterized cloud decks. Mie-scattering input files are provided.
- option to use physical time stepping with given runtime.
- improved T-P smoothing functionality that satisfies global energy conservation.
- script to easily produce stellar spectrum input files from common databases like PHOENIX.
- (almost) all input parameters are now as command-line options available.
- completely revised "ktable" program. It is now possible to generate k-distribution tables directly from HELIOS-K output files.
- enhanced overall clarify, user-friendliness and automatization based on input options.

### Documentation ###

A detailed documentation of HELIOS can be found [here](https://heliosexo.readthedocs.io/).


