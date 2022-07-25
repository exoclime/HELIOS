=========
**About**
=========

HELIOS is an open-source radiative transfer code, which is designed to study exoplanetary atmospheres in their full variety. The model atmospheres are one-dimensional and plane-parallel, and the equation of radiative transfer is solved in the hemispheric two-stream approximation with non-isotropic, multiple scattering. For given opacities and planetary parameters, HELIOS finds the atmospheric temperature profile in radiative-convective equilibrium and the corresponding planetary emission spectrum. Version 3 of the code now also has the option to include a solid surface with a non-gray albedo, calculating the surface temperature in equilibrium and the corresponding surface spectrum.

HELIOS is part of the Exoclimes Simulation Platform (`ESP <http://www.exoclime.org>`_) and is designed to be used together with the equilibrium chemistry solver `FASTCHEM <https://github.com/exoclime/FASTCHEM/>`_ and the opacity calculator `HELIOS-K <https://github.com/exoclime/HELIOS-K/>`_, both also part of the `ESP <http://www.exoclime.org>`_. Those codes compute the equilibrium chemical abundances and the opacities, respectively, both crucial ingredients for atmospheric radiative transfer modeling. It is possible to use chemistry and opacity data from other sources, but then the file format needs to be adjusted as HELIOS accepts only a certain kind of format for the input files. The final premixed opacity table (or individual tables for molecular opacities) to be used in HELIOS can be generated with the ktable program, included in the HELIOS package.

Note that there are sample files included in the installation package as reference to help get started without having to install and run a handful of different codes.

If you use HELIOS for your own work, please cite its two core method papers: `Malik et al. (2017) <http://adsabs.harvard.edu/abs/2017AJ....153...56M>`_ and `Malik et al. (2019a) <https://ui.adsabs.harvard.edu/abs/2019AJ....157..170M/>`_. If you use the solid surface, please also cite the two papers describing the surface implementation: `Malik et al. (2019b) <https://ui.adsabs.harvard.edu/abs/2019ApJ...886..142M>`_ and `Whittaker et al. (2022) <https://arxiv.org/abs/2207.08889>`_.

Any questions, issues or bug reports are appreciated and can be sent to *malik@umd.edu*. 

Thank you for considering HELIOS!

