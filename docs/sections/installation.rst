Installation
============

HELIOS
------

HELIOS lives on GitHub as part of the Exoclimes Simulation Platform (`ESP <https://github.com/exoclime>`_) at the repository `<https://github.com/exoclime/HELIOS>`_.

For the installation of HELIOS there are two choices:

* Using the terminal type::

    git clone https://github.com/exoclime/HELIOS.git "insert_target_dir"

  to clone into the GitHub repository. In addition to downloading the required files, this gives you full Git functionality. In case you don't have Git installed on your machine you can get it e.g. from `here <https://git-scm.com/downloads>`_.

* If you want to bypass Git, simply download the `ZIP archive <https://github.com/exoclime/HELIOS.matej/archive/master.zip>`_, which contains all the necessary files. Unpack the ZIP to a local directory and you are ready to go!

In order to combine various opacity sources and to construct the opacity tables, FastChem is required. It is **recommended to be installed**. 

If you want to calculate your own opacities, HELIOS-K is the tool for you. However, as many opacities are pre-calculated on the `ESP server <https://chaldene.unibe.ch/>`_ at the University of Bern, this step is **not necessary**.

FastChem
--------

FastChem is also part of the ESP and is found at `<https://github.com/exoclime/FastChem>`_.

Analogously to HELIOS, the files can either obtained by cloning into the GitHub repository or downloaded within a ZIP archive.

How to install and run FastChem is described in the ``README.md`` file, found in the FastChem repository.

HELIOS-K
--------

HELIOS-K is also part of the ESP and is found at `<https://github.com/exoclime/helios-k>`_.

Analogously to HELIOS, the files can either obtained by cloning into the GitHub repository or downloaded within a ZIP archive.

How to install and run HELIOS-K is described in the ``README.md`` file, found in the HELIOS-K repository.