Parameter File
==============

The input parameters are set in the file ``param.dat``. The individual parameters are the following, with additional info given in round brackets. The respective units are also given.

**GENERAL**

   ``name``

The name of the output. The output directory is given this name and the files begin with this string.

   ``output directory``

The root output directory. This is useful for chain runs or when running multiple instances of the code in parallel.

   ``precision (double, single)``

Under all normal conditions double precision should be used. Single precision should make the code run faster, but somehow does not. Hence, single precision does not provide any advantage at the moment. 

   ``realtime plotting (yes, no, number)``

Determines wether realtime plotting is shown during the iterative run. The output interval can be manually specified.

**GRID** 

   ``isothermal layers (yes, no)``

Determines whether the model uses a constant temperature across each layer (=isothermal layers) or expands the Planck function to first order (=non-isothermal layers). Usually for the temperature iteration non-isothermal layers are recommended, whereas for pure post-processing purposes isothermal layers are sufficient.

   ``number of layers (typically 50 - 200)``

Sets the number of vertical layers in the grid. Usually having 10 layers per magnitude in pressure provides a reasonable compromise between accuracy and computational effort. 

   ``TOA pressure [10^-6 bar] (typically 1)``

The pressure value at the topmost simulated atmospheric layer.

   ``BOA pressure [10^-6 bar] (typically 1e9)``

The pressure value at the bottommost simulated atmospheric layer. The model construct a grid between the BOA and TOA pressures equidistant in log10 P.

   ``planet type (gas, rocky)``

Sets the type of the planet to either rocky or gaseous. This choice impacts the calculation of the vertical altitude in the model.

**ITERATION** 

   ``post-processing only (yes, no)``

Sets whether the model proceeds with the temperature iteration routine ("no") or just propagates the fluxes once through the layers ("yes"). In the latter case, there is no iteration happening and the TP-profile needs to be provided. Setting this option to “yes” is typically used to produce a high resolution spectrum from a given TP-profile. If this value is set to "no", the effective irradiated temperature of the planet is used initially. 

   ``path to temperature file``

Sets the path to the temperature file. This parameter is irrelevant if the option "restart temperatures" is set to "no". 

   ``temperature file format & P unit (helios, TP, PT, TP bar, PT bar)``

Sets the format of the temperature file. See :doc:`structure` for more info.

   ``adaptive interval (6, 20)``

Sets the interval in numerical forward steps which is used between consequent adaptive adjustments of the stepping length. A value of 20 is the conservative safe choice for a stable algorithm. Smaller values may result in convergence issues, but also speed up the iteration. 

   ``TP-profile smoothing (yes, no)``

Determines whether a TP-profile smoothing is applied during the iteration. This option should be deactivated by default. Only if unrealistic kinks in the temperature profile appear can this option be used to smooth those. However, note that smoothing is an artificial intervention at the cost of local radiative balance.

**RADIATION** 

   ``direct irradiation beam (yes, no)``

Includes a separate irradiation beam for the stellar flux at a certain zenith angle (set later). Otherwise, the stellar flux is isotropic and treated by the diffuse flux equations. For the description of average hemispheric conditions this can be switched off. 

   ``scattering (yes, no)``

Determines whether gas (Rayleigh) scattering is used. 

   ``improved two-stream correction (yes, no)``

Activates the two-stream correction of `Heng et al. 2018 <http://adsabs.harvard.edu/abs/2018ApJS..237...29H>`_. This correction makes the two-stream formalism more accurate and should be on in general circumstances.

    ``asymmetry factor g_0 (in range [-1, 1])``

Determines the scattering orientation. As Rayleigh scattering is mostly isotropic, it is recommended to choose zero. A positive value implies forward scattering and a negative value backward scattering. 

   ``path to opacity file``

Sets the path to the opacity table file. For more info on the format of this file see :doc:`structure`. 

   ``diffusivity factor (typically 1.5 - 2)``

Sets the value of the diffusivity factor. If you are not sure, pick 2. 

   ``f factor (typically 0.25 - 1)``

The f factor determines the heat redistribution efficiency in the atmosphere. For day-side emission spectra one typically assumes f = 2/3 = 0.6667 or 0.5. For no redistribution (substellar point) f = 1 and for a full/global redistribution f = 0.25. This option is irrelevant if a direct irradiation beam is used. 

   ``stellar zenith angle [deg] (values: 0 - 89)`` 

The zenith angle of the star with 0 here being vertical irradiation. This parameter is exclusive with the f factor, because the latter is only used for hemispherically averaged condition.

   ``geometric zenith angle correction (yes, no)``

Switches the geometric correction of the zenith angle on/off. For zenith angles > 70 the correction is recommended. For smaller angles it is negligible and can be left switched off (The correction makes the code run slightly slower).

   ``internal temperature [K] (typically 0 - 300 for irrad. planets)``

The internal temperature determines the strength of the internal heating. In this case the internal heat is modeled as blackbody ration with the internal temperature. If internal heating is negligible on the resulting spectrum (e.g. strongly irradiated planets) it is safe to assume this parameter as zero. 

   ``surface/BOA temperature [K]``

Sets the surface temperature (or BOA temperature for a gas planet) manually. This is only relevant for post-processing a given TP profile. In normal circumstances the surface temperature is self-consistently calculated during the iterative run. Even for post-processing purposes, if the standard Helios TP-profile file is used, the value is already included and will be read in from there. So usually this entry can be ignored.

   ``surface/BOA albedo (0-0.999)``

Sets the albedo of the surface (or the BOA for a gas planet). Numerically it sets the reflectivity at the BOA. If set to "1", all radiation is perfectly reflected at the bottom boundary. If set to "0", all radiation is absorbed and re-emitted. For numerical stability reasons, the maximum value is 0.999. For gas planets without any surface the recommended value is "0". In that case the atmosphere below the modeled grid is assumed to possess BOA conditions.

   ``energy budget correction (yes, no)``

Corrects for cut-off wavelengths in the total incoming flux. Due to the lower bound of the wavelength range at 0.3 micron, the fraction of the external radiation at smaller wavelengths is not accounted for. This correction shifts all flux values to give the correct wavelength-integrated flux according to the Stefan-Boltzmann law. As a rule of thumb, this should be switched on for iterative runs and switched off for post-processing purposes.

**CONVECTIVE ADJUSTMENT** 

   ``convective adjustment (yes, no)``

Switches convective adjustment on or off. If set to off, only radiative equilibrium is sought during the temperature iteration. If this option is activated, convective adjustment is applied after a radiative solution has been found. In this way the temperature profile in radiative-convective equilibrium is obtained.

   ``kappa value (value, file)``

Sets manually a constant value to the adiabatic coefficient. For an ideal gas, the value is given by 2 / (2 + n), where n is the number of degrees of freedom for the gas particles. For diatomic particles n = 5. If "file" is set, pre-tabulated values are to be read from a file. See :doc:`structure` for more info on the format of this file. 

   ``entropy/kappa file path``

Sets the path to the file with the tabulated adiabatic coefficient (and optionally the entropy).

   ``damping parameter (auto, 1 - 5000)``

Sets the damping strength of the convective adjustment shift to match the global radiative equilibrium. In almost all cases, this should be set to "auto". For debugging purposes, e.g. if a convergence in a particular case fails, one can try with a fixed value. Usually a value between 1 and 5000 is a good starting point. The larger the number the more stable the model. However, it is also harder to achieve global equilibrium then.

**ASTRONOMICAL PARAMETERS**

   ``stellar spectral model (blackbody, HDF5 data set)``

Sets the model for the stellar irradiation. Simplest approach is to use a blackbody spectrum with the stellar temperature. For this choice no additional input is required. If a realistic stellar spectrum is desired, the spectrum needs to be provided in a HDF5 file with the correct format. A sample file is provided with the installation. See :doc:`structure` for more info.

   ``path to stellar spectrum file``

Sets the path to the HDF5 file containing the stellar spectrum. If "blackbody" is chosen above, this parameter is irrelevant.

   ``planet (manual, pre-defined name)``

The planetary parameters can be either specified manually or read in from a file. A sample file is provided with the installation. See :doc:`structure` for more info.

   ``path to planet data file``

Sets the path to the file with the planetary parameters. If the data are read in from a file the below stated manual parameters are ignored.

   ``surface gravity [cm s^-2] or [log10 (cm s^-2)]``

   ``orbital distance [AU]``

   ``radius planet [R_Jup]``

   ``radius star [R_Sun]``

   ``temperature star [K]``

Manual entry for the planetary, stellar and orbital parameters. These are only read if the "planet" option is set to "manual". The stellar temperature is not only used to calculate the correct blackbody emission. It may also be used together with a realistic stellar spectrum (see "stellar spectral model" option) to shift the stellar spectrum to give the correct bolometric flux (see "energy budget correction" option).

**EXPERIMENTAL / DANGER ZONE**

There are several experimental options, which are under testing for functionality. For the moment these parameters should be left alone. *Beware of the danger zone!*
