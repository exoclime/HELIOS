The purpose of this run is to benchmark the analytical chemistry(Heng & Lyons 2016) against thermochemical equilibrium code TEA (Blecic et al. 2016)

To reproduce Figure 4 in Malik et al. 2017:

1. Unzip all the files into the same directory, in this example, we use "achem". Move "plotTEA_analytical.py" into the directory /tea.
   "plotTEA_analytical.py" is modified from the original TEA script "plotTEA" to produce benchmarked results of the analytical chemistry (Heng & Lyons 2016) and TEA.
   "plotTEA_analytical.py" is required to run in the folder of /tea.
   "chem_analytical.py" is the python script for the analytical chemistry.
   "CtoO_T800.atm" and "CtoO_T3000.atm" are the two input files for TEA.
   The outputfiles are also included ("CtoO_analytical.npz" for analytical chemistry and "CtoO_T800.tea and" "CtoO_T3000.tea" for TEA).
   So for a quick plot, skip to 4. to generate the plot from the included output files. 
   Steps 2. and 3. are to re-produce the output files. 
 
2. Run plotTEA_analytical.py and the output will be stored in "CtoO_analytical.npz"

3. "CtoO_T800.atm" and "CtoO_T3000.atm" are the two pre-atmosphere input files to generate Figure 4.
They are not representing real atmospheres but a list of pressure (bar), temperature (K), elemental abundance for calculation. The temperature and pressure are always kept the same: T = 800K, P = 1bar in "CtoO_T800.atm" and T = 3000K, P = 1bar in "CtoO_T3000.atm". The elemental abundances correspond to varing C/O from 0.1 to 10. 

Run "runatm.py" as the stanrdard preocedure of performing multiple T,P calculation for TEA. Users are referred to the manual of TEA "TEA-UserManual.pdf".
The two T,P inputs are "CtoO_T800.atm" and "CtoO_T3000.atm". Copy the output files back into the current directory("achem").  

4. Finally, under the /run directory in TEA, run the following
../tea/plotTEA_analytical.py ../tea/achem/CtoO_T800.tea ../tea/achem/CtoO_T3000.tea ../tea/achem/CtoO_analytical.npz  plot_name
to generate the plot.

