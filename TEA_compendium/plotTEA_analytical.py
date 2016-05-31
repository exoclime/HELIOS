#! /usr/bin/env python

from readconf import *

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#from PIL import Image
import Image

# Correct directory names
if location_out[-1] != '/':
    location_out += '/'
    
CtoO_list = np.logspace(-1,1,100)

def plotTEA():
    '''
    This code plots a figure of C/O vs. abundances for validation of the analytical chemistry model,
    as shown in Figure. 4 in Malik et al. 2016.
    '''
    
    # Get plots directory, create if non-existent
    plots_dir = location_out + "/plots/"
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    
    # Counts number of arguments given
    noArguments = len(sys.argv)

    # Prints usage if number of arguments different from 4 or adiabatic profile
    if noArguments != 3:
        print("\nUsage  : ../TEA/tea/plotTEA.py atmfile species(divided by comma, no breaks)")
        print("Example: ../TEA/tea/plotTEA.py ../TEA/doc/examples/multiTP/results/multiTP_Example.tea CO,CH4,H2O,NH3\n")
     
    # Sets the first argument given as the atmospheric file
    filename = sys.argv[1]
    
    # Sets the second argument given as the atmospheric file
    filename2 = sys.argv[2]
    
    # Sets the second argument given as the atmospheric file
    vulcanname = sys.argv[3]

    # Sets the species for plotting ('H2' is always required becasue the mixing ratio in the analytical formula 
    # is defined as the ratio between the species and H2)
    species = ['CO', 'CO2', 'CH4', 'H2O','C2H2','H2']

    # Open the two TEA files and read
    f = open(filename, 'r')
    lines = np.asarray(f.readlines())
    f.close()
    f2 = open(filename2, 'r')
    lines2 = np.asarray(f2.readlines())
    f2.close()

    # Get molecules names
    imol = np.where(lines == "#SPECIES\n")[0][0] + 1
    molecules = lines[imol].split()
    nmol = len(molecules)
    for m in np.arange(nmol):
        molecules[m] = molecules[m].partition('_')[0]
    # Get molecules names for tea2
    imol2 = np.where(lines2 == "#SPECIES\n")[0][0] + 1
    molecules2 = lines[imol2].split()
    nmol2 = len(molecules2)
    for m in np.arange(nmol2):
        molecules2[m] = molecules2[m].partition('_')[0]

    #  convert the list to tuple
    species = tuple(species)
    nspec = len(species)

    # Populate column numbers for requested species and 
    #          update list of species if order is not appropriate
    columns = []
    spec    = []
    spec2 = []
    columns2 = []
    for i in np.arange(nmol):
        for j in np.arange(nspec):
            if molecules[i] == species[j]:
                columns.append(i+2)
                spec.append(species[j])
    for i in np.arange(nmol2):
        for j in np.arange(nspec):
            if molecules2[i] == species[j]:
                columns2.append(i+2)
                spec2.append(species[j])
                
    #Calulate the numer of user-input species which is not in moleculues
    nnot = 0
    for _ in species:
        if _ not in molecules:
            nnot += 1

    # Convert spec to tuple
    spec = tuple(spec)
    spec2 = tuple(spec2)
    
    # Concatenate spec with pressure for data and columns
    data    = tuple(np.concatenate((['p'], spec)))
    data2    = tuple(np.concatenate((['p'], spec2)))
    usecols = tuple(np.concatenate(([0], columns)))
    usecols2 = tuple(np.concatenate(([0], columns2)))
    # print 'data'
    # print data
    
    # Load all data for all interested species
    data = np.loadtxt(filename, dtype=float, comments='#', delimiter=None,    \
                    converters=None, skiprows=8, usecols=usecols, unpack=True)
    data2 = np.loadtxt(filename2, dtype=float, comments='#', delimiter=None,    \
                    converters=None, skiprows=8, usecols=usecols, unpack=True)
    
    #print molecules
    # Open a figure
    plt.figure(1)
    plt.clf()
 
    # Set different colours of lines
    colors = ['b','g','r','c','m','y','k', 'orange','pink', 'grey']
    color_index = 0

    # Read mixing ratios of H2
    for i in np.arange(nspec-nnot):
        if spec[i] == 'H2': H2_1 =data[i+1]
        if spec2[i] == 'H2': H2_2 =data2[i+1]
    
      
    tex_labels = {'H':'$H$','H2':'$H_2$','O':'$O$','OH':'$OH$','H2O':'$H_2O$','CH':'$CH$','C':'$C$','CH2':'$CH_2$','CH3':'$CH_3$','CH4':'$CH_4$',\
    'C2':'$C_2$','C2H2':'$C_2H_2$','C2H3':'$C_2H_3$','C2H':'$C_2H$','CO':'$CO$','CO2':'$CO_2$','He':'$He$','O2':'$O_2$'} 
    
    # plot C/O vs mixing ratios(normalized by H2)
    for i in np.arange(nspec-nnot):
        if spec[i]!='H2':
            plt.loglog(np.logspace(-1,1,len(data[i+1])), data[i+1]/H2_1, '-', color=colors[color_index],label=tex_labels[spec[i]],lw=1)
            plt.loglog(np.logspace(-1,1,len(data2[i+1])), data2[i+1]/H2_2, '-', color=colors[color_index],lw=3)
            color_index += 1
        
    
    # Plot analytical solutions
    vulcan_species = ['H','H2','O','OH','H2O','CH','C','CH2','CH3','CH4','C2','C2H2','C2H3','C2H','C2H4','C2H5','C2H6','C4H2','CO','CO2','CH2OH','H2CO','HCO','CH3O',\
    'CH3OH','CH3CO','O2','H2CCO','HCCO','He']
    colors = ['b','g','r','c','m','y','k', 'orange','pink', 'grey']
    

    ################################# VULCAN ######################################
    Heng = np.load(vulcanname)['n_mix'][0]
    
    # load solutions for 800K and 3000K
    Heng1 = Heng[800]
    Heng2 = Heng[3000] 
    
    color_index=0
    for sp in spec :
        if sp != 'H2': 
            plt.scatter(np.logspace(-1,1,len(Heng1[sp]))[::5],Heng1[sp][::5],color=colors[color_index], marker='o',facecolor='None')
            plt.scatter(np.logspace(-1,1,len(Heng2[sp]))[::5] ,Heng2[sp][::5],color=colors[color_index], marker='o')
            color_index+=1
    ################################# VULCAN ######################################
     

    
    # Label the plot
    plt.xlabel('C/O ratio', fontsize=14)
    plt.ylabel('volume mixing ratio' , fontsize=14)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    #display = range(len(plot_spec))
    #Create custom artists
    art0 = plt.Line2D((0,0),(0,0), ls='None')
    Artist1 = plt.Line2D(range(10),range(10), color='black', lw=1)
    Artist2 = plt.Line2D((0,1),(0,0), color='black', lw=3)


    #Create legend from custom artist/label lists
    #plt.legend(frameon=0, prop={'size':12}, loc='best')

    #,'C, EQ', 'C, $K_{zz} = 10^{10}$'  
    plt.legend([handle for i,handle in enumerate(handles)]+[Artist1,Artist2],\
    [label for i,label in enumerate(labels)]+['800 K', '3000 K'], \
    frameon=1, prop={'size':10}, loc=4)
    
    plt.xlim((0.1,10.))     
  
    # Place plot into plots directory with appropriate name 
    plot_out = plots_dir + filename.split("/")[-1][:-4] + '.png'
    plot_eps = plots_dir + filename.split("/")[-1][:-4] + '.eps'
    plt.savefig(plot_out) 
    plt.savefig(plot_eps)  
    plt.close()
    
    # Return name of plot created 
    return plot_out


# Call the function to execute
if __name__ == '__main__':
    # Make plot and retrieve plot's name
    plot_out = plotTEA()
    
    # Open plot
    plot = Image.open(plot_out)
    plot.show()
