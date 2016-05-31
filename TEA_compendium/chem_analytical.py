# Python script to compute atmospheric chemistry (C-H-O system including CO2)
# by Kevin Heng (2nd July 2015)
# modified by Shang-Min, Tsai (26th May 2016)

from numpy import mean,arange,zeros,polynomial,array,interp,exp
from matplotlib import pyplot as plt
import sys, os
import numpy as np
import Image


#Setting the current working directory to the script location
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

#######################################################
#######################################################
temperature1 = 800.0   # first temperature (K)
temperature2 = 3000.0  # second temperature (K)
t1 = str(int(temperature1)); t2 = str(int(temperature2))
pbar = 1.   # pressure in bars
#######################################################


# function to compute first equilibrium constant (K')
def kprime(my_temperature,pbar):
    runiv = 8.3144621   # J/K/mol
    temperatures = arange(500.0, 3100.0, 100.0)
    dg = [96378.0, 72408.0, 47937.0, 23114.0, -1949.0, -27177.0, -52514.0, -77918.0, -103361.0, -128821.0, -154282.0, -179733.0, -205166.0, -230576.0, -255957.0, -281308.0, -306626.0, -331911.0, -357162.0, -382380.0, -407564.0, -432713.0, -457830.0, -482916.0, -507970.0, -532995.0]
    my_dg = interp(my_temperature,temperatures,dg)
    result = exp(-my_dg/runiv/my_temperature)/pbar/pbar
    return result

# function to compute second equilibrium constant (K2')
def kprime2(my_temperature):
    runiv = 8.3144621   # J/K/mol
    temperatures = arange(500.0, 3100.0, 100.0)
    dg2 = [20474.0, 16689.0, 13068.0, 9593.0, 6249.0, 3021.0, -107.0, -3146.0, -6106.0, -8998.0, -11828.0, -14600.0, -17323.0, -20000.0, -22634.0, -25229.0, -27789.0, -30315.0, -32809.0, -35275.0, -37712.0, -40123.0, -42509.0, -44872.0, -47211.0, -49528.0]
    my_dg = interp(my_temperature,temperatures,dg2)
    result = exp(-my_dg/runiv/my_temperature)
    return result

# function to compute second equilibrium constant (K3')
def kprime3(my_temperature,pbar):
    runiv = 8.3144621   # J/K/mol
    temperatures = arange(500.0, 3100.0, 100.0)
    dg3 = [262934.0, 237509.0, 211383.0, 184764.0, 157809.0, 130623.0, 103282.0, 75840.0, 48336.0, 20797.0, -6758.0, -34315.0, -61865.0, -89403.0, -116921.0, -144422.0, -171898.0, -199353.0, -226786.0, -254196.0, -281586.0, -308953.0, -336302.0, -363633.0, -390945.0, -418243.0]
    my_dg = interp(my_temperature,temperatures,dg3)
    result = exp(-my_dg/runiv/my_temperature)/pbar/pbar
    return result
    
# function to compute mixing ratio for methane
# (note: n_o is oxygen abundance, n_c is carbon abundance, kk is K')
def n_methane(n_o,n_c,temp,pbar):
    k1 = kprime(temp,pbar)
    k2 = kprime2(temp)
    k3 = kprime3(temp,pbar)
    a0 = 8.0*k1*k3*k3/k2
    a1 = 8.0*k1*k3/k2
    a2 = 2.0*k1/k2*( 1.0 + 8.0*k3*(n_o-n_c) ) + 2.0*k1*k3
    a3 = 8.0*k1/k2*(n_o-n_c) + 2.0*k3 + k1
    a4 = 8.0*k1/k2*(n_o-n_c)*(n_o-n_c) + 1.0 + 2.0*k1*(n_o-n_c)
    a5 = -2.0*n_c
    result = polynomial.polynomial.polyroots([a5,a4,a3,a2,a1,a0])
    return result[4]   # picks out the correct root of the cubic equation

# function to compute mixing ratio for methane
def n_water(n_o,n_c,temp,pbar):
    k3 = kprime3(temp,pbar)
    n_ch4 = n_methane(n_o,n_c,temp,pbar)
    result = 2.0*k3*n_ch4*n_ch4 + n_ch4 + 2.0*(n_o-n_c)
    return result

# function to compute mixing ratio for carbon monoxide
def n_cmono(n_o,n_c,temp,pbar):
    kk = kprime(temp,pbar)
    n_ch4 = n_methane(n_o,n_c,temp,pbar)
    n_h2o = n_water(n_o,n_c,temp,pbar)
    result = kk*n_ch4*n_h2o
    return result

# function to compute mixing ratio for carbon dioxide
def n_cdio(n_o,n_c,temp,pbar):
    kk2 = kprime2(temp)
    n_h2o = n_water(n_o,n_c,temp,pbar)
    n_co = n_cmono(n_o,n_c,temp,pbar)
    result = n_co*n_h2o/kk2
    return result

# function to compute mixing ratio for acetylene
def n_acet(n_o,n_c,temp,pbar):
    kk3 = kprime3(temp,pbar)
    n_ch4 = n_methane(n_o,n_c,temp,pbar)
    result = kk3*n_ch4*n_ch4
    return result

# compute mixing ratios of methane, water, carbon monoxide and carbon dioxide
n_o = 5e-4          # elemental abundance of oxygen
#c_index = arange(-5.00, -2.01, 0.01)
n_c = np.logspace(-1.,1.,100) *n_o
#n_c = 10.0**c_index   # elemental abundance of carbon
n = len(n_c)

n_ch4 = zeros(n)
n_h2o = zeros(n)
n_co = zeros(n)
n_co2 = zeros(n)
n_c2h2 = zeros(n)
n_ch4_2 = zeros(n)
n_h2o_2 = zeros(n)
n_co_2 = zeros(n)
n_co2_2 = zeros(n)
n_c2h2_2 = zeros(n)

for i in range(0,n):
    n_h2o[i] = n_water(n_o,n_c[i],temperature1,pbar)
    n_ch4[i] = n_methane(n_o,n_c[i],temperature1,pbar)
    n_co[i] = n_cmono(n_o,n_c[i],temperature1,pbar)
    n_co2[i] = n_cdio(n_o,n_c[i],temperature1,pbar)
    n_c2h2[i] = n_acet(n_o,n_c[i],temperature1,pbar)
    n_h2o_2[i] = n_water(n_o,n_c[i],temperature2,pbar)
    n_ch4_2[i] = n_methane(n_o,n_c[i],temperature2,pbar)
    n_co_2[i] = n_cmono(n_o,n_c[i],temperature2,pbar)
    n_co2_2[i] = n_cdio(n_o,n_c[i],temperature2,pbar)
    n_c2h2_2[i] = n_acet(n_o,n_c[i],temperature2,pbar)

xx = n_c/n_o   # carbon-to-oxygen ratio

n_dic ={}
n_dic[int(temperature1)] = {}; n_dic[int(temperature2)] = {}
n_dic[int(temperature1)]['CH4'] = n_ch4
n_dic[int(temperature1)]['C2H2'] = n_c2h2
n_dic[int(temperature1)]['CO'] = n_co
n_dic[int(temperature1)]['CO2'] = n_co2
n_dic[int(temperature1)]['H2O'] = n_h2o

n_dic[int(temperature2)]['CH4'] = n_ch4_2
n_dic[int(temperature2)]['C2H2'] = n_c2h2_2
n_dic[int(temperature2)]['CO'] = n_co_2
n_dic[int(temperature2)]['CO2'] = n_co2_2
n_dic[int(temperature2)]['H2O'] = n_h2o_2

# save output in numpy .npz format
np.savez(dname+'/CtoO_analytical.npz', n_mix = [n_dic], CtoO = n_c, pbar=pbar)

# colors and labels for preview plot
colors = ['c', 'b','g','r','m','y','k','orange']
tex_labels = {'H':'$H$','H2':'$H_2$','O':'$O$','OH':'$OH$','H2O':'$H_2O$','CH':'$CH$','CH2':'$CH_2$','CH3':'$CH_3$','CH4':'$CH_4$','C2H2':'$C_2H_2$','CO':'$CO$','CO2':'$CO_2$','O2':'$O_2$'}

# preview plot
plt.figure(0)
color_index = 0
for sp in n_dic[int(temperature1)].keys():
    plt.plot(xx, n_dic[int(temperature1)][sp], linewidth=1.5, color=colors[color_index], label=sp)
    color_index += 1
color_index = 0
for sp in n_dic[int(temperature2)].keys():
    plt.plot(xx, n_dic[int(temperature2)][sp], linewidth=1.5, color=colors[color_index], ls='--')
    color_index += 1

plt.xscale('log')
plt.yscale('log')
plt.xlim([0.1,10])
plt.ylim([1e-22,1e0])
plt.xlabel(r'C/O', fontsize=14)
plt.ylabel(r'$\tilde{n}_{\rm X}$', fontsize=16)
plt.legend(loc=3, prop={'size':10}, fancybox=0, frameon=1) 
plt.savefig('CtoO.eps', format='eps') #save in EPS format
outname = 'CtoO.png'
plt.savefig(outname, format='png') #save in PNG format
plot = Image.open(outname)
plot.show()

