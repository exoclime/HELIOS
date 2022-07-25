import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from source import tools as tls

#################################### FUNCTIONS ####################################

def read_and_plot(ax,
                  path,
                  label="",
                  width=1,
                  style="-",
                  color="blue",
                  alpha=0.6,
                  rebin=0
                  ):

    lamda, spec = tls.read_helios_spectrum(path, type='emission')

    if rebin > 0:
        lamda, spec = tls.rebin_spectrum_to_resolution(lamda, spec, resolution=rebin, w_unit='micron')

    line, = ax.plot(lamda, spec, color=color,linewidth=width, linestyle=style, label=label, alpha=alpha)

    return line

########################################### READ & PLOT ##########################################

fig, ax = plt.subplots()

read_and_plot(ax, "../output/0/0_TOA_flux_eclipse.dat", label='your first spectrum')

ax.set(yscale='log', xlim=[0.25, 20], xscale='log', xlabel='wavelength ($\mu$m)', ylabel='flux (erg s$^{-1}$ cm$^{-3}$)')

ax.set_xticks([0.5, 1, 2, 3, 5, 10, 20])
ax.set_xticklabels(['0.5', '1', '2', '3', '5', '10', '20'])

ax.legend(loc='best', frameon=True)

plt.savefig("spectrum.pdf")
plt.show()