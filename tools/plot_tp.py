import matplotlib.pyplot as plt
import tools as tls

#################################### FUNCTIONS ####################################

def read_and_plot(ax, path, color='blue', shade='darkorange', label='', width=2, style='-'):

    press, temp, cpress0, ctemp0, cpress1, ctemp1, cpress2, ctemp2, cpress3, ctemp3 = tls.read_helios_tp(path)

    ax.plot(ctemp0, cpress0, color=shade,linewidth=8, alpha=0.7)
    ax.plot(ctemp1, cpress1, color=shade,linewidth=8, alpha=0.7)
    ax.plot(ctemp2, cpress2, color=shade,linewidth=8, alpha=0.7)
    ax.plot(ctemp3, cpress3, color=shade,linewidth=8, alpha=0.7)

    line, = ax.plot(temp, press, color=color, linewidth=width, linestyle=style, alpha=1, label=label)

    return line

########################################### PLOT ##########################################

fig, ax = plt.subplots()

read_and_plot(ax, "../output/0/0_tp.dat", label="a TP-profile")

ax.set(ylim=[1e3, 1e-6], yscale='log', xlabel=r'temperature (K)', ylabel=r'pressure (bar)')

ax.legend(loc='best', frameon=True)

plt.savefig("tp.pdf")
plt.show()