#!/usr/bin/env python3
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator

from sys import argv
import numpy as np
import datetime

# Matplotlib style
plt.style.use('seaborn-colorblind')

# Number of active flavours at initial scale
labels = {-6: "tbar", -5: "bbar", -4: "cbar", -3: "sbar", -2: "dbar", -1: "ubar",
          21: "g", 1: "u", 2: "u", 3: "s", 4: "c", 5: "b", 7: "t"}

if len(argv) is not 2:
    print("Usage")
    print(f"{argv[0]} [target GP archive]")
    exit()

script, target = argv
print(f"Loading archive {target}")
gpdata = np.load(target)

# Kinematics
Q0 = gpdata["Q0"]
xs = gpdata["xgrid"]
nx = len(xs)

# Number of samples
gp_values = gpdata["samples"]
ngen_gp = len(gp_values)

# GP Uncertainty
error = np.sqrt(np.diagonal(gpdata["covariance"]))

# Setup gridspec
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.00, hspace=0.00)
w, h = plt.figaspect(0.5)

# Plot PDFs
with PdfPages(f'plt_{gpdata["setname"]}.pdf') as output:
    for ipdf, pdf in enumerate(gpdata["flavours"]):
        # Setup figure
        fig = plt.figure(figsize=(w, h))
        logax = fig.add_subplot(gs[0])
        linax  = fig.add_subplot(gs[1])

        # Axis formatting
        linax.set_xlim([0.1, 1])
        logax.set_xlim([min(gpdata["xgrid"]), 0.1])
        plt.setp(linax.get_yticklabels(), visible=False)
        logax.set_xscale('log')
        linax.xaxis.set_major_locator(MaxNLocator(5, prune='lower'))

        # GP Mean and standard-deviation
        mslice = gpdata["mean"][ipdf*nx:(ipdf+1)*nx]
        eslice = error[ipdf*nx:(ipdf+1)*nx]

        for ax in fig.get_axes():
            for irep in range(0, ngen_gp):
                gpslice = gp_values[irep][ipdf*nx:(ipdf+1)*nx]
                if irep == 0:
                    ax.plot(xs, gpslice, alpha=0.1, color='b', label="GP Samples")
                else:
                    ax.plot(xs, gpslice, alpha=0.1, color='b')
            ax.plot(xs, mslice+eslice, color='black', linestyle='--')
            ax.plot(xs, mslice-eslice, color='black', linestyle='--', label="GP 1-sigma")

        label = labels[pdf]
        linax.legend(loc='best')
        logax.set_ylabel(f'$x${label}(x, Q={Q0})')
        fig.suptitle(f'Gaussian process sample of {label}(x, Q={Q0}) from {gpdata["prior"]}')

        output.savefig(fig)

    d = output.infodict()
    d['Title'] = 'GPPDF plots'
    d['Author'] = 'Nathan Hartland'
    d['CreationDate'] = datetime.datetime.today()
