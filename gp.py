#! /usr/bin/env python

import lhapdf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from xg import XGRID

# Number of gaussian processes
ngen_gp = 100

# Prior PDF
# prior = "180307-nh-002"
prior  = "171113-nh-002"
pdfset = lhapdf.getPDFSet(prior)
replicas  = pdfset.mkPDFs()[1:]

# Number of active flavours at initial scale
labels = {-6: "tbar", -5: "bbar", -4: "cbar", -3: "sbar", -2: "dbar", -1: "ubar",
          21: "g", 1: "u", 2: "u", 3: "s", 4: "c", 5: "b", 7: "t"}

# Available flavours
flavour_string = pdfset.get_entry("Flavors").split(",")
nfl, flavours = len(flavour_string), list(map(int, flavour_string))

# Initial scale and x-grid
Q0 = float(pdfset.get_entry("QMin"))
xs, nx = XGRID, len(XGRID)

print(f"Sampling {nx} x-points at initial scale: {Q0} GeV")
grid_points = list(itertools.product(flavours, xs))
pdf_values = np.empty([nfl*nx, len(replicas)])
for irep, rep in enumerate(replicas):
    for ipt, pt in enumerate(grid_points):
        pdf_values[ipt][irep] = rep.xfxQ(pt[0], pt[1], Q0)

print("Computing stats")
mean       = np.mean(pdf_values, axis=1)
covariance = np.cov(pdf_values)
error      = np.sqrt(np.diagonal(covariance))

# Condition covariance matrix a bit
# https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
# This is probably only neccesary because I'm including x=1 in the matrix
# but x=1 is needed in the matrix for LHAPDF reasons..
min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
while min_eig < 0:
    print("WARNING: Covariance matrix not positive-semidefinite")
    print(f"Minimum eigenvalue: {min_eig}")
    print("Introducing regulator...")
    covariance -= 100*min_eig * np.eye(*covariance.shape)
    min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
    print(f"New minimum: {min_eig}")

# Generate gaussian processes
print("Generating GPs")
gp_values = np.random.multivariate_normal(mean, covariance, ngen_gp, 'raise')

# Plot PDFs
for ipdf, pdf in enumerate(flavours):
    fig, ax = plt.subplots()
    mslice = mean[ipdf*nx:(ipdf+1)*nx]
    eslice = error[ipdf*nx:(ipdf+1)*nx]
    ax.plot(xs, mslice)
    ax.fill_between(xs, mslice-eslice, mslice+eslice)

    for irep in range(0, min(ngen_gp, 100)):
        gpslice = gp_values[irep][ipdf*nx:(ipdf+1)*nx]
        ax.plot(xs, gpslice, alpha=0.1, color='r')

    ax.set_xscale('log')
    fig.savefig(f'pdf_{labels[pdf]}.pdf')


np.savez_compressed(f'GP_{prior}_{len(gp_values)}',
                    prior=prior,
                    setname=f'GP_{prior}_{len(gp_values)}',
                    mean=mean,
                    covariance=covariance, Q0=Q0,
                    flavours=flavours, xgrid=xs,
                    samples=gp_values)

