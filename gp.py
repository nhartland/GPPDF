#! /usr/bin/env python
"""
    Gaussian Process Parton Distribution Functions (GPPDFs)

    This script takes a prior (replica) PDF Set from LHAPDF and uses it to
    define a Gaussian Process (GP). Rather than assuming a covariance function,
    the covariance function of the prior PDF set is measured and used.
    Correspondingly the GP mean is simply computed as the mean over the input
    replicas. The script outputs the parameters of the GP, along with `ngen`
    samples of the GP.

    The output is a numpy 'npz' format file containing
        - The name of the prior PDF
        - The list of flavours in the GP
        - The grid of x-points sampled in the GP
        - The GP mean function evaluated on the x-grid
        - The GP covariance function evaluated on the x-grid
        - A numpy array of `ngen` samples of the GP

    The sample array has shape (`ngen`, `nx*nf`) where `nx` is the number of
    points in the sampled x-grid, and `nf` is the number of active flavours in
    the GP. The x-grid points are currently hardcoded in xg.py, the flavour basis
    is read from the prior LHAPDF set.
"""

import lhapdf
import numpy as np
import itertools
from xg import XGRID

# Number of gaussian processes
ngen_gp = 100000

# Prior PDF
# prior = "180307-nh-002"
prior  = "171113-nh-002"
pdfset = lhapdf.getPDFSet(prior)
replicas  = pdfset.mkPDFs()[1:]

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

# Condition covariance matrix a bit
# https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
# This is due to including b in the covariance matrix
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

np.savez_compressed(f'GP_{prior}_{len(gp_values)}',
                    prior=prior,
                    setname=f'GP_{prior}_{len(gp_values)}',
                    mean=mean,
                    covariance=covariance, Q0=Q0,
                    flavours=flavours, xgrid=xs,
                    samples=gp_values)

