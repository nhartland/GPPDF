#! /usr/bin/env python

import lhapdf
import numpy as np
import matplotlib.pyplot as plt
import lh

# Number of gaussian processes
ngen_gp = 1000

# Prior PDF
# prior = "180307-nh-002"
prior = "171113-nh-002"
pdfs  = lhapdf.mkPDFs(prior)
pdfs  = pdfs[1:]  # Skip replica-0

# Number of active flavours at initial scale
nfl  = lh.NFL
flavours = lh.FLAVOURS
npdf = len(flavours)
assert(len(flavours) == npdf)
labels = {-6: "tbar", -5: "bbar", -4: "cbar", -3: "sbar", -2: "dbar", -1: "ubar",
           21: "g", 1: "u", 2: "u", 3: "s", 4: "c", 5: "b", 7: "t"}

# Kinematics
Q0 = lh.QGRID[0]
xs = lh.XGRID
nx = len(xs)

print("Reading prior PDF values")
pdf_values = np.empty([npdf*nx, len(pdfs)])
for irep, rep in enumerate(pdfs):
    for ipdf, pdf in enumerate(flavours):
        for ix, x in enumerate(xs):
            pdf_values[nx*ipdf + ix][irep] = rep.xfxQ(pdf, x, Q0)

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
    print("Covariance matrix not positive-semidefinite")
    print(min_eig)
    covariance -= 100*min_eig * np.eye(*covariance.shape)
    min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
    print(f"New min eig: {min_eig}")

# Generate gaussian processes
print("Generating GPs")
gp_values = np.empty([ngen_gp, npdf*nx])  # Generated transposed w.r.t pdf_values for ease
for i in range(0, ngen_gp):
    print(f"Generated: {i}/{ngen_gp}")
    gp_values[i][:] = np.random.multivariate_normal(mean, covariance)
    lh.print_lhapdf_replica(i, gp_values[i])

# Plot PDFs
for ipdf, pdf in enumerate(flavours):
    fig, ax = plt.subplots()
    mslice = mean[ipdf*nx:(ipdf+1)*nx]
    eslice = error[ipdf*nx:(ipdf+1)*nx]
    ax.plot(xs, mslice)
    ax.fill_between(xs, mslice-eslice, mslice+eslice)

    for rep in gp_values:
        gpslice = rep[ipdf*nx:(ipdf+1)*nx]
        ax.plot(xs, gpslice, alpha=0.1, color='r')

    ax.set_xscale('log')
    fig.savefig(f'pdf_{labels[pdf]}.pdf')


print("Generating replica zero and writing header")
lh.print_lhapdf_header(ngen_gp)
lh.generate_replica_zero()
