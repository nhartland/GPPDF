#! /usr/bin/env python

import lhapdf
import numpy as np
import matplotlib.pyplot as plt
import knn_divergence as knn
import scipy
import lh

# Number of gaussian processes
ngen_gp = 100

# Prior PDF
prior = "180307-nh-002"
pdfs  = lhapdf.mkPDFs(prior)
pdfs  = pdfs[1:]  # Skip replica-0

# Number of active flavours at initial scale
nfl  = 4
npdf = 2*nfl + 1
flavours = range(-nfl, nfl+1)
assert(len(flavours) == npdf)
labels = {-6: "tbar", -5: "bbar", -4: "cbar", -3: "sbar", -2: "dbar", -1: "ubar",
           0: "g", 1: "u", 2: "u", 3: "s", 4: "c", 5: "b", 7: "t"}

# Kinematics
Q0 = lh.QGRID[0]
xs = lh.XGRID
nx = len(xs)

pdf_values = np.empty([npdf*nx, len(pdfs)])
for irep, rep in enumerate(pdfs):
    for ipdf, pdf in enumerate(flavours):
        for ix, x in enumerate(xs):
            pdf_values[nx*ipdf + ix][irep] = rep.xfxQ(pdf, x, Q0*Q0)

mean       = np.mean(pdf_values, axis=1)
covariance = np.cov(pdf_values)
error      = np.sqrt(np.diagonal(covariance))

# Generate gaussian processes
gp_values = np.empty([ngen_gp, npdf*nx])  # Generated transposed w.r.t pdf_values for ease
for i in range(0, ngen_gp):
    gp_values[i][:] = np.random.multivariate_normal(mean, covariance)

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
#


lh.print_lhapdf_header(ngen_gp, nfl)

#for rep in gluon_xfs.T:
#    ax.plot(xs, rep, alpha=0.1, color='b')
#

#gluon_gp = gluon_gp.T
#kurtosis1   = scipy.stats.kurtosis(gluon_xfs, axis=1)
#kurtosis2   = scipy.stats.kurtosis(gluon_gp, axis=1)
#
#KLD = []
#for i in range(len(xs)):
#    slice1 = np.asarray([gluon_xfs[i]]).T
#    slice2 = np.asarray([gluon_gp[i]]).T
#    kl = knn.naive_estimator(slice1, slice2, k=1)
#    KLD.append(kl)
#
## print(knn.naive_estimator(gluon_gp, gluon_xfs))
#
#fig, ax = plt.subplots()
#ax.plot(xs, kurtosis1)
#ax.plot(xs, kurtosis2)
#ax.plot(xs, KLD)
#ax.set_xscale('log')

#plt.show()
