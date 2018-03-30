#! /usr/bin/env python

import lhapdf
import numpy as np
import matplotlib.pyplot as plt
import knn_divergence as knn
import scipy

Q0 = 2
# prior = "NNPDF31_nnlo_as_0118_1000"
prior = "180307-nh-002"
pdfs = lhapdf.mkPDFs(prior)
pdfs = pdfs[1:]

# Filling a numpy 2D array with xf(x,Q) samples
xs = [x for x in np.logspace(-5, 0, 50)]
gluon_xfs = np.empty([len(xs), len(pdfs)])
for irep, rep in enumerate(pdfs):
    for ix, x in enumerate(xs):
        gluon_xfs[ix][irep] = rep.xfxQ(0, x, Q0*Q0)

mean       = np.mean(gluon_xfs, axis=1)
covariance = np.cov(gluon_xfs)
error      = np.sqrt(np.diagonal(covariance))

# red dashes, blue squares and green triangles
fig, ax = plt.subplots()
#ax.plot(xs, mean)
#ax.fill_between(xs, mean-error, mean+error)
ax.set_xscale('log')

for rep in gluon_xfs.T:
    ax.plot(xs, rep, alpha=0.1, color='b')

ngen_gp = 100
gluon_gp = np.empty([ngen_gp, len(xs)])
for i in range(0, ngen_gp):
    gluon_gp[i][:] = np.random.multivariate_normal(mean, covariance)
    ax.plot(xs, gluon_gp[i], alpha=0.1, color='r')

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

plt.show()
