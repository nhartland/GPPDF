#! /usr/bin/env python3
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
        - A numpy array of `nsamples` samples of the GP

    The sample array has shape (`nsamples`, `nx*nf`) where `nx` is the number of
    points in the sampled x-grid, and `nf` is the number of active flavours in
    the GP. The x-grid points are currently hardcoded in xg.py, the flavour basis
    is read from the prior LHAPDF set.
"""

import lhapdf
import argparse
import itertools
import numpy as np
from xg import XGRID


def get_active_flavours(pdfset, Q0):
    """
     Trim pdfs if their thresholds are above the initial scale

     LHAPDF sets aren't very helpful here in that it does not have separate
     entries (typically) for mass and threshold. Have to assume q threshold is
     identical to q mass.
    """
    flavour_string = pdfset.get_entry("Flavors").split(",")
    flavours = list(map(int, flavour_string))
    mc = float(pdfset.get_entry("MCharm"))
    mb = float(pdfset.get_entry("MBottom"))
    mt = float(pdfset.get_entry("MTop"))
    if mc > Q0:
        flavours = [f for f in flavours if abs(f) != 4]
    if mb > Q0:
        flavours = [f for f in flavours if abs(f) != 5]
    if mt > Q0:
        flavours = [f for f in flavours if abs(f) != 6]
    return flavours


def generate_gp(prior, nsamples):
    """ Generate the GP and `nsamples` GP samples from a `prior` PDF """

    pdfset = lhapdf.getPDFSet(prior)
    replicas  = pdfset.mkPDFs()[1:]

    # Initial scale and x-grid
    Q0 = float(pdfset.get_entry("QMin"))
    xs, nx = XGRID, len(XGRID)

    # Available flavours
    flavours = get_active_flavours(pdfset, Q0)
    nfl = len(flavours)

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
    # Should use attempt on multivariate_normal instead
    min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
    while min_eig < 0:
        print("WARNING: Covariance matrix not positive-semidefinite")
        print(f"Minimum eigenvalue: {min_eig}")
        print("Introducing regulator...")
        covariance -= 100*min_eig * np.eye(*covariance.shape)
        min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
        print(f"New minimum: {min_eig}")

    # Generate gaussian processes
    # Break this into ch. decomp etc for progress monitoring/parallelisation
    print("Generating GPs")
    gp_values = np.random.multivariate_normal(mean, covariance, nsamples, 'raise')

    outfile = f'GP_{prior}_{len(gp_values)}'
    np.savez_compressed(outfile,
                        prior=prior,
                        setname=f'GP_{prior}_{len(gp_values)}',
                        mean=mean,
                        covariance=covariance, Q0=Q0,
                        flavours=flavours, xgrid=xs,
                        samples=gp_values)
    return outfile


#TODO x-grid selection, separate out GP definition from sample generation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("priorpdf", help="prior LHAPDF grid used to generate GP", type=str)
    parser.add_argument("nsamples", help="number of GP samples", type=int)
    args = parser.parse_args()
    outfile = generate_gp(args.priorpdf, args.nsamples)
    print(f'Results output to {outfile}')

