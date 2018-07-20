# Gaussian Process samples of Parton Distribution Functions
<p align="center">
  <img src="https://i.imgur.com/5cavM2x.png">
</p>

This repository contains code for the construction and sampling of a Gaussian
Process (GP) from an input Parton Distribution Function (PDF) set.

Aside from a python installation and some standard python packages (numpy,
matplotlib, argparse) the only other requirement is an LHAPDF6 installation
callable from python.

### Usage
```Shell
    gppdf.py [PDF set] [N]
```
Generates a numpy archive containing `N` samples of a GP defined according to
the mean- and covariance-functions of the input `PDF set`. For the format of the
archive see the header of `gppdf.py`.

### Visualisation
To quickly plot the resulting samples, use
```Shell
    gppdf_plot.py [Output of gppdf.py]
```
to generate plots like the header image.

### Limitations
This code will only work with Monte Carlo PDF Sets.
