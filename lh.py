#!/usr/bin/env python
""" LHAPDF Grid export """
import os
import numpy as np
from jinja2 import Environment, FileSystemLoader

QGRID = [1.64, 1.66]
XGRID = [x for x in np.logspace(-9, 0, 100)]
NX = len(XGRID)

NFL = 4
FLAVOURS = list(range(-NFL, NFL+1))
FLAVOURS[NFL] = 21  # Gluon


def print_lhapdf_header(ngen_gp):
    data = {
        'XMIN': np.min(XGRID),
        'XMAX': np.max(XGRID),
        'QMIN': np.min(QGRID),
        'QMAX': np.max(QGRID),
        'NMEMBERS': ngen_gp-1,
        'FLAVOURS': FLAVOURS,
        'NFLAVORS': len(FLAVOURS)}

    TEMP_ENV = Environment( autoescape=False,
                            loader=FileSystemLoader(os.getcwd()),
                            trim_blocks=False)
    render = TEMP_ENV.get_template('template.info').render(data)
    with open("output/GPPDF.info", 'w') as f:
        f.write(render)


# No subgrids here for now
# Need to add subgrids, otherwise something wierd is happening
def print_lhapdf_replica(irep, data):
    with open(f'output/GPPDF_{irep:04d}.dat', 'w') as f:
        f.write(f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {irep}\n---\n")
        for x in XGRID:
            f.write(f'{x:.7E} ')
        f.write('\n')
        for q in QGRID:
            f.write(f'{q:.7E} ')
        f.write('\n')
        for p in FLAVOURS:
            f.write(f'{p} ')
        f.write('\n')

        for ix, x in enumerate(XGRID):
            for iq, q in enumerate(QGRID):
                for ip, p in enumerate(FLAVOURS):
                    gpslice = data[ip*NX:(ip+1)*NX]
                    point = gpslice[ix]
                    f.write(f'{point:.7E} ')
                f.write('\n')
        f.write('---\n')
