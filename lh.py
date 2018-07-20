#!/usr/bin/env python3
""" LHAPDF Grid export """
import os
import numpy as np
from jinja2 import Environment, FileSystemLoader
from sys import argv


def print_lhapdf_header(gpdata):
    data = {
        'SETNAME': gpdata["setname"],
        'XMIN': np.min(gpdata["xgrid"]),
        'XMAX': np.max(gpdata["xgrid"]),
        'QMIN': gpdata["Q0"]-0.01,
        'QMAX': gpdata["Q0"]+0.01,
        'NMEMBERS': len(gpdata["samples"])+1,
        'FLAVOURS': ','.join(map(str, gpdata["flavours"])),
        'NFLAVORS': len(gpdata["flavours"])}

    TEMP_ENV = Environment( autoescape=False,
                            loader=FileSystemLoader(os.getcwd()),
                            trim_blocks=False)
    render = TEMP_ENV.get_template('template.info').render(data)
    with open(f'{data["SETNAME"]}/{data["SETNAME"]}.info', 'w') as f:
        f.write(render)


def print_lhapdf_replica(irep, gpdata, data):
    setname = gpdata["setname"]
    nx = len(gpdata["xgrid"])
    qgrid = [gpdata["Q0"] - 0.01, gpdata["Q0"] + 0.01]  # Hack for now
    with open(f'{setname}/{setname}_{irep:04d}.dat', 'w') as f:
        f.write(f"PdfType: replica\nFormat: lhagrid1\nFromMCReplica: {irep}\n---\n")
        for x in gpdata["xgrid"]:
            f.write(f'{x:.7E} ')
        f.write('\n')
        for q in qgrid:
            f.write(f'{q:.7E} ')
        f.write('\n')
        for p in gpdata["flavours"]:
            f.write(f'{p} ')
        f.write('\n')

        for ix, x in enumerate(gpdata["xgrid"]):
            for iq, q in enumerate(qgrid):
                for ip, p in enumerate(gpdata["flavours"]):
                    gpslice = data[ip*nx:(ip+1)*nx]
                    point = gpslice[ix]
                    f.write(f'{point:.7E} ')
                f.write('\n')
        f.write('---\n')


if len(argv) is not 2:
    print("Usage")
    print(f"{argv[0]} [target GP archive]")
    exit()

script, target = argv
print(f"Loading archive {target}")
gpdata = np.load(target)
os.mkdir(f'{gpdata["setname"]}')

print("Printing LHAPDF replicas")
for igp, gp in enumerate(gpdata["samples"]):
    print(f'Printing replica {igp}')
    print_lhapdf_replica(igp+1, gpdata, gpdata["samples"][igp])
print("Printing replica zero and writing header")
print_lhapdf_replica(0, gpdata, np.mean(gpdata["samples"], axis=0))
print_lhapdf_header(gpdata)
