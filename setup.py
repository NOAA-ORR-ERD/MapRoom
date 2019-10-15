#!/usr/bin/env python

"""
setup.py for MapRoom -- mostly to build the Cython extensions

right now - it mostly is to be used as follows:

python setup.py build_ext --inplace

"""
from setuptools import setup, find_packages

from glob import *
import os
import shutil
import subprocess
import sys

is_64bit = sys.maxsize > 2**32
is_conda = len(list(filter(lambda a: a, ["CONDA" in k for k in os.environ.keys()]))) > 0

exec(compile(open('maproom/_version.py').read(), 'maproom/_version.py', 'exec'))


data_files = []
options = {}
packages = find_packages()
package_data = {
    'maproom': [
        'renderer/gl/font.png',
        'templates/*.bna',
        'icons/*',
        ],
}

install_requires = [
    'numpy',
    'scipy',
    'pyopengl',
    'cython',
    'docutils',
    'markdown',
    'reportlab',
    'pyparsing',
    'requests',
    'python-dateutil',
    'pytz',
    'cftime',  # required by netcdf4 but not always installed?
    'wxpython',
    'sawx>=1.5.3',
    'pillow',
    'urllib3',
    'certifi',
    'chardet',
    'idna',
    'packaging',
    'libmaproom>=5.1',
    'omnivore_framework>=4',  # dependencies not yet in pypi or conda: glsvg, lat_lon_parser, post_gnome, pyugrid
]

if is_conda:
    install_requires.extend([
        # 'owslib',  # commented out because it overwrites the conda install of OWSLib
        'shapely',
        'pyproj',
        'netCDF4',
    ])
else:
    install_requires.extend([
        'owslib',
        'shapely<1.7',
        'pyproj==1.9.6',  # pyproj version 2 fails outside the -180/+180 range
        'netCDF4==1.3.1',  # newer versions in pypi fail with missing symbol
    ])


#if sys.platform != "win32":
if False:  # disabling for everybody temporarily
    # pyopengl_accelerate can fail on windows, sometimes. It's not necessary,
    # so by default I'm not including it.
    install_requires.extend([
        'pyopengl_accelerate',
    ])

setup(
    name="MapRoom",
    version=__version__,
    description="High-performance 2d mapping",
    author="NOAA",
    install_requires=install_requires,
    setup_requires=[
        'packaging',
    ],
    data_files=data_files,
    packages=packages,
    package_data=package_data,
    app=["maproom.py"],
    entry_points = {
        # NOTE: entry points are processed lexicographically, not in the order
        # specified, so force e.g. verdat loader to come before text loader
        "sawx.loaders": [
            '01verdat = maproom.loaders.verdat',
            '00project = maproom.loaders.project',
            '08nc_particles = maproom.loaders.nc_particles',
            '08ugrid = maproom.loaders.ugrid',
            '10bna = maproom.loaders.bna',
            '10gps = maproom.loaders.gps',
            '10cmd = maproom.loaders.logfile',
            '20gdal = maproom.loaders.gdal',
            '20shapefile = maproom.loaders.shapefile',
            '90text = maproom.loaders.text',
        ],

        "sawx.documents": [
            '00layer_manager = maproom.layer_manager',
        ],

        "sawx.editors": [
            'editor = maproom.editor',
        ],

        "sawx.remember": [
            'styles = maproom.styles',
            'servers = maproom.servers',
        ],
    },
    options=options,
    zip_safe = False,
)
