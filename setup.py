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

# setuptools / pip can really make a mess of this
# install the dependencies manually
# e.g. conda install --file conda_requirements.txt

install_requires = []
# install_requires = [
#     'numpy',
#     'scipy',
#     'pyopengl',
#     'cython',
#     'docutils',
#     'markdown',
#     'reportlab',
#     'pyparsing',
#     'requests',
#     'python-dateutil',
#     'pytz',
#     'cftime',  # required by netcdf4 but not always installed?
#     'wxpython',
#     'pillow',
#     'urllib3',
#     'certifi',
#     'chardet',
#     'idna',
#     'packaging',
#     'libmaproom>=5.1',
#     'pyugrid',
#     'lat_lon_parser',
#     'jsonpickle>=0.9.4',
#     'bson<1.0.0',
#     'appdirs',
#     'configobj',
# ]

# if is_conda:
#     install_requires.extend([
#         # 'owslib',  # commented out because it overwrites the conda install of OWSLib
#         'shapely',
#         'pyproj',
#         'netCDF4',
#     ])
# else:
#     install_requires.extend([
#         'owslib',
#         'shapely<1.7',
#         'pyproj==1.9.6',  # pyproj version 2 fails outside the -180/+180 range
#         'netCDF4==1.3.1',  # newer versions in pypi fail with missing symbol
#     ])


# #if sys.platform != "win32":
# if False:  # disabling for everybody temporarily
#     # pyopengl_accelerate can fail on windows, sometimes. It's not necessary,
#     # so by default I'm not including it.
#     install_requires.extend([
#         'pyopengl_accelerate',
#     ])

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
#    app=["maproom.py"],
    entry_points = {
        # NOTE: entry points are processed lexicographically, not in the order
        # specified, so force e.g. verdat loader to come before text loader
        "maproom.app_framework.loaders": [
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
            '99generic_text = maproom.app_framework.loaders.text',
        ],

        "maproom.app_framework.documents": [
            '00layer_manager = maproom.layer_manager',
            '99text = maproom.app_framework.documents.text',
        ],

        "maproom.app_framework.editors": [
            'editor = maproom.editor',
            'html = maproom.app_framework.editors.html_viewer',
            'text = maproom.app_framework.editors.text_editor',
        ],

        "maproom.app_framework.remember": [
            'app = maproom.app_framework.application',
            'styles = maproom.styles',
            'servers = maproom.servers',
        ],
    },
    options=options,
    zip_safe = False,
)
