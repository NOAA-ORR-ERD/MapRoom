#!/usr/bin/env python

"""
setup.py for MapRoom -- mostly to build the Cython extensions

right now - it mostly is to be used as follows:

python setup.py build_ext --inplace

"""


from Cython.Distutils import build_ext
from setuptools import setup, find_packages, Extension

from glob import *
import os
import shutil
import subprocess
import sys

is_64bit = sys.maxsize > 2**32
is_conda = len(list(filter(lambda a: a, ["CONDA" in k for k in os.environ.keys()]))) > 0

exec(compile(open('maproom/_version.py').read(), 'maproom/_version.py', 'exec'))

# find the various headers, libs, etc.
import numpy
gl_include_dirs = [numpy.get_include()]
gl_library_dirs = []
gl_libraries = ["GL", "GLU"]

if sys.platform.startswith("win"):
    gl_libraries = ["opengl32", "glu32"]
elif sys.platform == "darwin":
    gl_include_dirs.append(
        "/System/Library/Frameworks/OpenGL.framework/Headers",
    )
    gl_library_dirs.append(
        "/System/Library/Frameworks/OpenGL.framework/Libraries",
    )

# Definintion of compiled extension code:
bitmap = Extension("maproom.library.Bitmap",
                   sources=["maproom/library/Bitmap.pyx"],
                   include_dirs=[numpy.get_include()],
                   extra_compile_args = ["-O3" ],
                   )

shape = Extension("maproom.library.Shape",
                  sources=["maproom/library/Shape.pyx"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args = ["-O3" ],
                  )

tree = Extension("maproom.library.scipy_ckdtree",
                 sources=["maproom/library/scipy_ckdtree.pyx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args = ["-O3" ],
                 )

tessellator = Extension("maproom.renderer.gl.Tessellator",
                        sources=["maproom/renderer/gl/Tessellator.pyx"],
                        include_dirs=gl_include_dirs,
                        library_dirs=gl_library_dirs,
                        libraries=gl_libraries,
                        extra_compile_args = ["-O3" ],
                        )

render = Extension("maproom.renderer.gl.Render",
                   sources=["maproom/renderer/gl/Render.pyx"],
                   include_dirs=gl_include_dirs,
                   library_dirs=gl_library_dirs,
                   libraries=gl_libraries,
                   extra_compile_args = ["-O3" ],
                   )

# pytriangle extension

DEFINES = [("TRILIBRARY", None), # this builds Triangle as a lib, rather than as a command line program.
           ("NO_TIMER", None), # don't need the timer code (*nix specific anyway)
           ("REDUCED", None),
           ]

# Add the defines for disabling the FPU extended precision           ] 
## fixme: this needs a lot of work!
##        it's really compiler dependent, not machine dependent
if sys.platform == 'darwin':
    print("adding no CPU flags for mac")
    ## according to:
    ## http://www.christian-seiler.de/projekte/fpmath/
    ## nothing special is required on OS-X !
    ##
    ## """
    ##     the precision is always determined by the largest operhand type in C.
    ## 
    ##     Because of this, Mac OS X does not provide any C wrapper macros to
    ##     change the internal precision setting of the x87 FPU. It is simply
    ##     not necessary. Should this really be wanted, inline assembler would
    ##     probably be possible, I haven't tested this, however.


    ##     Simply use the correct datatype and the operations performed will have the
    ##     correct semantics
    ## """
elif sys.platform == 'win32':
    print("adding define for Windows for FPU management")
    DEFINES.append(('CPU86', None))
elif 'linux' in sys.platform :#  something for linux here...
    print("adding CPU flags for Intel Linux")
    DEFINES.append(('LINUX', None))
else:
    raise RuntimeError("this system isn't supported for building yet")

pytriangle = Extension(
    "pytriangle",
    sources = [ "deps/pytriangle-1.6.1/src/pytriangle.pyx",
                "deps/pytriangle-1.6.1/triangle/triangle.c" ],
    include_dirs = [ numpy.get_include(),
                     "deps/pytriangle-1.6.1/triangle",
                     ],
    define_macros = DEFINES,
)




ext_modules = [bitmap, shape, tree, tessellator, render, pytriangle]
#ext_modules = [tessellator]

data_files = []
options = {}
package_data = {
    'maproom': [
        'renderer/gl/font.png',
        'templates/*.bna',
        'icons/*',
        ],
}

install_requires = [
    'numpy',
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
    'sawx>=1.3.0',
    'pillow',
    'urllib3',
    'certifi',
    'chardet',
    'idna',
    'packaging',
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
    packages=find_packages(),
    package_data=package_data,
    ext_modules=ext_modules,
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
