#!/usr/bin/env python

"""
setup.py for libmaproom to build python extensions

right now - it mostly is to be used as follows:

python setup.py build_ext --inplace

"""
import sys

from Cython.Distutils import build_ext
from setuptools import setup, Extension
from ctypes.util import find_library


version_path = "../maproom/_version.py"
exec(compile(open(version_path).read(), version_path, 'exec'))

# find the various headers, libs, etc.
import numpy
gl_libraries = []
gl_include_dirs = [numpy.get_include()]
gl_library_dirs = []
extra_link_args = []

if sys.platform.startswith("win"):
    gl_libraries = ["opengl32", "glu32"]
elif sys.platform == "darwin":
#    gl_libraries = ["GL", "GLU"]

    extra_link_args.extend(("-framework", "OpenGL"))

    # gl_include_dirs.append(
    #     "/System/Library/Frameworks/OpenGL.framework/Headers",
    # )
    # gl_library_dirs.append(
    #      # find_library("OpenGL"),
    #      "/System/Library/Frameworks/OpenGL.framework/"
    #      # "/System/Library/Frameworks/OpenGL.framework/Libraries",
    # )

# Definition of compiled extension code:
bitmap = Extension("libmaproom.Bitmap",
                   sources=["libmaproom/Bitmap.pyx"],
                   include_dirs=[numpy.get_include()],
                   extra_compile_args = ["-O3" ],
                   )

shape = Extension("libmaproom.Shape",
                  sources=["libmaproom/Shape.pyx"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args = ["-O3" ],
                  )

tessellator = Extension("libmaproom.Tessellator",
                        sources=["libmaproom/Tessellator.pyx"],
                        include_dirs=gl_include_dirs,
                        library_dirs=gl_library_dirs,
                        libraries=gl_libraries,
                        extra_link_args=extra_link_args,
                        extra_compile_args = ["-O3" ],
                        )

render = Extension("libmaproom.Render",
                   sources=["libmaproom/Render.pyx"],
                   include_dirs=gl_include_dirs,
                   library_dirs=gl_library_dirs,
                   libraries=gl_libraries,
                   extra_link_args=extra_link_args,
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
    ##     the precision is always determined by the largest operand type in C.
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
    "libmaproom.pytriangle",
    sources = [ "libmaproom/pytriangle-1.6.1/src/pytriangle.pyx",
                "libmaproom/pytriangle-1.6.1/triangle/triangle.c" ],
    include_dirs = [ numpy.get_include(),
                     "libmaproom/pytriangle-1.6.1/triangle",
                     ],
    define_macros = DEFINES,
)

# Extension module is named libmaproom.contour rather than libmaproom.py_contour
# because imports get confused with the py_contour source directory within the libmaproom
# directory (error message "dynamic module doesn't define module export function"), but
# moving the py_contour directory outside of the libmaproom directory doesn't seem to help.
py_contour = Extension(
    "libmaproom.contour",
    sources = [
        "libmaproom/py_contour/py_contour/contour.pyx",
        "libmaproom/py_contour/py_contour/conrec.cxx",
    ],
    include_dirs = [
        numpy.get_include(),
    ],
)


ext_modules = [bitmap, shape, tessellator, render, pytriangle, py_contour]
#ext_modules = [tessellator]

data_files = []
options = {}
package_data = {}

install_requires = [
    'numpy',
    'pyopengl',
    'cython',
]

setup(
    name="libmaproom",
    version=__version__,
    description="Compiled support libraries for MapRoom",
    author="NOAA",
    install_requires=install_requires,
    setup_requires=[
        'packaging',
    ],
    data_files=data_files,
    packages=['libmaproom'],
    package_data=package_data,
    ext_modules=ext_modules,
    options=options,
    zip_safe = False,
)
