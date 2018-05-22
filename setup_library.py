#!/usr/bin/env python

"""
setup.py for MapRoom -- mostly to build the Cython extensions

right now - it mostly is to be used as follows:

python setup_library.py build_ext --inplace

"""


from Cython.Distutils import build_ext
from setuptools import setup, find_packages, Extension

import sys

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

print(gl_include_dirs)
print(gl_library_dirs)
print(gl_libraries)

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

PYTRIANGLE_DEFINES = [
  ("TRILIBRARY", None), # builds as a lib, rather than a command line program.
  ("NO_TIMER", None), # don't need the timer code (*nix specific anyway)
  ("REDUCED", None),
]

# Add the defines for disabling the FPU extended precision
## fixme: this needs a lot of work!
##        it's really compiler dependent, not machine dependent
if sys.platform == 'darwin':
    print("adding no CPU flags for mac")
    ## according to: http://www.christian-seiler.de/projekte/fpmath/
    ## nothing special is required on OS-X !
    ##
    ## """
    ##     the precision is always determined by the largest operhand type in
    ##     C.
    ## 
    ##     Because of this, Mac OS X does not provide any C wrapper macros to
    ##     change the internal precision setting of the x87 FPU. It is simply
    ##     not necessary. Should this really be wanted, inline assembler would
    ##     probably be possible, I haven't tested this, however.
    ##     Simply use the correct datatype and the operations performed will
    ##     have the correct semantics
    ## """
elif sys.platform == 'win32':
    print("adding define for Windows for FPU management")
    PYTRIANGLE_DEFINES.append(('CPU86', None))
elif 'linux' in sys.platform :#  something for linux here...
    print("adding CPU flags for Intel Linux")
    PYTRIANGLE_DEFINES.append(('LINUX', None))
else:
    raise RuntimeError("this system isn't supported for building yet")

pytriangle = Extension("pytriangle",
                       sources = [
                           "deps/pytriangle-1.6.1/src/pytriangle.pyx",
                           "deps/pytriangle-1.6.1/triangle/triangle.c",
                           ],
                       include_dirs = [
                           numpy.get_include(),
                           "deps/pytriangle-1.6.1/triangle",
                           ],
                       define_macros = PYTRIANGLE_DEFINES,
)


ext_modules = [bitmap, shape, tree, tessellator, render, pytriangle]
#ext_modules = [tessellator]

# setup to build
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
