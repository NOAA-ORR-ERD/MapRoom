#!/usr/bin/env python

"""
setup.py for MapRoom -- mostly to build the Cython extensions

right now - it mostly is to be used as follows:

python setup.py build_ext --inplace

"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

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

print gl_include_dirs
print gl_library_dirs
print gl_libraries

# Definintion of compiled extension code:
bitmap = Extension("library.Bitmap",
                   sources=["library/Bitmap.pyx"],
                   include_dirs=[numpy.get_include()],
                   )

shape = Extension("library.Shape",
                  sources=["library/Shape.pyx"],
                  include_dirs=[numpy.get_include()],
                  )

tree = Extension("library.scipy_ckdtree",
                 sources=["library/scipy_ckdtree.pyx"],
                 include_dirs=[numpy.get_include()],
                 )

tessellator = Extension("library.Opengl_renderer.Tessellator",
                        sources=["library/Opengl_renderer/Tessellator.pyx"],
                        include_dirs=gl_include_dirs,
                        library_dirs=gl_library_dirs,
                        libraries=gl_libraries,
                        )

render = Extension("library.Opengl_renderer.Render",
                   sources=["library/Opengl_renderer/Render.pyx"],
                   include_dirs=gl_include_dirs,
                   library_dirs=gl_library_dirs,
                   libraries=gl_libraries,
                   )

ext_modules = [bitmap, shape, tree, tessellator, render]
#ext_modules = [tessellator]

# setup to build
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
