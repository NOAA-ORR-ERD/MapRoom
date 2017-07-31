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

print gl_include_dirs
print gl_library_dirs
print gl_libraries

# Definintion of compiled extension code:
bitmap = Extension("maproom.library.Bitmap",
                   sources=["maproom/library/Bitmap.pyx"],
                   include_dirs=[numpy.get_include()],
                   )

shape = Extension("maproom.library.Shape",
                  sources=["maproom/library/Shape.pyx"],
                  include_dirs=[numpy.get_include()],
                  )

tree = Extension("maproom.library.scipy_ckdtree",
                 sources=["maproom/library/scipy_ckdtree.pyx"],
                 include_dirs=[numpy.get_include()],
                 )

tessellator = Extension("maproom.renderer.gl.Tessellator",
                        sources=["maproom/renderer/gl/Tessellator.pyx"],
                        include_dirs=gl_include_dirs,
                        library_dirs=gl_library_dirs,
                        libraries=gl_libraries,
                        )

render = Extension("maproom.renderer.gl.Render",
                   sources=["maproom/renderer/gl/Render.pyx"],
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
