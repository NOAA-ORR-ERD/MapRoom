#!/usr/bin/env python

import os, sys, platform
import numpy
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

## I found these by going through the Makefile that comes with triangle
## fixme -- should we test if running in single precision is faster?
##          according to the Triangle docs, it may be slower.
##          but its more memory to push around...

#~ def HAVE_HYPOT_Patch(filename):
    #~ """
    #~ This is a hack to get around a bug in Cython 0.14
     #~ -- the bug has been fixed, but the fix has not yet gotten into a release.

    #~ This may be required for other Cython files, so perhaps it should live elsewhere.
    
    #~ This should be unnecessary when we update to a newer release

    #~ """
    #~ print "Monkey patching %s for HAVE_HYPOT bug"
    #~ contents = file(filename).read()
    #~ contents = contents.replace("#if HAVE_HYPOT", "#ifdef HAVE_HYPOT")
    #~ file(filename, 'w').write(contents)

# don't need this with post 0.14 cython
# HAVE_HYPOT_Patch("src/pytriangle.c")
    
DEFINES = [("TRILIBRARY", None), # this builds Triangle as a lib, rather than as a command line program.
           ("NO_TIMER", None), # don't need the timer code (*nix specific anyway)
           ("REDUCED", None),
           ]

# Add the defines for disabling the FPU extended precision           ] 
## fixme: this needs a lot of work!
##        it's really compiler dependent, not machine dependent
if sys.platform == 'darwin':
    print "adding no CPU flags for mac"
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
    print "adding define for Windows for FPU management"
    DEFINES.append(('CPU86', None))
elif 'linux' in sys.platform :#  something for linux here...
    print "adding CPU flags for Intel Linux"
    DEFINES.append(('LINUX', None))
else:
    raise RuntimeError("this system isn't supported for building yet")

pytriangle = Extension(
    "pytriangle",
    sources = [ "src/pytriangle.pyx",
                "triangle/triangle.c" ],
    include_dirs = [ numpy.get_include(),
                     "triangle",
                     ],
    define_macros = DEFINES,
)

setup(
    name = "PyTriangle",
    version = "1.6.1",
    description = "A python wrapper for A two-dimensional quality mesh generator and Delaunay triangulator",
    author = ["Jonathan Richard Shewchuk", "Dan Helfman", "Chris Barker"],
    cmdclass = {'build_ext': build_ext}, # fixme: this doesn't appear to be working for running Cython
    packages = find_packages(),
    zip_safe = False,
    ext_modules = [ pytriangle ],
)


