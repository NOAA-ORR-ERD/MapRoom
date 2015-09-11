=========
Maproom 3
=========


Prerequisites
=============

Easily installable stuff
------------------------

This combination of library versions is known to work::

    pip install numpy==1.9.2 PyOpenGL==3.1.0 pyproj==1.9.4 Cython==0.22.1
    # pip install PyOpenGL_accelerate==3.1.0

PyOpenGL_accelerate is currently not used on all platforms because a paint
event is apparently being triggered before the window is realized on screen,
or something similar.  It seems like the GLCanvas isn't fully initialized,
perhaps? Not sure, but the workaround for the moment is just to not use
PyOpenGL_accelerate if this error occurs on a particular platform.

Platform library dependencies
-----------------------------

Loading and saving triangle meshes requires pyugrid.  This requires NetCDF4
which in turn requires hdf5 support, neither of which is directly buildable
using pip.  The package gattai supports building library dependencies, so the
following steps are required::

    git clone https://github.com/MacPython/gattai.git
    cd gattai
    python setup.py install
    cd ..
    git clone https://github.com/MacPython/mac-builds.git
    cd mac-builds/packages/netCDF4
    gattai netcdf.gattai


NOTE: GCC 4.9 isn't supported in the configuration for hdf5, so I had to manually edit the file mac-builds/packages/netCDF4/hdf5-1.8.11/config/gnu-flags after a failed compile and restart gattai::

    --- config/gnu-flags~   2015-08-08 08:15:46.592158772 -0700
    +++ config/gnu-flags    2015-08-08 08:15:52.088158628 -0700
    @@ -189,7 +189,7 @@
     # Closer to the gcc 4.8 release, we should check for additional flags to
     # include and break it out into it's own section, like the other versions
     # below. -QAK
    -  gcc-4.[78]*)
    +  gcc-4.[789]*)
         # Replace -ansi flag with -std=c99 flag
         H5_CFLAGS="`echo $H5_CFLAGS | sed -e 's/-ansi/-std=c99/g'`"


netcdf if needed to be compiled manually::

    ./configure --prefix=/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild --disable-shared CFLAGS="-I/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild/include -fPIC" CXXFLAGS="-I/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild/include -fPIC" LDFLAGS=-L/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild/lib prefix=/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild LIBS=-ldl\
    make
    make install

Manual build of netcdf4-python::

    # On ubuntu, libcurl.so isn't properly installed, so need to manually link
    # sudo ln -s libcurl.so.4 /usr/lib/x86_64-linux-gnu/libcurl.so
    tar xfvz netCDF4-1.0.4.tar.gz
    cd netCDF4-1.0.4/
    python ../patch_setup.py
    python setup_static.py install


GEOS and Shapely
----------------

On OS X, the `Homebrew package manager <http://brew.sh/>`_ is required to install the GEOS dependency::

    brew install geos

Other platforms need the appropriate libgeos_c file installed in a library
directory.  Then, install Shapely::

    pip install shapely


GDAL
----

GDAL must be built by hand, after numpy is installed::

    cd gdal-1.11.2/
    ./configure --prefix=/data/virtualenv/wx3
    make -j3
    make install
    cd swig/python/
    python setup.py install

wxPython
--------

On ubuntu, wxPython fails to compile the python modules due to some formatting
warnings being treated as errors.  Changing the CFLAGS is required::

    CFLAGS=-Wno-error=format-security CPPFLAGS=-Wno-error=format-security python setup.py install

OWSLib
------

Currently, WMS 1.3.0 support is experimental and only on an unofficial branch
of the OWSLib official repository.  I have forked the repo and fixed the
1.3.0 support to handle the nonstandard servers that we deal with, so use this
command to clone my repository::

    git clone https://github.com/robmcmullen/OWSLib.git owslib130





Usage
=====

Requires the peppy2 framework, which in turn requires Enthought and its slew
of dependencies.

To run, add the maproom plugin to the peppy2 framework by::

    python setup.py develop

and then run the peppy2 framework by::

    peppy2 TestData/Verdat/000011.verdat


Building redistributable versions
=================================

The setup.py script has the ability to build py2exe and py2app bundles.

py2app
------

Debugging py2app:

* ./dist-3.X/mac/Maproom.app/Contents/MacOS/Maproom

will display stdout to the terminal

py2exe
------

NOTE: Don't install any packages as eggs (zip files).  py2exe can't include
dependencies correctly if they are inside eggs.

