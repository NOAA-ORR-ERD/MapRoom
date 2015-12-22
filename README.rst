=========
MapRoom 3
=========

A product of the Emergency Response Division of the `NOAA <http://www.noaa.gov/>`_ `Office of
Response and Restoration <http://response.restoration.noaa.gov/>`_.
Visit the `Response And Restoration Blog
<https://usresponserestoration.wordpress.com/>`_ to see some `examples of
MapRoom <https://usresponserestoration.wordpress.com/2015/12/16/on-the-hunt-for-shipping-containers-lost-off-california-coast/>`_
in use.


Overview
========

* Install normal packages using pip
* Manual install:

  * wxPython
  * GDAL
  * GEOS and shapely
  
* Scripted install:

  * pytriangle
  * netcdf4-python
  * post_gnome
  * pyugrid
  * OWSLib


Normal pip install
==================

Note that on Mac, libjpeg must be installed before pillow using the `Homebrew package manager <http://brew.sh/>`_ ::

    brew install libjpeg

Other platforms do not need this dependency.  This combination of library
versions is known to work::

    pip install numpy==1.10.1 PyOpenGL==3.1.1a1 PyOpenGL_accelerate==3.1.1a1 pyproj==1.9.4 Cython==0.23.4 Pillow=3.0.0 reportlab=3.2.0 omnivore

PyOpenGL_accelerate is currently not used on all platforms because a paint
event is apparently being triggered before the window is realized on screen,
or something similar.  It seems like the GLCanvas isn't fully initialized,
perhaps? Not sure, but the workaround for the moment is just to not use
PyOpenGL_accelerate if this error occurs on a particular platform.

Manual Install
==============

wxPython
--------

On ubuntu, wxPython fails to compile the python modules due to some formatting
warnings being treated as errors.  Changing the CFLAGS is required::

    CFLAGS=-Wno-error=format-security CPPFLAGS=-Wno-error=format-security python setup.py install


GDAL
----

GDAL must be built by hand, after numpy is installed::

    cd gdal-1.11.3/
    ./configure --prefix=$VIRTUAL_ENV
    make -j3
    make install
    cd swig/python/
    python setup.py install


GEOS and Shapely
----------------

Linux users must install GEOS, typically through the package manager.  Shapely
can then be automatically installed using pip.

On OS X, GEOS can be installed using brew::

    brew install geos

and the dynamic library search path must be added to the environment (which can
be added automatically if you add this to a bash login file)::

    export DYLD_LIBRARY_PATH=/usr/local/Cellar/geos/<GEOS_VERSION>/lib

Other platforms need the appropriate libgeos_c file installed in a library
directory.  Then, install Shapely::

    pip install shapely


Scripted Install
================

The python script installdeps.py in the deps directory installs everything else
needed that can't be installed using pip directly.

OWSLib
------

Currently, WMS 1.3.0 support is experimental and only on an unofficial branch
of the OWSLib official repository.  I have forked the repo and fixed the
1.3.0 support to handle the nonstandard servers that we deal with.

pyugrid
-------

Loading and saving triangle meshes requires pyugrid.  This requires NetCDF4
which in turn requires hdf5 support, neither of which is directly buildable
using pip.  The package gattai supports building library dependencies, and the
gattai dependency is itself included in the installdeps.py script.

netcdf4-python
--------------

If a manual build of netcdf4-python is needed::

    # On ubuntu, libcurl.so isn't properly installed, so need to manually link
    # sudo ln -s libcurl.so.4 /usr/lib/x86_64-linux-gnu/libcurl.so
    tar xfvz netCDF4-1.0.4.tar.gz
    cd netCDF4-1.0.4/
    python ../patch_setup.py
    python setup_static.py install



Usage
=====

To run, maproom must be installed as it is a plugin to the omnivore framework::

    python setup.py develop

and then run maproom by::

    python maproom TestData/Verdat/000011.verdat


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
dependencies correctly if they are inside eggs. Pip can be forced to not use eggs by adding a distutils.cfg file in the C:/Python27/Lib directory containing::

    [easy_install]
    zip_ok = False
