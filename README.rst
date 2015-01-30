=========
Maproom 3
=========


Prerequisites
=============

Easily installable stuff
------------------------

This combination of library versions is known to work:

pip install numpy==1.7.2 PyOpenGL==3.0.2 pyproj==1.9.3 GDAL==1.10.0 Cython==0.19.2
# pip install PyOpenGL_accelerate==3.0.2

PyOpenGL_accelerate is currently not used because a paint event is apparently
being triggered before the window is realized on screen, or something similar.
It seems like the GLCanvas isn't fully initialized, perhaps? Not sure, but
the workaround for the moment is just to not use PyOpenGL_accelerate.

Platform library dependencies
-----------------------------

Loading and saving triangle meshes requires pyugrid.  This requires NetCDF4
which in turn requires hdf5 support, neither of which is directly buildable
using pip.  The package gattai supports building library dependencies, so the
following steps are required:

git clone https://github.com/MacPython/gattai.git
cd gattai
python setup.py install
cd ..
git clone https://github.com/MacPython/mac-builds.git
cd mac-builds/packages/netCDF4
gattai netcdf.gattai
cd ../../..
git clone https://github.com/pyugrid/pyugrid.git
cd pyugrid
python setup install




netcdf:

./configure --prefix=/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild --disable-shared CFLAGS=-I/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild/include -fPIC CXXFLAGS=-I/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild/include -fPIC LDFLAGS=-L/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild/lib prefix=/data/virtualenv/wx3/src/mac-builds/packages/DepsBuild LIBS=-ldl



GEOS and Shapely
----------------

brew install geos
pip install shapely



Usage
=====

Requires the peppy2 framework, which in turn requires Enthought and its slew
of dependencies.

To run, add the maproom plugin to the peppy2 framework by:

python setup.py develop

and then run the peppy2 framework by

peppy2 TestData/Verdat/000011.verdat


Building redistributable versions
=================================

The setup.py script has the ability to build py2exe and py2app bundles.

py2app
------

Debugging py2app:

* ./dist-3.X/mac/Maproom.app/Contents/MacOS/Maproom

will display stdout to the terminal
