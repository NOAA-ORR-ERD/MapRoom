Maproom
=======

The following dependencies are required for running Maproom:

  * Python 2.6
  * setuptools 0.6
  * Maproomlib
  * wxPython 2.8.9.2+ unicode
  * NumPy 1.x
  * PyOpenGL 3.x
  * greenlet 0.2 or py lib 0.9
  * GDAL 1.6.x (1.6.3+ highly recommended, due to bugs in earlier releases)
  * pyproj 1.8.x
  * PROJ.4 4.x
  * graphics chipset and drivers supporting OpenGL with 24/32-bit color depth

The following dependencies are optional:

  * SciPy 0.7.x: Only needed to use the duplicate point tool or contour.
  * PyOpenGL-accelerate: Only needed to make PyOpenGL somewhat faster.
  * GDAL built with OGDI support: Only needed to read NGA DNC files.
  * py2exe: Only needed to build an installer for Windows.
  * py2app: Only needed to build an installer for Mac OS X.
  * nose: Only needed to run unit tests.
  * Sphinx: Only needed to build documentation.

Note that many of the dependencies are available in source form within the
separate "libs" repository. Some are also pre-built on the build server.

To launch Maproom, run:

  python maproom.py


Linux Installation
==================

On Ubuntu Karmic, you can install the dependencies as follows:

  * On your Ubuntu computer, open System > Administration > Software Sources.
  * Click the "Other Software" tab.
  * Click the "Add..." button.
  * Copy and paste the following line into the "APT line:" text field:

  deb http://ppa.launchpad.net/ubuntugis/ubuntugis-unstable/ubuntu karmic main 

  * Click the "Add Source" button.
  * Click the "Close" button.
  * Click the "Reload" button when prompted.

Open a terminal and enter the following commands:

  sudo aptitude install python-setuptools python-wxgtk2.8 python-numpy \
                        libgdal1-1.6.0 python-gdal python-pyproj python-dev \
                        libglu1-mesa-dev
  sudo ln -s /usr/lib/libproj.so.0 /usr/lib/libproj.so
  sudo easy_install pyopengl pyopengl-accelerate greenlet
  python setup.py build_ext -i


Unit Tests
==========

To execute the unit tests, run:

  nosetests --exe --with-coverage --cover-inclusive --cover-package=maproom

You'll need the Python nose and coverage packages installed to run unit tests.


Building Installers
===================

To build a Mac OS X disk image on Mac OS X, change to the "Maproom" directory
(the one containing "maproomlib") and run:

  python setup.py py2app
  ./BuildDiskImage

You'll need the Python py2app package installed in order to build the bundle.
These commands will produce a disk image (dmg) in the current directory.

To build a Windows installer on Windows, change to the "Maproom" directory
(the one containing "maproomlib") and run:

  python setup.py py2exe

You'll need the Python py2exe package installed in order for this to work.
This command will produce an installer executable within the dist/Output/
directory.

Note that the setup.py defaults to using Visual Studio as the compiler on
Windows. If you'd like to use MinGW instead, then replace "msvc" with
"mingw32" in setup.py


Building Documentation
======================

To build the documentation, run:

  PYTHONPATH=.. sphinx-build doc/source doc/build

You'll need the Python sphinx package installed to build documentation.
