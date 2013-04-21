Building Extensions
===================

To build required extensions, run from the directory containing setup.py:

  python setup.py build_ext -i

In order for this to work, you'll need a working C compiler, along with header
and development libraries for OpenGL.

Note that the setup.py defaults to using Visual Studio as the compiler on
Windows. If you'd like to use MinGW instead, then replace "msvc" with
"mingw32" in setup.py


Installation
============

To install maproomlib, run:

  python setup.py install


Unit Tests
==========

To execute the unit tests, run:

  nosetests --exe --with-coverage --cover-inclusive --cover-package=maproomlib

You'll need the Python nose and coverage packages installed to run unit tests.


Building Documentation
======================

To build the documentation, run:

  PYTHONPATH=.. sphinx-build doc/source doc/build

You'll need the Python sphinx package installed to build documentation.
