=========
Maproom 3
=========


Prerequisites
=============

This combination of library versions is known to work:

$ pip install numpy==1.7.2 PyOpenGL==3.0.2 PyOpenGl_accelerate==3.0.2 GDAL==1.10.0


Usage
=====

Requires the peppy2 framework, which in turn requires Enthought and its slew
of dependencies.

To run, add the maproom plugin to the peppy2 framework by:

python setup.py develop

and then run the peppy2 framework by

peppy2 TestData/Verdat/000011.verdat
