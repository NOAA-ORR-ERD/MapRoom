=========
Maproom 3
=========


Prerequisites
=============

This combination of library versions is known to work:

pip install numpy==1.7.2 PyOpenGL==3.0.2 pyproj==1.9.3 GDAL==1.10.0 Cython==0.19.2
# pip install PyOpenGL_accelerate==3.0.2

PyOpenGL_accelerate is currently not used because a paint event is apparently
being triggered before the window is realized on screen, or something similar.
It seems like the GLCanvas isn't fully initialized, perhaps? Not sure, but
the workaround for the moment is just to not use PyOpenGL_accelerate.

Usage
=====

Requires the peppy2 framework, which in turn requires Enthought and its slew
of dependencies.

To run, add the maproom plugin to the peppy2 framework by:

python setup.py develop

and then run the peppy2 framework by

peppy2 TestData/Verdat/000011.verdat
