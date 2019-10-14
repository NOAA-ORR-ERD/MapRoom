**********
py_contour
**********

Computing contour polygons from data on a rectangular grid.

Cython wrappers around a C library contouring code found here:

http://paulbourke.net/papers/conrec/

Generates contour lines and polygons from recatagular grid of data.

Note that the C lib generates only line segments, which is great for drawing contour lines, but if you need to fill polygons, the code also sorts the line segments into polygons and polylines (for non-closed contours)

Installation
============

Dependencies
------------

Run Time:

* numpy

Built Time:

* setuptools'
* numpy
* cython>0.23
* pytest

Building
--------

I don't currently have binary wheels available, so you need to have a properly configured C complier, etc.

Otherwise, download the source code from gitHub, either as a clone or source tarball::

  pip install ./

should do it. or::

  pip install -e ./

For "editable mode"

Regular old::

  setup.py install

Should work too, but then you get that easy-install garbage scattered around.

Testing
-------

You should be able to run the tests from source with::

  python setup.py test

Or run the installed tests with::

  py.test --pyargs py_contour 






