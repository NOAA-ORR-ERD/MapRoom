=========
MapRoom 5
=========

A product of the Emergency Response Division of the `NOAA <http://www.noaa.gov/>`_ `Office of
Response and Restoration <http://response.restoration.noaa.gov/>`_.
Visit the `Response And Restoration Blog
<https://usresponserestoration.wordpress.com/>`_ to see some `examples of
MapRoom <https://usresponserestoration.wordpress.com/2015/12/16/on-the-hunt-for-shipping-containers-lost-off-california-coast/>`_
in use.

The newest versions for Mac OS X and Windows are on the `download page <https://gitlab.orr.noaa.gov/erd/MapRoom/wikis/downloads>`_.


Installation
============

Package management is through conda. Download the
`Miniconda binary installer <http://conda.pydata.org/miniconda.html>`_ and run it
to install the base environment. On Windows, open an **Anaconda Prompt** from the start menu.

Configure the conda global environment with::

    conda config --add channels conda-forge
    conda config --add channels NOAA-ORR-ERD
    conda install conda-build

Create and populate the MapRoom virtual environment with::

    conda create --name maproom-test python=3.8
    activate maproom-test

Install the dependencies that conda can install::

    conda install numpy pillow pytest-cov cython docutils markdown requests configobj netcdf4 reportlab python-dateutil gdal pyproj shapely pyopengl wxpython owslib scipy pyugrid

and the dependencies that aren't in conda::

    pip install sawx omnivore-framework

Additionally, on MacOS only, install the ``pythonw`` command that allows programs to use GUI frameworks (like wxPython)::

    conda install python.app


Usage
=====

Install the source code if you have not already::

    git clone git@gitlab.orr.noaa.gov:erd/MapRoom.git
    cd maproom

To develop, MapRoom must be installed as it uses entry points than must be registered with
the python interpreter::

    python setup.py develop

and then run MapRoom by::

    python maproom TestData/Verdat/000011.verdat


Building redistributable versions
=================================

MapRoom uses pyinstaller to build standalone/redistributable binary versions.

I (Rob McMullen) have not yet been successful creating pyinstaller bundles
using conda. I have been able to build pyinstaller bundles using pip virtual
environments, but this requires some by-hand building of some major
dependencies: GEOS and GDAL. There are notes on the wiki for both MacOS and
Windows:

* https://gitlab.orr.noaa.gov/erd/MapRoom/-/wikis/dev/MacOS:-building-app-bundles-without-conda
* https://gitlab.orr.noaa.gov/erd/MapRoom/-/wikis/dev/Windows-10:-building-app-bundles-without-conda

There is a script in the ``maproom/pyinstaller`` directory called
``build_pyinstaller.py`` that includes the configuration data to create a
bundle. On a non-conda install, this creates a working app bundle.

On a conda install, the operation to create the bundle completes successfully
and creates the application ``maproom/pyinstaller/dist/MapRoom_build.app``.
However, running this fails with a crash dialog box.

Trying to run executable the unpacked version in the
``maproom/pyinstaller/dist/MapRoom_build`` directory results in::

    $ ./MapRoom_build
    This program needs access to the screen. Please run with a
    Framework build of python, and only when you are logged in
    on the main display of your Mac.

Even using the ``--windowed`` flag to pyinstaller results in this same error.

Some references:

* https://github.com/chriskiehl/Gooey/issues/259

Debugging pyinstaller problems is very tedious, as it is difficult to get
error messages. On a non-conda install, running the application out of the
build folder would send error messages to the screen, but on a conda install
it doesn't get far enough because it can't seem to do the equivalent of the
the pythonw command.


Top-Level Directory Layout
=================================

TestData
------------------------

This directory contains many subdirectories of example data; see the
00-README.txt in that directory for more details.

libmaproom
-------------------

All the C and Cython code is contained in this directory, and it can be
compiled with a single call to ``python setup.py develop`` inside this
directory.


maproom
---------------

The pure-python code for MapRoom is contained in this directory.


tests
----------

The test framework, using ``pytest``, is contained in this directory. Changing
to this directory and running ``py.test`` will execute the test suite.

scripts
----------

* ``maproom.py`` -- The main script used to start MapRoom

* ``setup.py`` -- python install script

* ``release-and-tag-new-version.sh`` -- helper script to create a new version
  of the code (updating the version number in the code, tagging a new version
  in git, creating a ChangeLog, and building a new version using pyinstaller).
  This would be run on MacOS to create the new version number & tag it in git,
  and then the resulting repository would be checked out on a Windows machine
  to use pyinstaller there to create the executable for Windows.

* ``make-changelog.py`` -- helper script used by the above script to generate
  and update the ChangeLog


Code Architecture - libmaproom
===================================

The libmaproom directory contains a separate python package that includes all
the Cython and C code used by MapRoom. There are 6 modules, 4 of which are
used directly by MapRoom to help accelerate rendering. The other 2 are
standalone modules for accelerating specific tasks: pytriangle for creating
triangular meshes, and py_contour for creating contours of particle layers.

libmaproom/libmaproom/*.pyx files
--------------------------------------

The 4 Cython files (.pyx) are helpers for OpenGL rendering.

libmaproom/libmaproom/py_contour/
--------------------------------------

This is a copy of the py_contour code found `here
<https://github.com/NOAA-ORR-ERD/py_contour>`_. There are no changes to the
code, it is just included here to streamline the install and development
process.

libmaproom/libmaproom/pytriangle-1.6.1/
-------------------------------------------

This is an implementation of `Richard Shewchuk's Triangle library
<http://www.cs.cmu.edu/~quake/triangle.html>`_ that is used for mesh
generation. It is Cython code, consisting of a Cython file
``libmaproom/libmaproom/pytriangle-1.6.1/src/triangle.pyx`` and Shewchuk's
original C source in the ``libmaproom/libmaproom/pytriangle-1.6.1/triangle/``
directory.

The ``triangle.pyx`` file is divided into two python functions, where
``triangulate_simple`` is the function designed to be called from user code,
where it uses the Multiprocess package to call ``triangulate_simple_child``
(which wraps the Shewchuk C code). If the C code were not run in another
process, it could kill the entire program as the C code uses the ``exit()``
system call.




Code Architecture - maproom
===================================

The MapRoom program is started using the ``maproom.py`` script in the top
level directory. It contains the ``MapRoomApp`` class and the ``main``
function that is the driver for the whole program. The ``get_image_path`` call
to determine paths for icons and other things is used here because bundled
apps (using pyinstaller) can have different locations for code and resource
data.


Application Init
----------------------

The UI is built using the ``maproom.app_framework`` utilities. Its classes use
the ``Maf`` prefix.

The application, ``MafApplication``, wraps the wx.App class. Its ``OnInit``
method sets up some initial data and event handling, but the main application
start occurs in the ``process_command_line_args`` method. This routine is
responsible for creating the first ``MafFrame`` window.

If no file is not specified on the command line, a default project will be
used. The command line supports loading a project file or a layer file; if a
layer only is specified, a default project will be created and the layer
loaded into that project. The ``MafFrame.load_file`` method is used to load
whichever type of file is specified, and once loaded the frame will be created
and displayed.


File Load
--------------------

Projects or files are loaded using the ``MafFrame.load_file`` method. The file
is identified through the ``maproom.app_framework.loader.identify_file``
routine to determine the loader that can parse the data, and the loader
creates the layers that are used for display.

At the start of the ``identify_file`` routine, the file data is loaded into
``maproom.app_framework.loader.FileGuess`` class instance that supplies
convenience functions for accessing the data in the file. It then loops over
every loader to find the best match. Each loader can test the data using
convenience methods of the FileGuess class without having to read the file
over again.

Loaders are registered as setuptools plugins with the entry point
"maproom.app_framework.loaders". Loaders are modules that implement a
module-level function called ``identify_loader`` that returns a dictionary
containing the MIME type and loader class that can handle the file, or None if
the loader can't handle that file.

The ``identify_file`` routine returns the "best" loader if an exact match is
found, or tries to supply a generic loader as a fallback.

At this point, the code is back in the ``MafFrame.load_file`` method with a
dictionary called ``file_metadata`` containing the loader class and the
FileGuess object. Here is where the difference between a project load and a
layer load is handled: if the attempted load is a project, the call to
``MafEditor.can_load_file`` will return false and a new document will be
created. (A document corresponds to a tab in the user interface.)  If the file
to be loaded can be represented as single layer (or group of layers under a
single layer like a NetCDF particles file), the layer will be added to the
current project.


Document Load
------------------
