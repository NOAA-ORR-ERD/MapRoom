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


Code Architecture
=====================

TODO

