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

    conda install numpy pillow pytest-cov cython docutils markdown requests configobj netcdf4 reportlab python-dateutil gdal pyproj shapely pyopengl wxpython owslib scipy

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

TODO


Code Architecture
=====================

TODO

