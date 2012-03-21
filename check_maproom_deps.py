#!/usr/bin/env python
"""
simple test script to see if all the dependencies are installed correctly
for MapRoom

"""

## here are all the deps:
## (module_name, version)
deps = [ ('numpy', '1.6.1'),
         ('wx', '2.8.12.1'), # wxPython
         ('OpenGL','3.0.1'),
         ('pyproj', '1.9.0'),
         ('osgeo', '1.9.0'), # gdal/ogr
         ('netCDF4', '0.9.9'),
         ('cython', ''),
        ]


for mod_name, version in deps:
    try:
        mod = __import__(mod_name)
        try:
            if mod.__version__ <> version:
                print "Module: %s imported, but is the wrong version"%mod_name
                print "need: %s, got %s"%(version, mod.__version__)
            else:
                print "Module: %s is properly installed"%mod_name
        except AttributeError:
            print "Module: %s is installed, but could not check the version"%mod_name
    except ImportError:
        print "module: %s could not be imported"%mod_name
        