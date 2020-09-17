#!/usr/bin/env python
"""Calculate dependencies for MapRoom (and the Omnivore framework)

NOTE! When debugging, this must be run from the directory level above
pyinstaller so the script can find the symlinked dependency
directories.
"""
DEBUG = False

import os
import sys

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.utils.hooks import string_types, get_package_paths, exec_statement

import logging
logger = logging.getLogger(__name__)

# Anaconda distributions use pythonw if the app needs access to the screen
# and some of the imports will trigger this. So we have to hack the exec
# functions in PyInstaller.compat to use pythonw instdea of plain ol python.
if "CONDA_PREFIX" in os.environ:
    logger.debug("Using pythonw in this hook for conda support")
    save_exec = sys.executable
    sys.executable = "pythonw"
else:
    save_exec = None


subpkgs = [
    "maproom",
]

hiddenimports = ["netCDF4.utils", "cftime"]
for s in subpkgs:
    hiddenimports.extend(collect_submodules(s))

if DEBUG:
    print("\n".join(sorted(hiddenimports)))

subpkgs = [
    "maproom",
    "osgeo",
]

datas = []
skipped = []
maproom_allowed = set([
    "templates/RNCProdCat_latest.bna",
    "templates/blank_project.maproom",
    "templates/default_project.maproom",
    "templates/maproom.project.default_layout",
    "renderer/gl/font.png",
    ])
for s in subpkgs:
    possible = collect_data_files(s)
    # Filter out stuff.  Handle / and \ for path separators!
    for src, dest in possible:
        include = True
        pathcheck = src.replace("\\", "/")
        if src.endswith(".pyx") or src.endswith(".c") or src.endswith(".h") or src.endswith(".orig") or src.endswith(".sav"):
            include = False
        elif "maproom/maproom/"in pathcheck:
            _, mpath = pathcheck.split("maproom/maproom/", 1)
            if mpath in maproom_allowed or mpath.startswith("icons/"):
                include = True
            else:
                include = False

        if include:
            datas.append((src, dest))
        else:
            skipped.append((src, dest))

if DEBUG:
    print("\n".join(["%s -> %s" % d for d in datas]))
    print("SKIPPED:")
    print("\n".join(["%s -> %s" % d for d in skipped]))

# Restore sys.executable if changed because using pythonw causes failures
# further along in the process, and this is the only place where this hack
# is needed.
if save_exec is not None:
    sys.executable = save_exec

# _gdal shared library is not found (see pyintaller issue
# https://github.com/pyinstaller/pyinstaller/issues/1522), so this hack adds it
# to the build directory. On windows, it adds it as _gdal.cp36-win_amd64.pyd
# instead of just _glad.pyd, but it seems to work anyway.
binaries = []
from osgeo import gdal
from imp import get_suffixes
gdal_folder = os.path.abspath(os.path.dirname(gdal.__file__))
for suffix in get_suffixes():
    if suffix[2] == 3:  # C_EXTENSION
        gdal_pyd_path = os.path.join(gdal_folder, '_gdal%s' % suffix[0])
        if os.path.exists(gdal_pyd_path):
            binaries = [(gdal_pyd_path, "."), ]
