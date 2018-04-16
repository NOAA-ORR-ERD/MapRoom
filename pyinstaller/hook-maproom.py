#!/usr/bin/env python
"""Calculate dependencies for MapRoom (and the Omnivore framework)

NOTE! When debugging, this must be run from the directory level above
pyinstaller so the script can find the symlinked dependency
directories.
"""
DEBUG = True

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

# Need special version for traitsui because it raises a RuntimeError when
# hitting the traitsui.qt package and doesn't scan any further.
def is_package(package):
    return True

def collect_submodules(package, filter=lambda name: True):
    # Accept only strings as packages.
    if not isinstance(package, string_types):
        raise ValueError

    logger.debug('Collecting submodules for %s' % package)
    # Skip a module which is not a package.
    if not is_package(package):
        logger.debug('collect_submodules - Module %s is not a package.' % package)
        return []

    # Determine the filesystem path to the specified package.
    pkg_base, pkg_dir = get_package_paths(package)

    # Walk the package. Since this performs imports, do it in a separate
    # process.
    names = exec_statement("""
        import sys
        import pkgutil

        def ignore_err(name, err):
            # Can't print anything because printing is captured as module names
            # print ("error importing %s: %s" % (name, err))
            pass

        # ``pkgutil.walk_packages`` doesn't walk subpackages of zipped files
        # per https://bugs.python.org/issue14209. This is a workaround.
        def walk_packages(path=None, prefix='', onerror=ignore_err):
            def seen(p, m={{}}):
                if p in m:
                    return True
                m[p] = True

            for importer, name, ispkg in pkgutil.iter_modules(path, prefix):
                if not name.startswith(prefix):   ## Added
                    name = prefix + name          ## Added
                yield importer, name, ispkg

                if ispkg:
                    try:
                        __import__(name)
                    except ImportError, e:
                        if onerror is not None:
                            onerror(name, e)
                    except Exception, e:
                        if onerror is not None:
                            onerror(name, e)
                        else:
                            raise
                    else:
                        path = getattr(sys.modules[name], '__path__', None) or []

                        # don't traverse path items we've seen before
                        path = [p for p in path if not seen(p)]

                        ## Use Py2 code here. It still works in Py3.
                        for item in walk_packages(path, name+'.', onerror):
                            yield item
                        ## This is the original Py3 code.
                        #yield from walk_packages(path, name+'.', onerror)

        for module_loader, name, ispkg in walk_packages([{}], '{}.'):
            print(name)
        """.format(
                  # Use repr to escape Windows backslashes.
                  repr(pkg_dir), package))

    # Include the package itself in the results.
    mods = {package}
    # Filter through the returend submodules.
    for name in names.split():
        if filter(name):
            mods.add(name)

    logger.debug("collect_submodules - Found submodules: %s", mods)
    return list(mods)

def qt_filter(pymod):
    if ".tests" in pymod or ".qt" in pymod or ".null" in pymod:
        logger.debug("qt_filter: skipping %s" % pymod)
        return False
    return True


subpkgs = [
    "traits",
    "traitsui",
    "traitsui.wx",
    "pyface",
    "omnivore",
    "maproom",
]

hiddenimports = ["netCDF4_utils"]
for s in subpkgs:
    hiddenimports.extend(collect_submodules(s, qt_filter))

if DEBUG:
    print "\n".join(sorted(hiddenimports))

subpkgs = [
    "traitsui",
    "pyface",
    "omnivore",
    "maproom",
]

datas = []
skipped = []
maproom_allowed = set([
    "templates/RNCProdCat_latest.bna",
    "templates/default_project.maproom",
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
    print "\n".join(["%s -> %s" % d for d in datas])
    print "SKIPPED:"
    print "\n".join(["%s -> %s" % d for d in skipped])

# Restore sys.executable if changed because using pythonw causes failures
# further along in the process, and this is the only place where this hack
# is needed.
if save_exec is not None:
    sys.executable = save_exec
