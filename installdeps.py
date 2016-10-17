#!/usr/bin/env python
import subprocess
import os
import sys

deps = [
    [".", 'pytriangle-1.6.1', 'python setup.py install'],
    ['https://github.com/NOAA-ORR-ERD/GnomeTools.git', 'post_gnome',],
    ['https://github.com/robmcmullen/OWSLib.git',],
    ['https://github.com/fathat/glsvg.git',],
    ['https://github.com/robmcmullen/pyugrid.git',],
]

using_conda = "Continuum Analytics" in sys.version or "conda" in sys.version
needs_netcdf = not sys.platform.startswith("win")

if needs_netcdf and not using_conda:
    # extra stuff isn't available through pypi or not easily built by hand
    deps.extend([
        ['https://github.com/MacPython/gattai.git',],
        ['https://github.com/robmcmullen/mac-builds.git', 'packages/netCDF4', 'gattai netcdf.gattai'],
        ])

topdir = os.path.join(os.getcwd(), "deps")

for dep in deps:
    os.chdir(topdir)
    repourl = dep[0]
    if repourl.startswith("http"):
        print "UPDATING %s" % repourl
        _, repo = os.path.split(repourl)
        repodir, _ = os.path.splitext(repo)
        if os.path.exists(repodir):
            os.chdir(repodir)
            subprocess.call(['git', 'pull'])
        else:
            subprocess.call(['git', 'clone', repourl])
    else:
        repodir = repourl
    if len(dep) == 1:
        subdir = "."
        command = "python setup.py develop"
    elif len(dep) == 2:
        subdir = dep[1]
        command = "python setup.py develop"
    else:
        subdir = dep[1]
        command = dep[2]
    os.chdir(topdir)
    os.chdir(repodir)
    os.chdir(subdir)
    subprocess.call(command.split())
